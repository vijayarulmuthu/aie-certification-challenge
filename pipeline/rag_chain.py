"""
pipeline/rag_chain.py

This module defines an agentic Retrieval-Augmented Generation (RAG) pipeline using LangGraph.
It decomposes complex Bible-related queries into sub-questions, retrieves relevant verses using
contextual compression and re-ranking, then summarizes the context into a final answer.

Key components:
- Sub-query decomposition via an LLM
- Contextual retrieval with Cohere Rerank (v3.5)
- Graph-based orchestration using LangGraph
- Rich terminal output formatting for responses
"""

import re
import json

from config import settings
from typing_extensions import TypedDict, List
from pydantic import BaseModel

from langgraph.graph import START, END, StateGraph

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

from rich.console import Console
from rich.markdown import Markdown

console = Console()

# ──────────────────────────────────────────────────────────────
# STATE DEFINITION
# ──────────────────────────────────────────────────────────────
class State(TypedDict):
    question: str
    context: List[Document]
    response: str

class AgenticRAGState(TypedDict):
    query: str
    sub_questions: List[str]
    retrieval_results: List[dict]
    final_answer: str

# ──────────────────────────────────────────────────────────────
# MAIN RAG CHAIN CONSTRUCTION
# ──────────────────────────────────────────────────────────────
def build_rag_chain(vectorstore, streaming: bool = False):
    """
    Constructs a LangGraph-based RAG pipeline that performs:
    1. Query decomposition into sub-questions.
    2. Document retrieval using Cohere Rerank contextual compression.
    3. Chain-of-thought summarization across sub-questions.

    Args:
        vectorstore: LangChain-compatible vector store (e.g., QdrantVectorStore)
        streaming (bool): Whether to enable LLM streaming

    Returns:
        Compiled LangGraph StateGraph
    """

    # Define a schema for parsing sub-query JSON output
    class SubQueryResponse(BaseModel):
        subQuestions: List[str]

    def parse_subquery_response(response_text: str) -> SubQueryResponse:
        """
        Extract and validate JSON output of sub-questions from LLM response.

        Args:
            response_text (str): Raw LLM output containing JSON object

        Returns:
            SubQueryResponse: Parsed response with a list of sub-questions

        Raises:
            ValueError: If no valid JSON is found
        """
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON found in sub-query response.")
        data = json.loads(json_match.group())
        return SubQueryResponse(**data)

    # Setup LLM and retriever
    llm = ChatOpenAI(model=settings.OPENAI_GENERATION_MODEL, streaming=streaming)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # ────────────────
    # PROMPTS
    # ────────────────
    decompose_prompt = PromptTemplate.from_template("""
    You are a query decomposition assistant for a Bible question-answering system.
    Break the following multi-part question into simpler, logically independent sub-questions.
    Return a JSON object like:
    {{
        "subQuestions": ["..."]
    }}

    Question:
    {query}

    Output:
    """)

    summarize_prompt = PromptTemplate.from_template("""
    You are a Bible Scholar AI. Summarize the following retrieved verses by grouping them into thematic or book-based clusters.
    Use chain-of-thought reasoning to provide a coherent and faithful summary.

    Context:
    {docs}

    Answer:
    """)

    # ────────────────
    # GRAPH NODES
    # ────────────────

    def decompose_query(state: dict) -> dict:
        """
        Decomposes a user query into multiple sub-questions using an LLM.

        Args:
            state (dict): Current state containing the main query.

        Returns:
            dict: Updated state with sub_questions
        """
        query = state["query"]
        messages = [HumanMessage(content=decompose_prompt.format(query=query))]
        result = llm.invoke(messages).content
        sub_questions = parse_subquery_response(result).subQuestions
        return {**state, "sub_questions": sub_questions}

    def retrieve_context(state: dict) -> dict:
        """
        Retrieves relevant context for each sub-question using contextual compression.

        Args:
            state (dict): State containing sub-questions

        Returns:
            dict: Updated state with retrieval results
        """
        results = []
        compressor = CohereRerank(model="rerank-v3.5", top_n=10)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever,
            search_kwargs={"k": 5},
        )

        for q in state["sub_questions"]:
            docs = compression_retriever.invoke(q)
            results.append({"question": q, "docs": docs})

        return {**state, "retrieval_results": results}

    def summarize_results(state: dict) -> dict:
        """
        Summarizes retrieved passages into a coherent answer.

        Args:
            state (dict): State containing retrieval results

        Returns:
            dict: Updated state with `final_answer`
        """
        combined_docs = []
        for entry in state["retrieval_results"]:
            for doc in entry["docs"]:
                combined_docs.append(doc.page_content)

        joined = "\n\n".join(combined_docs)
        summary = llm.invoke([HumanMessage(content=summarize_prompt.format(docs=joined))]).content
        return {**state, "final_answer": summary}

    # ────────────────
    # GRAPH DEFINITION
    # ────────────────
    graph_builder = StateGraph(AgenticRAGState)

    graph_builder.add_node("decompose", RunnableLambda(decompose_query))
    graph_builder.add_node("retrieve", RunnableLambda(retrieve_context))
    graph_builder.add_node("summarize", RunnableLambda(summarize_results))

    graph_builder.set_entry_point("decompose")
    graph_builder.add_edge("decompose", "retrieve")
    graph_builder.add_edge("retrieve", "summarize")
    graph_builder.add_edge("summarize", END)

    return graph_builder.compile()

# ──────────────────────────────────────────────────────────────
# FORMATTED TERMINAL PRINTING
# ──────────────────────────────────────────────────────────────
def print_rag_response(response: dict):
    """
    Pretty-prints a full RAG response to the terminal using rich.

    Args:
        response (dict): Final output state from the graph execution
    """
    console.rule("[bold blue]Bible Explorer")

    # Main Query
    console.print(f"[bold yellow]Main Query:[/bold yellow] {response['query']}\n")

    # Sub-questions
    console.print("[bold green]Sub-Questions:[/bold green]")
    for i, sq in enumerate(response['sub_questions'], 1):
        console.print(f"{i}. {sq}")
    console.print()

    # Contextual Documents
    for item in response['retrieval_results']:
        console.rule(f"[bold cyan]🔍 {item['question']}")
        for doc in item['docs']:
            meta = doc.metadata
            passage_ref = f"{meta['book']} {meta['chapter']}:{meta['verse_range']}"
            console.print(f"[bold blue]{passage_ref}[/bold blue]")
            console.print(f"{doc.page_content}\n", style="dim")

    # Final Answer
    console.rule("[bold magenta]🧠 Final Answer")
    console.print(Markdown(response['final_answer']))
