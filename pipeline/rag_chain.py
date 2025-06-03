# pipeline/rag_chain.py

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

from rich.console import Console
from rich.markdown import Markdown

console = Console()

class State(TypedDict):
    question: str
    context: List[Document]
    response: str

def build_rag_chain(vectorstore):

    class SubQueryResponse(BaseModel):
        """
        Validated schema for the output of the sub-query decomposition LLM.
        Ensures that the parsed result contains a list of sub-questions.
        """
        subQuestions: List[str]

    def parse_subquery_response(response_text: str) -> SubQueryResponse:
        """
        Parses the JSON string from the LLM sub-query splitter into a validated SubQueryResponse object.

        Args:
            response_text (str): Raw string output from LLM containing a JSON response.

        Returns:
            SubQueryResponse: Validated and structured list of sub-queries.
        """
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON found in sub-query response.")
        data = json.loads(json_match.group())
        return SubQueryResponse(**data)


    # Setup LLM + retriever
    llm = ChatOpenAI(model=settings.OPENAI_GENERATION_MODEL)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # === PROMPTS ===

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


    # === GRAPH NODES ===

    def decompose_query(state):
        query = state["query"]
        messages = [HumanMessage(content=decompose_prompt.format(query=query))]
        result = llm.invoke(messages).content
        sub_questions = parse_subquery_response(result).subQuestions
        return {**state, "sub_questions": sub_questions}

    def retrieve_context(state):
        results = []
        for q in state["sub_questions"]:
            docs = retriever.invoke(q)
            results.append({"question": q, "docs": docs})
        return {**state, "retrieval_results": results}

    def summarize_results(state):
        combined_docs = []
        for entry in state["retrieval_results"]:
            for doc in entry["docs"]:
                combined_docs.append(doc.page_content)
        joined = "\n\n".join(combined_docs)
        summary = llm.invoke([HumanMessage(content=summarize_prompt.format(docs=joined))]).content
        return {**state, "final_answer": summary}


    # === GRAPH DEFINITION ===

    class AgenticRAGState(TypedDict):
        query: str
        sub_questions: List[str]
        retrieval_results: List[dict]
        final_answer: str

    graph_builder = StateGraph(AgenticRAGState)

    graph_builder.add_node("decompose", RunnableLambda(decompose_query))
    graph_builder.add_node("retrieve", RunnableLambda(retrieve_context))
    graph_builder.add_node("summarize", RunnableLambda(summarize_results))

    graph_builder.set_entry_point("decompose")
    graph_builder.add_edge("decompose", "retrieve")
    graph_builder.add_edge("retrieve", "summarize")
    graph_builder.add_edge("summarize", END)

    return graph_builder.compile()

def print_rag_response(response):
    console.rule("[bold blue]Bible Explorer")

    # Query
    console.print(f"[bold yellow]Main Query:[/bold yellow] {response['query']}\n")

    # Sub-Questions
    console.print("[bold green]Sub-Questions:[/bold green]")
    for i, sq in enumerate(response['sub_questions'], 1):
        console.print(f"{i}. {sq}")
    console.print()

    # Documents retrieved for each sub-question
    for item in response['retrieval_results']:
        console.rule(f"[bold cyan]🔍 {item['question']}")
        for doc in item['docs']:
            meta = doc.metadata
            passage_ref = f"{meta['book']} {meta['chapter']}:{meta['verse_range']}"
            console.print(f"[bold blue]{passage_ref}[/bold blue]")
            console.print(f"{doc.page_content}\n", style="dim")

    # Final synthesized answer
    console.rule("[bold magenta]🧠 Final Answer")
    console.print(Markdown(response['final_answer']))
