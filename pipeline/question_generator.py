"""
pipeline/question_generator.py

This module uses an OpenAI language model to generate question-answer pairs (QA) from a
collection of documents. The questions are directly answerable from the content of each
document. It is designed to support dataset creation for training and evaluating retrieval-augmented
generation (RAG) systems.

Key features:
- Uses structured prompts to generate multiple questions per document
- Strictly enforces JSON output formatting
- Asynchronous batch processing with concurrency limits
"""

import uuid
import asyncio
import json
from tqdm import tqdm
from typing import Tuple, Dict, List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import settings

# Initialize the LLM for question generation with deterministic behavior
llm = ChatOpenAI(model=settings.OPENAI_GENERATION_MODEL, temperature=0)

# Define the prompt template with strict JSON output enforcement
messages = [
    (
        "system", "You are a helpful assistant that generates questions from a given context.",
    ),
    (
        "human", """
Given the following context, generate {n_questions} questions that are directly answerable from it.

Context:
{context}

IMPORTANT: You must respond ONLY with valid JSON in the exact format shown below:

{{
  "questions": [
    "Question 1?",
    "Question 2?"
  ]
}}

No other text, explanations, or formatting should be included in your response.
"""
    ),
]

prompt = ChatPromptTemplate.from_messages(messages)
chain = prompt | llm

async def process_document(
    document: Document,
    n_questions: int
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Process a single document to generate `n_questions` and return question-ID and mapping.

    Args:
        document (Document): LangChain Document object with `.page_content` and `.metadata["id"]`
        n_questions (int): Number of questions to generate for this document

    Returns:
        Tuple[Dict[str, str], Dict[str, str]]:
            - {question_id: question_text}
            - {question_id: document_id}
    """
    if not document.page_content or len(document.page_content.strip()) < 20:
        print(f"Warning: Document content too short or empty: {document.page_content[:30]}...")
        return {}, {}

    if "id" not in document.metadata:
        print(f"Warning: Document missing ID in metadata: {document.metadata}")
        return {}, {}

    try:
        # Generate questions using the prompt chain
        questions_output = await chain.ainvoke({
            "context": document.page_content,
            "n_questions": n_questions
        })

        content = questions_output.content.strip()
        if not content:
            print(f"Warning: Empty response from LLM for document: {document.metadata.get('id', 'unknown')}")
            return {}, {}

        # Parse the strict JSON output
        try:
            parsed_json = json.loads(content)
            question_list = parsed_json.get("questions", [])
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed for document {document.metadata.get('id', 'unknown')}: {e}")
            print(f"Raw output: {content}")
            return {}, {}

        if not isinstance(question_list, list) or not question_list:
            print(f"Warning: 'questions' key missing or not a list for document {document.metadata.get('id', 'unknown')}")
            return {}, {}

        doc_questions = {}
        doc_relevant_docs = {}
        doc_id = document.metadata.get("id", "")

        # Assign unique question IDs and associate with document
        for question in question_list:
            if question and isinstance(question, str):
                q_id = str(uuid.uuid4())
                doc_questions[q_id] = question.strip()
                doc_relevant_docs[q_id] = doc_id

        return doc_questions, doc_relevant_docs

    except Exception as e:
        print(f"Error processing document {document.metadata.get('id', 'unknown')}: {e}")
        return {}, {}

async def generate_qa_dataset(
    documents: List[Document],
    n_questions: int = 2,
    desc: str = "Generating questions",
    concurrency_limit: int = 5
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Asynchronously generate question-answer pairs for a list of documents.

    Args:
        documents (List[Document]): LangChain documents with page_content and metadata['id']
        n_questions (int): Number of questions to generate per document
        desc (str): Description for tqdm progress bar
        concurrency_limit (int): Maximum number of concurrent API calls

    Returns:
        Tuple[Dict[str, str], Dict[str, str]]:
            - All question ID → question mappings
            - All question ID → document ID mappings
    """
    if not documents:
        print("Warning: Empty document list provided to generate_qa_dataset")
        return {}, {}

    # Warn if any document is missing an ID
    missing_ids = sum(1 for doc in documents if "id" not in doc.metadata)
    if missing_ids > 0:
        print(f"Warning: {missing_ids} documents are missing IDs in their metadata")

    all_questions = {}
    all_relevant_docs = {}

    # Use asyncio.Semaphore to throttle concurrent API calls
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def bounded_process_document(doc):
        async with semaphore:
            return await process_document(doc, n_questions)

    tasks = [bounded_process_document(doc) for doc in documents]

    failures = 0
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
        try:
            q_map, r_map = await task
            all_questions.update(q_map)
            all_relevant_docs.update(r_map)
        except Exception as e:
            failures += 1
            print(f"Error processing document task: {e}")

    if failures > 0:
        print(f"Warning: {failures} document processing tasks failed")

    print(f"Generated {len(all_questions)} questions from {len(documents)} documents")

    if not all_questions:
        print("ERROR: No questions were generated. Check the logs above for errors.")

    return all_questions, all_relevant_docs
