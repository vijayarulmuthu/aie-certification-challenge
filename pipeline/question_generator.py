# pipeline/question_generator.py

import uuid
import asyncio
import json
from tqdm import tqdm
from typing import Tuple, Dict, List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import settings

# Configure the zero-temperature LLM for generation
llm = ChatOpenAI(model=settings.OPENAI_GENERATION_MODEL, temperature=0)

# QA prompt template with JSON output
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
    Generate n_questions for a single document and return mappings.
    """
    if not document.page_content or len(document.page_content.strip()) < 20:
        print(f"Warning: Document content too short or empty: {document.page_content[:30]}...")
        return {}, {}

    if "id" not in document.metadata:
        print(f"Warning: Document missing ID in metadata: {document.metadata}")
        return {}, {}

    try:
        questions_output = await chain.ainvoke({
            "context": document.page_content,
            "n_questions": n_questions
        })

        content = questions_output.content.strip()
        if not content:
            print(f"Warning: Empty response from LLM for document: {document.metadata.get('id', 'unknown')}")
            return {}, {}

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
    Generate QA pairs for a list of documents asynchronously.
    
    Args:
        documents: List of documents to process
        n_questions: Number of questions to generate per document
        desc: Description for the progress bar
        concurrency_limit: Maximum number of concurrent API calls
    """
    if not documents:
        print("Warning: Empty document list provided to generate_qa_dataset")
        return {}, {}

    # Check that all documents have IDs
    missing_ids = sum(1 for doc in documents if "id" not in doc.metadata)
    if missing_ids > 0:
        print(f"Warning: {missing_ids} documents are missing IDs in their metadata")

    all_questions = {}
    all_relevant_docs = {}

    # Use a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def bounded_process_document(doc):
        async with semaphore:
            return await process_document(doc, n_questions)

    # Create tasks with bounded concurrency
    tasks = [bounded_process_document(doc) for doc in documents]

    # Process tasks as they complete
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
