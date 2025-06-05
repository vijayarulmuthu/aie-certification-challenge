"""
core/dataset_builder.py

This module provides utility functions to prepare, split, and persist datasets
used for training and evaluating fine-tuned RAG pipelines. It supports:
- Assigning UUIDs to LangChain Documents
- Splitting datasets into train/val/test
- Creating corpus maps for retrieval
- Caching/loading dataset splits in JSONL-style format

Dependencies:
- langchain_core.documents.Document
"""

import os
import uuid
import json
from typing import List, Tuple, Dict, Optional

from langchain_core.documents import Document


def assign_uuids(documents: List[Document]) -> List[Document]:
    """
    Assigns a unique UUID to each document in the list.

    This ensures that every document has a persistent identifier stored in its metadata.
    Useful when constructing training corpora, tracking retrieval, or mapping context in evaluation.

    Args:
        documents (List[Document]): List of LangChain Document objects

    Returns:
        List[Document]: Same list with 'id' key added to each doc's metadata
    """
    seen = set()
    for doc in documents:
        doc_id = str(uuid.uuid4())
        while doc_id in seen:
            doc_id = str(uuid.uuid4())
        doc.metadata["id"] = doc_id
        seen.add(doc_id)
    return documents


def split_documents(
    documents: List[Document],
    train_frac: float = 0.75,
    val_frac: float = 0.125
) -> Tuple[List[Document], List[Document], List[Document]]:
    """
    Splits a list of documents into train, validation, and test subsets.

    Args:
        documents (List[Document]): Full list of documents
        train_frac (float): Fraction of data for training set
        val_frac (float): Fraction of data for validation set

    Returns:
        Tuple[List[Document], List[Document], List[Document]]:
            - Training documents
            - Validation documents
            - Test documents (remaining)
    """
    n = len(documents)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    return (
        documents[:train_end],
        documents[train_end:val_end],
        documents[val_end:]
    )


def build_corpus_map(documents: List[Document]) -> Dict[str, str]:
    """
    Builds a corpus dictionary mapping document IDs to content.

    This is used by retrievers and QA evaluators (e.g., RAGAS) to reference ground-truth passages.

    Args:
        documents (List[Document]): List of documents with metadata['id'] present

    Returns:
        Dict[str, str]: Mapping from document ID to page content
    """
    return {
        doc.metadata["id"]: doc.page_content for doc in documents
    }


def load_cached_dataset(cache_dir: str, split_name: str) -> Optional[Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]]:
    """
    Loads a previously cached dataset split from a JSONL-style file.

    Each split file contains:
        - questions: {qid: question}
        - relevant_contexts: {qid: doc_id}
        - corpus: {doc_id: page_content}

    Args:
        cache_dir (str): Directory where cached files are stored
        split_name (str): One of 'train', 'val', or 'test'

    Returns:
        Optional[Tuple[questions, relevant_contexts, corpus]]:
            Loaded data if file exists, else None
    """
    filepath = f"{cache_dir}/{split_name}_dataset.jsonl"
    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        print(f"✅ Loaded cached QA dataset from {filepath}")
        return data["questions"], data["relevant_contexts"], data["corpus"]
    except Exception as e:
        print(f"❌ Error loading cached dataset: {e}")
        return None


def export_datasets_to_jsonl(
    output_dir: str,
    split_name: str,
    questions: Dict[str, str],
    relevant_contexts: Dict[str, str],
    corpus: Dict[str, str]
) -> None:
    """
    Saves a dataset split (train/val/test) to disk in a JSONL-style format.

    Although stored as a single JSON object, the format mimics line-delimited datasets.
    This is used to preserve testsets generated during pipeline evaluation and QA training.

    Args:
        output_dir (str): Directory to save the file
        split_name (str): One of 'train', 'val', or 'test'
        questions (Dict[str, str]): {qid: question}
        relevant_contexts (Dict[str, str]): {qid: relevant_doc_id}
        corpus (Dict[str, str]): {doc_id: page_content}
    """
    data = {
        "questions": questions,
        "relevant_contexts": relevant_contexts,
        "corpus": corpus,
    }
    filepath = f"{output_dir}/{split_name}_dataset.jsonl"
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
