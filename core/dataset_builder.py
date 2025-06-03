# core/dataset_builder.py

import os
import uuid
import json

from typing import List, Tuple, Dict, Optional

from langchain_core.documents import Document

def assign_uuids(documents: List[Document]) -> List[Document]:
    """
    Assign a unique UUID to each document for identification during fine-tuning.
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
    Split documents into train / val / test sets.

    Returns:
        (train_docs, val_docs, test_docs)
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
    Build a map of {doc_id: page_content} for all documents.
    """
    return {
        doc.metadata["id"]: doc.page_content for doc in documents
    }


def load_cached_dataset(cache_dir: str, split_name: str) -> Optional[Tuple[Dict[str, str], Dict[str, str]]]:
    """
    Load a cached dataset from a JSONL file if it exists.
    
    Args:
        cache_dir: Directory containing the cached files
        split_name: Name of the split (train, val, test)
        
    Returns:
        Tuple of (questions, relevant_docs) if file exists, None otherwise
    """
    filepath = f"{cache_dir}/{split_name}_dataset.jsonl"
    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        print(f"Loaded cached QA dataset from {filepath}")
        return data["questions"], data["relevant_contexts"], data["corpus"]
    except Exception as e:
        print(f"Error loading cached dataset: {e}")
        return None


def export_datasets_to_jsonl(
    output_dir: str,
    split_name: str,
    questions: Dict[str, str],
    relevant_contexts: Dict[str, str],
    corpus: Dict[str, str]
) -> None:
    """
    Save each dataset split to disk as JSONL-style (but single-line JSON).
    """
    data = {
        "questions": questions,
        "relevant_contexts": relevant_contexts,
        "corpus": corpus,
    }
    filepath = f"{output_dir}/{split_name}_dataset.jsonl"
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
