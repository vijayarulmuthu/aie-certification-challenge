"""
pipeline/finetune.py

This module handles the full pipeline for preparing and fine-tuning
a SentenceTransformer embedding model using a question-answering dataset
generated from chunked Bible passages.

It performs:
- Conversion of chunked DataFrame to LangChain Document format
- UUID assignment and document splitting into train/val/test
- Synthetic QA pair generation via LLM
- Dataset export and caching
- Model training and optional upload to Hugging Face Hub
"""

import os
import asyncio
import pandas as pd
import nest_asyncio
from pathlib import Path
from langchain_core.documents import Document

from config import settings
from core.train import prepare_examples, train_model
from core.dataset_builder import (
    assign_uuids, split_documents, build_corpus_map,
    load_cached_dataset, export_datasets_to_jsonl
)
from pipeline.question_generator import generate_qa_dataset

# ───────────────────────────────────────────────────────────────
# 🔄 Utility: Convert DataFrame to LangChain Document format
# ───────────────────────────────────────────────────────────────

def convert_chunked_docs_to_langchain(chunked_docs_df):
    """
    Converts a DataFrame of chunked Bible passages to LangChain Document objects.

    Args:
        chunked_docs_df (pd.DataFrame): Must include columns 'chunk_text', 'book', 'chapter', 'verse_range'

    Returns:
        List[Document]: LangChain-compatible documents
    """
    print("Converting chunked docs to Langchain docs...")
    langchain_docs = []
    for doc in chunked_docs_df.to_dict('records'):
        langchain_docs.append(
            Document(
                page_content=doc["chunk_text"],
                metadata={
                    "source": f"{doc['book']} {doc['chapter']}:{doc['verse_range']}",
                    "book": doc["book"],
                    "chapter": doc["chapter"],
                    "verse_range": doc["verse_range"],
                }
            )
        )
    print("Langchain docs converted successfully.")
    return langchain_docs

# ───────────────────────────────────────────────────────────────
# 📦 Dataset Preparation: Train/Val/Test + QA Pair Generation
# ───────────────────────────────────────────────────────────────

def prepare_finetune_datasets():
    """
    Prepares the dataset for fine-tuning by:
    - Loading or generating the train/val/test split
    - Generating synthetic QA pairs
    - Exporting datasets to disk (JSONL format)

    Returns:
        Tuple of train, val, and test dataset splits:
            (questions_dict, relevant_contexts_dict, corpus_dict)
    """
    if os.path.exists("datasets-finetune"):
        print("Loading cached fine-tune datasets...")
        return load_cached_dataset("datasets-finetune", "train"), \
               load_cached_dataset("datasets-finetune", "val"), \
               load_cached_dataset("datasets-finetune", "test")

    print("Loading chunked docs...")
    chunked_docs_df = pd.read_csv(f"cache/{settings.DATASET_PREFIX}/chunked_docs.csv")
    
    # Convert to LangChain docs and assign UUIDs
    langchain_docs = convert_chunked_docs_to_langchain(chunked_docs_df)
    train_docs, val_docs, test_docs = split_documents(assign_uuids(langchain_docs))

    # Corpus maps: {id: passage}
    print("Building corpus maps...")
    train_corpus = build_corpus_map(train_docs)
    val_corpus = build_corpus_map(val_docs)
    test_corpus = build_corpus_map(test_docs)

    # Required for nested asyncio environments like Jupyter
    nest_asyncio.apply()

    # LLM-generated QA pairs
    print("Generating QA datasets...")
    train_q, train_r = asyncio.run(generate_qa_dataset(train_docs, 2, desc="Train"))
    val_q, val_r = asyncio.run(generate_qa_dataset(val_docs, 2, desc="Val"))
    test_q, test_r = asyncio.run(generate_qa_dataset(test_docs, 2, desc="Test"))

    # Cache dataset to disk
    print("Exporting datasets to JSONL...")
    Path("datasets-finetune").mkdir(exist_ok=True)
    export_datasets_to_jsonl("datasets-finetune", "train", train_q, train_r, train_corpus)
    export_datasets_to_jsonl("datasets-finetune", "val", val_q, val_r, val_corpus)
    export_datasets_to_jsonl("datasets-finetune", "test", test_q, test_r, test_corpus)

    return (train_q, train_r, train_corpus), (val_q, val_r, val_corpus), (test_q, test_r, test_corpus)

# ───────────────────────────────────────────────────────────────
# 🧠 Fine-Tuning Orchestration
# ───────────────────────────────────────────────────────────────

def run_finetune(push_to_hub: bool = False):
    """
    Orchestrates the fine-tuning of the embedding model:
    - Loads or prepares datasets
    - Trains the SentenceTransformer model
    - Optionally pushes to Hugging Face Hub

    Args:
        push_to_hub (bool): Whether to upload the fine-tuned model to HF Hub

    Returns:
        SentenceTransformer: The trained model object
    """
    print("Starting fine-tuning phase...")
    print(f"Using model: {settings.HF_MODEL_NAME}")

    finetuned_model_name = settings.HF_FINETUNED_MODEL_NAME
    print(f"Finetuned model name: {finetuned_model_name}")
    print(f"Push to Hugging Face Hub: {'Yes' if push_to_hub else 'No'}")

    # Prepare data
    (train_q, train_r, train_corpus), (val_q, val_r, val_corpus), _ = prepare_finetune_datasets()
    train_data = prepare_examples(train_q, train_r, train_corpus)

    # Train the model
    model = train_model(
        model_name=settings.HF_MODEL_NAME,
        output_path=finetuned_model_name,
        train_examples=train_data,
        val_queries=val_q,
        val_corpus=val_corpus,
        val_relevant_docs=val_r,
    )

    # Optionally push to Hugging Face
    if push_to_hub:
        print("Pushing model to Hugging Face Hub...")
        model.push_to_hub(finetuned_model_name)
        print("✅ Model pushed to Hugging Face Hub successfully.")

    return model
