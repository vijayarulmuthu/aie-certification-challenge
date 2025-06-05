"""
pipeline/golden_testset_generator.py

This module is responsible for generating a "golden" test set using RAGAS's TestsetGenerator.
The golden set consists of high-quality QA pairs derived from input Bible passages,
intended for evaluating RAG systems with reliable ground truth.

It leverages:
- LangChain document loaders
- OpenAI LLM and embedding wrappers
- RAGAS generation APIs
"""

import pandas as pd

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader

from config import settings

def generate_golden_testset(chunked_docs):
    """
    Generates a high-quality test set of question-answer pairs from Bible text chunks
    using RAGAS's synthetic testset generator.

    Args:
        chunked_docs (List[Dict]): List of dicts with keys: 'chunk_text', 'book', 'chapter', 'verse_range'

    Returns:
        pd.DataFrame: A golden test set with columns such as:
                      ['user_input', 'ground_truth', 'reference_contexts']
    """
    
    # Step 1: Create DataFrame with content and metadata
    chunk_df = pd.DataFrame([
        {
            "page_content": c["chunk_text"],
            "source": f"{c['book']} {c['chapter']}:{c['verse_range']}"
        }
        for c in chunked_docs
    ])

    # Step 2: Convert to LangChain documents for compatibility with RAGAS
    loader = DataFrameLoader(chunk_df, page_content_column="page_content")
    docs = loader.load()

    # Step 3: Wrap OpenAI LLM and Embedding models for RAGAS compatibility
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model=settings.OPENAI_GENERATION_MODEL))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    # Step 4: Generate golden QA dataset using RAGAS
    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
    ragas_dataset = generator.generate_with_langchain_docs(docs, testset_size=10)

    # Step 5: Return as Pandas DataFrame
    return ragas_dataset.to_pandas()
