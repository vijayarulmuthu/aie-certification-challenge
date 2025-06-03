# pipeline/golden_testset_generator.py

import pandas as pd

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader

from config import settings

def generate_golden_testset(chunked_docs):
    chunk_df = pd.DataFrame([
        {"page_content": c["chunk_text"], "source": f"{c['book']} {c['chapter']}:{c['verse_range']}"}
        for c in chunked_docs
    ])

    # Convert to LangChain documents
    loader = DataFrameLoader(chunk_df, page_content_column="page_content")
    docs = loader.load()

    # Wrap LLM and embedding model
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model=settings.OPENAI_GENERATION_MODEL))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    # Generate test set
    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
    ragas_dataset = generator.generate_with_langchain_docs(docs, testset_size=10)

    return ragas_dataset.to_pandas()
