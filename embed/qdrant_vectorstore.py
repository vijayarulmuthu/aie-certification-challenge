"""
embed/qdrant_vectorstore.py

This module provides a utility function to create or load a Qdrant vector store
using LangChain's QdrantVectorStore integration. It handles both the conversion
of raw document chunks into LangChain Document format and the embedding ingestion.

Supports:
- Fresh collection creation with automatic embedding
- Reuse of existing Qdrant collections (persistent local DB)
"""

import os
from typing import List, Dict, Any

from langchain_qdrant import QdrantVectorStore
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance

from config import settings

def create_qdrant_vectorstore(
    documents: List[Dict[str, Any]],
    embedding_model,
    collection_name: str,
    vector_dim: int = 1536,
) -> VectorStore:
    """
    Create and populate (or load) a Qdrant vector store using LangChain's integration.

    Args:
        documents (List[Dict]): Chunked input documents where each dict contains
            - 'chunk_text': the text to embed
            - 'book', 'chapter', 'verse_range': metadata for source reference
        embedding_model: A LangChain-compatible embedding model (OpenAI or HF).
        collection_name (str): Unique name for the Qdrant collection.
        vector_dim (int): Dimensionality of the embedding vectors.

    Returns:
        VectorStore: A LangChain-compatible Qdrant vector store object.
    """
    
    # Step 1: Convert each chunk to LangChain's Document format with metadata
    langchain_docs = [
        Document(
            page_content=doc["chunk_text"],
            metadata={
                "source": f"{doc['book']} {doc['chapter']}:{doc['verse_range']}",
                "book": doc["book"],
                "chapter": doc["chapter"],
                "verse_range": doc["verse_range"],
            },
        )
        for doc in documents
    ]

    print(f"Length of langchain_docs: {len(langchain_docs)}")

    # Step 2: Check if the collection already exists on disk (persisted Qdrant DB)
    if os.path.exists(f"{settings.QDRANT_PATH}/collection/{collection_name}"):
        print(f"Qdrant collection '{collection_name}' already exists — loading it.")
        
        # Reconnect to existing collection without re-ingesting
        vector_store = QdrantVectorStore(
            client=QdrantClient(path=settings.QDRANT_PATH),
            collection_name=collection_name,
            embedding=embedding_model,
        )
    else:
        try:
            print(f"Creating new Qdrant collection: '{collection_name}'")

            # Step 3: Create and populate new collection with documents and embeddings
            vector_store = QdrantVectorStore.from_documents(
                path=settings.QDRANT_PATH,
                documents=langchain_docs,
                embedding=embedding_model,
                collection_name=collection_name,
                distance_func=Distance.COSINE,
                vector_dim=vector_dim,
                force_disable_check_same_thread=True,  # Important for multithreading use
            )
            print(f"Qdrant collection '{collection_name}' created successfully.")
        except Exception as e:
            print(f"Error creating Qdrant collection: {e}")
            raise

    return vector_store
