# embed/qdrant_vectorstore.py

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
    Create and populate a Qdrant vector store using LangChain's QdrantVectorStore.

    Args:
        documents (List[Dict]): Chunked documents with metadata.
        embedding_model: LangChain-compatible embedding model instance.
        collection_name (str): Name of the Qdrant collection.
        vector_dim (int): Dimensionality of the embedding model.

    Returns:
        QdrantVectorStore: LangChain-compatible vector store instance.
    """
    # Convert documents to LangChain format
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
    if os.path.exists(f"{settings.QDRANT_PATH}/collection/{collection_name}"):
        print(f"Qdrant collection '{collection_name}' already exists — loading it.")
        # Create the vector store without adding documents
        vector_store = QdrantVectorStore(
            client=QdrantClient(path=settings.QDRANT_PATH),
            collection_name=collection_name,
            embedding=embedding_model,
        )
    else:
        try:
            print(f"Creating new Qdrant collection: '{collection_name}'")
            # Create the vector store and add documents
            vector_store = QdrantVectorStore.from_documents(
                path=settings.QDRANT_PATH,
                documents=langchain_docs,
                embedding=embedding_model,
                collection_name=collection_name,
                distance_func=Distance.COSINE,
                vector_dim=vector_dim,
                force_disable_check_same_thread=True,
            )
            print(f"Qdrant collection '{collection_name}' created")
        except Exception as e:
            print(f"Error creating Qdrant collection: {e}")
            raise
    return vector_store
