"""
embed/embeddings.py

This module provides utility functions to initialize and return embedding models
from either OpenAI (via LangChain integration) or Hugging Face.

It supports:
- Out-of-the-box usage of OpenAI embedding APIs.
- Fine-tuned models hosted on Hugging Face for domain-specific embeddings.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings

def get_openai_embedding_model():
    """
    Instantiate and return the OpenAI embedding model via LangChain.

    The model name is loaded from `settings.OPENAI_EMBEDDING_MODEL`.

    Returns:
        OpenAIEmbeddings: A LangChain-compatible embedding model for use in vector stores or pipelines.
    """
    return OpenAIEmbeddings(model=settings.OPENAI_EMBEDDING_MODEL)

def get_finetuned_embedding_model():
    """
    Instantiate and return a fine-tuned Hugging Face embedding model.

    The model is typically fine-tuned for a domain-specific retrieval task and stored locally or on the HF Hub.
    It uses the model name from `settings.HF_FINAL_FINETUNED_MODEL_NAME`.

    Returns:
        HuggingFaceEmbeddings: A LangChain-compatible embedding model for custom use cases.
    """
    return HuggingFaceEmbeddings(model_name=settings.HF_FINAL_FINETUNED_MODEL_NAME)
