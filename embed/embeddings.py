# embed/embeddings.py

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings

def get_openai_embedding_model():
    return OpenAIEmbeddings(model=settings.OPENAI_EMBEDDING_MODEL)

def get_finetuned_embedding_model():
    return HuggingFaceEmbeddings(model_name=settings.HF_FINAL_FINETUNED_MODEL_NAME)
