# config/settings.py

import os
import openai
import uuid

from dotenv import load_dotenv

# Load API keys and other environment variables from a .env file
load_dotenv()

# ──────────────────────────────────────────────────────────────
# 🔐 API Keys and Client Setup
# ──────────────────────────────────────────────────────────────

# OpenAI API key for using LLMs and embedding models
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# ──────────────────────────────────────────────────────────────
# 📚 Dataset Configuration
# ──────────────────────────────────────────────────────────────

# Identifier prefix for the dataset being used (e.g., KJV Bible)
DATASET_PREFIX = "kjv"

# Base path where raw and intermediate data is stored
DATA_PATH = "data/"

# Max token length allowed for a single chunk (used for chunking and LLM limits)
MAX_TOKENS = 512

# Number of verses to group into one semantic chunk
VERSES_PER_CHUNK = 8

# Hugging Face tokenizer used to estimate token length during chunking
TOKENIZER_NAME = "bert-base-uncased"

# ──────────────────────────────────────────────────────────────
# 🔢 Embedding Configuration
# ──────────────────────────────────────────────────────────────

# Dimensionality for baseline (OpenAI) and fine-tuned (HF) embeddings
VECTOR_DIM_BASELINE = 1536
VECTOR_DIM_FINETUNED = 1536

# ──────────────────────────────────────────────────────────────
# 🤖 OpenAI Models
# ──────────────────────────────────────────────────────────────

# Embedding model used for baseline vector representation
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# LLM used for RAGAS evaluation (metrics like Faithfulness, Context Recall)
OPENAI_EVAL_MODEL = "gpt-4.1"

# LLM used for generation tasks like sub-question decomposition and summarization
OPENAI_GENERATION_MODEL = "gpt-4.1-mini"

# ──────────────────────────────────────────────────────────────
# 💾 Qdrant Vector Store
# ──────────────────────────────────────────────────────────────

# Local path where Qdrant stores its data
QDRANT_PATH = "./qdrant_data"

# Collection name for storing baseline embeddings
COLLECTION_NAME_BASELINE = f"{DATASET_PREFIX}_verse_chunks"

# Collection name for storing fine-tuned embeddings
COLLECTION_NAME_FINETUNED = f"{DATASET_PREFIX}_verse_chunks_ft"

# ──────────────────────────────────────────────────────────────
# 🤗 Hugging Face Configuration
# ──────────────────────────────────────────────────────────────

# Your Hugging Face username
HF_USERNAME = "vijayarulmuthu"

# Name of the repository for storing the fine-tuned model
HF_REPO_NAME = "finetuned_arctic_kjv_bible"

# Full URL of the Hugging Face repository
HF_REPO_URL = f"https://huggingface.co/{HF_USERNAME}/{HF_REPO_NAME}"

# Base model used for fine-tuning (Snowflake Arctic Embed)
HF_MODEL_NAME = "Snowflake/snowflake-arctic-embed-m"

# Temporary name for the fine-tuned model (includes a random UUID suffix for uniqueness)
HF_FINETUNED_MODEL_NAME = f"{HF_USERNAME}/{HF_REPO_NAME}-{uuid.uuid4()}"

# Final fixed name for the fine-tuned model once pushed
HF_FINAL_FINETUNED_MODEL_NAME = "vijayarulmuthu/finetuned_arctic_kjv_bible-f2989784-6473-4f78-a30e-f532a6360101"
