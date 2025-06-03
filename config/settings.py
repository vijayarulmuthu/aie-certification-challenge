# config/settings.py

import os
import openai
import uuid

from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

DATASET_PREFIX = "kjv"
DATA_PATH = "data/"
MAX_TOKENS =  512
VERSES_PER_CHUNK = 8
TOKENIZER_NAME = "bert-base-uncased"
VECTOR_DIM_BASELINE = 1536
VECTOR_DIM_FINETUNED = 768

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EVAL_MODEL = "gpt-4.1"
OPENAI_GENERATION_MODEL = "gpt-4.1-mini"

QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME_BASELINE = f"{DATASET_PREFIX}_verse_chunks"
COLLECTION_NAME_FINETUNED = f"{DATASET_PREFIX}_verse_chunks_ft"

HF_USERNAME = "vijayarulmuthu"
HF_REPO_NAME = "finetuned_arctic_kjv_bible"
HF_REPO_URL = f"https://huggingface.co/{HF_USERNAME}/{HF_REPO_NAME}"
HF_MODEL_NAME = "Snowflake/snowflake-arctic-embed-m"

HF_FINETUNED_MODEL_NAME = f"{HF_USERNAME}/{HF_REPO_NAME}-{uuid.uuid4()}"

HF_FINAL_FINETUNED_MODEL_NAME = "vijayarulmuthu/finetuned_arctic_kjv_bible-0032306b-7760-4502-aeec-80a62c6097e6"
