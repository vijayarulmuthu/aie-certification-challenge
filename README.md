# 📖 Bible Explorer — RAG-Powered Question Answering App

Bible Explorer is a Retrieval-Augmented Generation (RAG) application that allows users to ask deep questions about the Bible and receive contextualized, scholarly responses. It uses fine-tuned embeddings, LangGraph orchestration, and Gradio UI.

---

## 🚀 Features

* Semantic search across Bible passages
* Sub-query decomposition for multi-part questions
* Thematic summarization via LLM
* Fine-tuned embeddings for better relevance
* RAGAS evaluation framework support
* Interactive Gradio chatbot with citations

---

## 🐳 Running with Docker

### 📦 Step 1: Build the Docker image

```bash
docker build -t bible-explorer .
```

### 🚀 Step 2: Run the container

```bash
docker run -it --rm -p 7860:7860 --env-file .env bible-explorer
```

> Visit the app at: [http://localhost:7860](http://localhost:7860)

### 🛠 Optional: Bind your cache or data

Mount a volume if you want persistent caching:

```bash
docker run -it --rm \
  -v $PWD/cache:/home/user/app/cache \
  -v $PWD/data:/home/user/app/data \
  -p 7860:7860 \
  --env-file .env \
  bible-explorer
```

---

## 🧪 Running Locally Without Docker

You can also run the app directly on your machine.

### ✅ Prerequisites

Install:

* Python **>=3.10**
* [uv](https://github.com/astral-sh/uv) *(optional but fast)*

---

### 📥 Step 1: Clone the Repo

```bash
git clone https://github.com/vijayarulmuthu/bible-explorer.git
cd bible-explorer
```

---

### 📦 Step 2: Set Up Environment

Create a virtual environment:

```bash
uv venv  # or python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
uv sync
```

---

### ⚙️ Step 3: Configure `.env`

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_key
DATASET_PREFIX=kjv
COLLECTION_NAME_BASELINE=kjv_baseline
COLLECTION_NAME_FINETUNED=kjv_finetuned
VECTOR_DIM_BASELINE=1536
VECTOR_DIM_FINETUNED=1536
```

---

### 💬 Step 4: Launch Gradio App

```bash
python gradio_app.py
```

Visit [http://127.0.0.1:7860](http://127.0.0.1:7860)

---

## 📂 Directory Structure

```
bible-explorer/
│
├── data/                     # Raw & preprocessed Bible datasets
├── cache/                    # Cached embeddings, metrics
├── embed/                    # Embedding + chunking logic
├── pipeline/                 # LangGraph chain and UI
├── core/                     # RAGAS evaluator
├── gradio_app.py             # Main Gradio interface
├── Dockerfile                # Docker definition
└── .env                      # Environment config
```
