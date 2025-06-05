# 📖 Bible Explorer — RAG-Powered Question Answering App

Bible Explorer is a Retrieval-Augmented Generation (RAG) application that allows users to ask deep questions about the Bible and receive contextualized, scholarly responses. It uses LangGraph for orchestration, Cohere Rerank for contextual compression, OpenAI models for query decomposition and summarization, fine-tuned embeddings, and Gradio UI.

---

## 🚀 Features

* Semantic search across Bible passages
* Sub-query decomposition for multi-part questions
* Thematic summarization via LLM
* Fine-tuned embeddings for better relevance
* RAGAS evaluation framework support
* Interactive Gradio chatbot with citations

---

## 🧱 System Components

| Layer          | Component                          | Purpose                                                       |
| -------------- | ---------------------------------- | ------------------------------------------------------------- |
| **Input**      | User Query                         | Natural language input from user                              |
| **Agent Flow** | LangGraph Agentic RAG Graph        | Multi-stage processing: Decompose → Retrieve → Summarize      |
| **LLMs**       | OpenAI (`gpt-4o`)                  | Used for decomposition and summarization                      |
| **Retrieval**  | Qdrant VectorStore + Cohere Rerank | Retrieves semantically similar Bible passages with re-ranking |
| **Data Store** | Chunked CSV + Qdrant DB            | Stores indexed Bible verse chunks and corpus metadata         |
| **Evaluation** | RAGAS + Recall\@K                  | Evaluates output faithfulness and retrieval quality           |

---

## 🔁 High-Level Flow

```text
User Query
   │
   ▼
[LangGraph AgenticRAG Graph]
   ├─> [Decompose Node] ──> Sub-Questions via OpenAI
   ├─> [Retrieve Node] ───> Qdrant + CohereRerank Contextual Retrieval
   └─> [Summarize Node] ──> Final Answer via OpenAI Chain-of-Thought

   ▼
[Final Synthesized Answer]
```

---

## 🧠 RAG Graph Nodes

### 1. `decompose`

* 🔧 Uses OpenAI (`gpt-4o`) to split multi-part questions into independent sub-questions.
* 📤 Output: `sub_questions: List[str]`

### 2. `retrieve`

* 🔍 Retrieves top-K results for each sub-question.
* 💡 Uses:

  * `vectorstore.as_retriever(k=5)`
  * `ContextualCompressionRetriever` with `Cohere Rerank v3.5`

### 3. `summarize`

* 📚 Combines all retrieved contexts.
* 🧵 Applies chain-of-thought summarization using OpenAI.

---

## 🗃️ Vector Store & Embeddings

| Component            | Model                               | Description                                    |
| -------------------- | ----------------------------------- | ---------------------------------------------- |
| **Vector Store**     | Qdrant                              | Local persistence for Bible passage embeddings |
| **Embedding Models** | OpenAI or Fine-Tuned HF Model       | Used for vector indexing and retrieval         |
| **Passage Chunking** | Token-based (max token + verse cap) | Combines nearby verses for meaningful passages |

---

## 🧪 Evaluation

| Tool         | Metric                                                  | Purpose                                     |
| ------------ | ------------------------------------------------------- | ------------------------------------------- |
| RAGAS        | Faithfulness, Factual Correctness, Context Recall, etc. | Measures output alignment with ground truth |
| IR Evaluator | Recall\@k, MRR\@k, MAP                                  | Measures embedding retrieval quality        |

---

## 🧰 Key Python Modules

| File                          | Purpose                                 |
| ----------------------------- | --------------------------------------- |
| `rag_chain.py`                | LangGraph-based orchestration           |
| `question_generator.py`       | Generates QA training samples           |
| `chunking.py`                 | Chunks verses into token-safe passages  |
| `evaluate.py`                 | Computes Recall\@K, MAP, MRR            |
| `ragas_evaluator.py`          | Runs RAGAS metrics                      |
| `finetune.py`                 | Prepares and fine-tunes embedding model |
| `qdrant_vectorstore.py`       | Manages Qdrant ingestion/loading        |
| `golden_testset_generator.py` | Generates gold-standard evaluation set  |

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
COHERE_API_KEY=your_cohere_api_key
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
