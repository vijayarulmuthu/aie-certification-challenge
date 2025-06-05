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
| **LLMs**       | OpenAI (`gpt-4o`, `gpt-4.1-mini`)  | Used for decomposition and summarization                      |
| **Retrieval**  | Qdrant VectorStore + Cohere Rerank | Retrieves semantically similar Bible passages with re-ranking |
| **Data Store** | Chunked CSV + Qdrant DB            | Stores indexed Bible verse chunks and corpus metadata         |
| **Evaluation** | RAGAS + Recall\@K                  | Evaluates output faithfulness and retrieval quality           |

---

## 🔁 High-Level Architecture

![Alt text](./data/architecture.png)

### 🟧 **1. Feature Extraction Pipeline**

**Purpose:**
Transform raw Bible data into a structured, machine-friendly format.

#### Steps:

* **Load Raw KJV TSV File:**

  * Reads the King James Bible (`kjv.tsv`), which contains references and text for each verse.

* **Parse Book, Chapter & Verse:**

  * Splits each verse’s “Reference” (e.g., “John 3:16”) into `Book`, `Chapter`, `Verse`.
  * Handles special book names and edge cases.

* **Export Cleaned Data:**

  * Produces a well-structured CSV (`kjv_preprocessed.csv`) ready for downstream semantic chunking and embedding.

---

### 🟦 **2. Ingest Pipeline**

**Purpose:**
Convert cleaned verse data into semantic, vectorized chunks for efficient retrieval.

#### Steps:

* **Chunk Verses:**

  * Groups adjacent verses (e.g., 8 at a time), within token and thematic constraints, to form meaningful passages for search.

* **Embedding for Chunked Verses:**

  * Uses the **OpenAI embedding model** (`text-embedding-3-small`) to generate dense vector embeddings for each passage.

* **Qdrant Ingestion:**

  * Embeds and metadata are indexed in a **Qdrant** vector database.
  * Supports fast similarity search (HNSW or other algorithms).

---

### 🟥 **3. RAG Pipeline (Retrieval-Augmented Generation)**

**Purpose:**
Answer complex, nuanced user questions about the Bible using multi-agent reasoning and retrieval.

#### Steps:

* **User’s Question:**

  * The user asks an open-ended, possibly multi-part question.

* **RAG Chain (Decompose):**

  * An **LLM agent** (using `gpt-4.1-mini`) breaks the question into simpler sub-questions for more targeted retrieval.

* **Embedding-based Retrieval:**

  * For each sub-question, similar semantic chunks are fetched from Qdrant using vector similarity.

* **LLM-based Semantic Reranker (Cohere):**

  * The set of retrieved passages is **reranked** using **Cohere’s LLM-based reranker** for higher precision and better alignment with the query intent.
  * Ensures the most relevant contexts are passed to the summarizer.

* **Summarize:**

  * Another LLM agent (again `gpt-4.1-mini`) combines the most relevant passages, using chain-of-thought prompting, to produce a coherent, markdown-formatted answer.
  * Output includes theme grouping, book-level summaries, and possibly citations.

* **Final Answer:**

  * Returned to the user in rich markdown for clarity (e.g., bold, links).

---

### 🟨 **4. Evaluation Pipeline using RAGAS**

**Purpose:**
Objectively measure the quality and reliability of the RAG pipeline outputs.

#### Steps:

* **Generate Golden Testset:**

  * Use LLMs to create a reference set of gold questions and answers, grounded in Bible content.

* **RAGAS Evaluation:**

  * Run the full RAG pipeline on these test queries.
  * Compute **RAGAS metrics**:

    * `context_recall`: Did the model retrieve the right passages?
    * `faithfulness`: Are answers grounded in context?
    * `factual_correctness`, `entity_recall`, `noise_sensitivity`: Further fine-grained measures.

* **Review Metrics:**

  * Scores are tabulated and visualized (e.g., bar charts).
  * Used to compare **baseline vs. fine-tuned** embedding models, track improvements, and guide further tuning.

---

### 🔑 **Key Architecture Advantages**

* **Semantic & Agentic:**
  Each query is decomposed and answered using a blend of retrieval and reasoning, not just keyword search.

* **Composable Agents:**
  Modular pipeline: decompose → retrieve → rerank → summarize.
  Each agent can be tuned or swapped independently.

* **LLM and Embedding Agnostic:**
  Easily swap OpenAI models, Qdrant for Pinecone/FAISS, or Cohere for other rerankers.

* **Rich Evaluation Loop:**
  Built-in, standards-based QA ensures that improvements (or regressions) are measurable and actionable.

* **Markdown & Hyperlinks:**
  Outputs are user-friendly and citeable for teaching, research, or study group settings.

* **For Bible Study Leaders / Theologians / Educators:**

  * Get multi-faceted, cross-book answers to deep questions.
  * Save time vs. manual study.
  * Trust in faithfulness and grounding via strong evaluation metrics.

* **For Developers:**

  * Highly modular, extensible stack (LangGraph, Qdrant, OpenAI, Cohere, RAGAS).
  * Easily test new retrieval, reranking, or evaluation strategies.


---

## 🧠 RAG Graph Nodes

### 1. `decompose`

* 🔧 Uses OpenAI (`gpt-4.1-mini`) to split multi-part questions into independent sub-questions.
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
