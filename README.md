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

* **Bible Study Leaders / Theologians / Educators:**

  * Get multi-faceted, cross-book answers to deep questions.
  * Save time vs. manual study.
  * Trust in faithfulness and grounding via strong evaluation metrics.

* **Conclusion:**

  * Highly modular, extensible stack (LangGraph, Qdrant, OpenAI, Cohere, RAGAS).
  * Easily test new retrieval, reranking, or evaluation strategies.

---

## 📊 **RAGAS Evaluation Summary: Baseline vs. Fine-Tuned RAG**

![Alt text](./data/metrics-comparision.png)

| Metric                             | Baseline Score | Finetuned Score | % Change | Insights                                                                 |
|------------------------------------|----------------|-----------------|----------|--------------------------------------------------------------------------|
| context_recall                     | 0.93           | 0.975           | +4.84%   | Fine-tuned model retrieves more relevant context consistently.            |
| faithfulness                       | 0.85           | 0.80            | -5.88%   | Slight drop in staying close to source; may be due to abstraction.        |
| factual_correctness (mode=F1)     | 0.32           | 0.165           | -48.44%  | Major drop; factual hallucinations increased post-finetuning.             |
| answer_relevancy                   | 0.84           | 0.825           | -1.79%   | Minor decrease; fine-tuned answers are still topically aligned.           |
| context_entity_recall             | 0.72           | 0.90            | +25.00%  | Big gain; fine-tuned model is much better at identifying key entities.    |
| noise_sensitivity (mode=relevant) | 0.095          | 0.04            | -57.89%  | Strong improvement; less distracted by irrelevant content.                |


---

## 📌 **Insights: RAGAS Metrics Interpretation**

### 🔼 Improved Metrics

- **context_recall (+4.84%)**
  - The fine-tuned model retrieves more relevant passages with higher consistency.
  - This suggests improved alignment between embeddings and document content.

- **context_entity_recall (+25.00%)**
  - Fine-tuning helped the model focus better on key biblical entities (e.g., people, books, places).
  - This is valuable for question answering tasks grounded in specific theological references.

- **noise_sensitivity (mode=relevant) (-57.89%)**
  - A significant reduction in retrieving irrelevant or distracting content.
  - Indicates better semantic discrimination by the fine-tuned model.

---

### ⚖️ Stable Metric

- **answer_relevancy (-1.79%)**
  - Slight drop, but within tolerance. The answers remain mostly aligned with the user's question.
  - Suggests that despite changes in retrieval, coherence in responses is preserved.

---

### 🔽 Regressed Metrics

- **faithfulness (-5.88%)**
  - Minor decrease in how well answers adhere to source material.
  - May indicate increased abstraction or paraphrasing behavior in the generation phase.

- **factual_correctness (mode=F1) (-48.44%)**
  - Major decline in factual accuracy.
  - Signals the need for enhanced control mechanisms (e.g., factuality re-rankers or answer verification) during generation.

---

## 📈 **Recommended Next Steps**

1. **Improve Factual Correctness:**

   * Use RAGAS-generated hallucination examples to fine-tune the model on factual answering.
   * Add NLI (Natural Language Inference) or factual reranking stages post-generation.

2. **Boost Faithfulness:**

   * Modify prompt instructions to enforce answer grounding.
   * Add more "chain-of-thought + evidence alignment" examples during fine-tuning.

3. **RAG Optimization:**

   * Consider reranking retrieved chunks or increasing `k` for retrieval.
   * Use sentence-level retrieval or chunk-fusion if entity recall is crucial.

4. **Instrumentation:**

   * Integrate [RAGAS feedback loop](https://github.com/explodinggradients/ragas) for continuous evaluation & regression tracking.
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

## 🧠 RAG Graph Nodes

### 1. `Decompose`

* 🔧 Uses OpenAI (`gpt-4.1-mini`) to split multi-part questions into independent sub-questions.
* 📤 Output: `sub_questions: List[str]`

### 2. `Retrieve`

* 🔍 Retrieves top-K results for each sub-question.
* 💡 Uses:

  * `vectorstore.as_retriever(k=5)`
  * `ContextualCompressionRetriever` with `Cohere Rerank v3.5`

### 3. `Summarize`

* 📚 Combines all retrieved contexts.
* 🧵 Applies chain-of-thought summarization using OpenAI.

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
