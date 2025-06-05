# AI Engineering Bootcamp - Certification Challenge

[Notebook](https://drive.google.com/file/d/1TzBp6et9waUfH6ZUMiviDw3bSxfUj8ow/view?usp=sharing)  [GitHub](https://github.com/vijayarulmuthu/aie-certification-challenge)  [Demo](https://huggingface.co/spaces/vijayarulmuthu/AIE-Certification-Challenge)  [Video](https://www.loom.com/looms/videos)

# **Defining your Problem and Audience**

Users of the King James Bible lack an intelligent, context-aware way to search, explore, and understand biblical content across books, chapters, and themes.

## **Problem Context and Target Audience**

The target user is a Bible Study Leader, Theology Student, or Christian Educator who regularly engages with Scripture for research, teaching, or sermon preparation.

These users often ask nuanced or multi-faceted questions such as:

+ "Where does the Bible talk about forgiveness across the Old and New Testaments?"
+ "Which verses describe God’s mercy using similar language?"
+ "Can you compare how love is portrayed in Psalms vs. in Paul's letters?"

Currently, they rely on keyword-based search or manual reading, which:

+ Fails to capture semantic meaning or cross-book relationships
+ Is time-consuming, especially when preparing thematic studies
+ Lacks a structured, intelligent interface for exploration or summarization

By automating semantic search, cross-referencing, and thematic summarization, we can drastically improve how these users interact with Scripture — making Bible study more efficient, insightful, and personalized.

# **Propose a Solution**

## **Proposed Solution & User Experience**

We propose building an **Agentic RAG-powered Bible Explorer**: an interactive semantic search and summarization tool for users to ask complex, open-ended questions about the Bible and receive coherent, faith-aligned, multi-verse responses.

The user interface will feel like a chat with an “AI Bible Scholar.” A Bible Study Leader can ask:

> *“Where does the Bible talk about God’s promises during times of despair?”*

The system will retrieve semantically relevant verses across books, apply agentic reasoning to group and summarize them into themes (e.g., “Hope in Psalms,” “Faith in Hebrews”), and present the output in an interpretable format — optionally citing verse locations and summaries.

Users will save **hours of manual cross-referencing**, gain **deep thematic insight**, and improve the quality of their lessons or sermons.


## **Tools and Architecture Stack**

| Layer                   | Tool                                    | Rationale                                                                              |
| ----------------------- | --------------------------------------- | -------------------------------------------------------------------------------------- |
| **LLM**                 | OpenAI (`gpt-4o`, `gpt-4.1-mini`)       | High performance in language generation, handles abstract religious themes with nuance |
| **Embedding Model**     | `text-embedding-3-small`                | Accurate, fast, OpenAI-supported, and highly cost-effective                            |
| **Orchestration**       | `LangGraph`                             | Ideal for agentic reasoning and stateful multi-step retrieval/summarization            |
| **Vector DB**           | `Qdrant`                                | Fast, production-ready, with robust metadata filtering (e.g., by book, chapter)        |
| **Evaluation**          | `RAGAS`                                 | Standard for retrieval-based evaluation (faithfulness, precision, relevance, etc.)     |
| **UI**                  | `Gradio`                                | Streamlined UI for multi-turn semantic chat, markdown-friendly and developer-friendly  |
| **Serving & Inference** | `Docker + Hugging Face Space`           | Reliable, scalable, and community-shareable deployment setup                           |


## **Agent Usage and Agentic Reasoning**

We will integrate **two LangGraph-based agents**:

1. **Query Decomposition Agent** – breaks down multi-part queries like:

   > *"What does Jesus say about prayer in the Gospels, and how is it applied in the Epistles?"*

   Into:

   * Sub-query 1: *“What does Jesus say about prayer in the Gospels?”*
   * Sub-query 2: *“How is prayer discussed in the Epistles?”*

2. **Summarization Agent** – interprets retrieved verses, clusters them by semantic meaning or book, and summarizes the findings using chain-of-thought prompting.

Agentic reasoning allows contextual memory, thematic consistency, and accurate mapping of scriptural concepts across Testaments — far beyond traditional keyword search.

# **Dealing with the Data**

## **Default Chunking Strategy**

### **Strategy**: *Semantic Chunking with Context-Aware Merging*

* **Step 1**: Start with verse-level entries as base units (each row in `kjv_preprocessed.csv`)
* **Step 2**: Group **2–5 adjacent verses** together if:

  * They share the same `Book` and `Chapter`
  * They form a logical thematic unit (measured via cosine similarity threshold ≥ 0.8 between embeddings)
* **Step 3**: Ensure the resulting chunk is ≤ 256 tokens for efficient embedding

### **Why This Strategy?**

* Verse-level granularity is too small for meaningful semantic search.
* Chapter-level is often too broad or diluted.
* Grouping adjacent verses allows richer semantic context while preserving theological integrity and referential clarity.
* This also improves **context recall** and **faithfulness** in downstream RAG outputs, which is critical in religious settings where misinterpretation must be avoided.


# 🔁 High-Level Architecture

![Alt text](./data/architecture.png)

## 🟧 **1. Feature Extraction Pipeline**

**Purpose:**
Transform raw Bible data into a structured, machine-friendly format.

### Steps:

* **Load Raw KJV TSV File:**

  * Reads the King James Bible (`kjv.tsv`), which contains references and text for each verse.

* **Parse Book, Chapter & Verse:**

  * Splits each verse’s “Reference” (e.g., “John 3:16”) into `Book`, `Chapter`, `Verse`.
  * Handles special book names and edge cases.

* **Export Cleaned Data:**

  * Produces a well-structured CSV (`kjv_preprocessed.csv`) ready for downstream semantic chunking and embedding.

## 🟦 **2. Ingest Pipeline**

**Purpose:**
Convert cleaned verse data into semantic, vectorized chunks for efficient retrieval.

### Steps:

* **Chunk Verses:**

  * Groups adjacent verses (e.g., 8 at a time), within token and thematic constraints, to form meaningful passages for search.

* **Embedding for Chunked Verses:**

  * Uses the **OpenAI embedding model** (`text-embedding-3-small`) to generate dense vector embeddings for each passage.

* **Qdrant Ingestion:**

  * Embeds and metadata are indexed in a **Qdrant** vector database.
  * Supports fast similarity search (HNSW or other algorithms).


## 🟥 **3. RAG Pipeline (Retrieval-Augmented Generation)**

**Purpose:**
Answer complex, nuanced user questions about the Bible using multi-agent reasoning and retrieval.

### Steps:

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

## 🟨 **4. Evaluation Pipeline using RAGAS**

**Purpose:**
Objectively measure the quality and reliability of the RAG pipeline outputs.

### Steps:

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

## 🔑 **Key Architecture Advantages**

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


# **Summary**

Here's a detailed summary report comparing **Baseline** and **Fine-tuned** RAG models:

## 📊 **RAGAS Evaluation Summary: Baseline vs. Fine-Tuned RAG**

![Alt text](./data/architecture.png)

| Metric                             | Baseline Score | Finetuned Score | % Change | Insights                                                                 |
|------------------------------------|----------------|-----------------|----------|--------------------------------------------------------------------------|
| context_recall                     | 0.93           | 0.975           | +4.84%   | Fine-tuned model retrieves more relevant context consistently.            |
| faithfulness                       | 0.85           | 0.80            | -5.88%   | Slight drop in staying close to source; may be due to abstraction.        |
| factual_correctness (mode=F1)     | 0.32           | 0.165           | -48.44%  | Major drop; factual hallucinations increased post-finetuning.             |
| answer_relevancy                   | 0.84           | 0.825           | -1.79%   | Minor decrease; fine-tuned answers are still topically aligned.           |
| context_entity_recall             | 0.72           | 0.90            | +25.00%  | Big gain; fine-tuned model is much better at identifying key entities.    |
| noise_sensitivity (mode=relevant) | 0.095          | 0.04            | -57.89%  | Strong improvement; less distracted by irrelevant content.                |


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

### ⚖️ Stable Metric

- **answer_relevancy (-1.79%)**
  - Slight drop, but within tolerance. The answers remain mostly aligned with the user's question.
  - Suggests that despite changes in retrieval, coherence in responses is preserved.

### 🔽 Regressed Metrics

- **faithfulness (-5.88%)**
  - Minor decrease in how well answers adhere to source material.
  - May indicate increased abstraction or paraphrasing behavior in the generation phase.

- **factual_correctness (mode=F1) (-48.44%)**
  - Major decline in factual accuracy.
  - Signals the need for enhanced control mechanisms (e.g., factuality re-rankers or answer verification) during generation.

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
