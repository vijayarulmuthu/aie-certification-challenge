import re

import pandas as pd
import gradio as gr

from pathlib import Path

from config import settings
from embed.embeddings import get_finetuned_embedding_model, get_openai_embedding_model
from embed.qdrant_vectorstore import create_qdrant_vectorstore
from pipeline.rag_chain import build_rag_chain

CACHE_DIR = Path(f"cache/{settings.DATASET_PREFIX}")
CACHE_DIR.mkdir(exist_ok=True, parents=True)

# ──────────────────────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────────────────────

def save_df(df: pd.DataFrame, name: str):
    df.to_csv(CACHE_DIR / f"{name}.csv", index=False)

def load_df(name: str):
    path = CACHE_DIR / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None

def highlight_bible_refs(text: str) -> str:
    """
    Replace book references like 'Romans 11:25-32' with markdown hyperlinks.
    """
    pattern = r'([1-3]?\s?[A-Z][a-z]+)\s+(\d+):(\d+(?:-\d+)?)'
    def linkify(match):
        book = match.group(1).replace(" ", "+")
        chapter = match.group(2)
        verses = match.group(3).replace("–", "-")
        ref = f"{match.group(1)} {chapter}:{verses}"
        url = f"https://www.biblegateway.com/passage/?search={book}+{chapter}%3A{verses}"
        return f"[{ref}]({url})"
    return re.sub(pattern, linkify, text)

# ──────────────────────────────────────────────────────────────
# STREAMING CHAIN RESPONSE
# ──────────────────────────────────────────────────────────────

def generate_stream_response(chain, question: str):
    try:
        yield f"\n\n# 🤔 Main Question\n\n{question}\n\n"

        full_answer = ""
        for step in chain.stream({"query": question}):
            sub_questions = step.get("decompose", {}).get("sub_questions", [])
            if sub_questions:
                yield f"---"
                yield f"\n\n# 🔹 Sub-Questions\n\n"
                for i, sub_q in enumerate(sub_questions):
                    yield f"{i+1}. {sub_q}\n\n"

            retrieval_results = step.get("retrieve", {}).get("retrieval_results", [])
            if retrieval_results:
                yield f"---"
                yield f"\n\n# 🔹 Retrieval Results\n\n"
                for i, result in enumerate(retrieval_results):
                    yield f"**{i+1}. {result.get('question', '')}**\n\n"
                    for doc in result.get("docs", []):
                        passage = highlight_bible_refs(doc.page_content)
                        yield f"> {highlight_bible_refs(doc.metadata.get('source', ''))}: {passage}\n\n"

            final_answer = step.get("summarize", {}).get("final_answer", "")
            if final_answer and final_answer != full_answer:
                full_answer = final_answer
                yield f"---"
                yield f"\n\n# 🧠 Final Answer\n\n"
                yield f"{highlight_bible_refs(final_answer)}\n\n"

    except Exception as e:
        yield f"❌ **Error:** {str(e)}"

# ──────────────────────────────────────────────────────────────
# GRADIO UI
# ──────────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), css="footer {display:none !important}") as demo:
    print("Loading chunked documents for ingestion...")
    chunked_docs_df = load_df("chunked_docs")
    if chunked_docs_df is None:
        raise FileNotFoundError("chunked_docs.csv is missing in cache directory")

    vectorstore = create_qdrant_vectorstore(
        documents=chunked_docs_df.to_dict("records"),
        embedding_model=get_openai_embedding_model(),
        collection_name=settings.COLLECTION_NAME_FINETUNED,
    )

    rag_chain = build_rag_chain(vectorstore, streaming=True)

    gr.Markdown("## 📖 Bible Explorer")

    chatbot = gr.Chatbot(label="Fine-Tuned RAG Chat", show_label=True, render_markdown=True, height=800)
    user_input = gr.Textbox(placeholder="Ask a question...", label="Your Question")
    send_button = gr.Button("Send")

    def chat_with_rag(user_msg):
        buffer = ""
        for chunk in generate_stream_response(rag_chain, user_msg):
            buffer += chunk
            yield [(user_msg, buffer)]

    send_button.click(
        fn=chat_with_rag,
        inputs=[user_input],
        outputs=[chatbot],
        concurrency_limit=3
    )

# === LAUNCH ===
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
