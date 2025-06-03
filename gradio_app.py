import pandas as pd
import gradio as gr
from pathlib import Path

from config import settings

from embed.embeddings import get_finetuned_embedding_model
from embed.qdrant_vectorstore import create_qdrant_vectorstore
from pipeline.rag_chain import build_rag_chain

CACHE_DIR = Path(f"cache/{settings.DATASET_PREFIX}")
CACHE_DIR.mkdir(exist_ok=True, parents=True)

def save_df(df: pd.DataFrame, name: str):
    df.to_csv(CACHE_DIR / f"{name}.csv", index=False)

def load_df(name: str):
    path = CACHE_DIR / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None

# Streaming response generator
def generate_stream_response(chain, question: str):
    try:
        yield f"🤔 Question: {question}\n\n"

        response = chain.invoke({"query": question})
        answer = response.get("response", "")

        yield f"🧠 **Answer:** {answer}\n\n"
    except Exception as e:
        yield f"❌ **Error:** {str(e)}"

# Gradio App
with gr.Blocks(theme=gr.themes.Soft(), css="footer {display:none !important}") as demo:
    print("Loading chunked documents for ingestion...")
    chunked_docs_df = load_df("chunked_docs")
    if chunked_docs_df is None:
        raise FileNotFoundError("chunked_docs.csv is missing in cache directory")

    vectorstore = create_qdrant_vectorstore(
        documents=chunked_docs_df.to_dict("records"),
        embedding_model=get_finetuned_embedding_model(),
        collection_name=settings.COLLECTION_NAME_FINETUNED,
    )

    rag_chain = build_rag_chain(vectorstore)

    gr.Markdown("## 📖 Bible Explorer — Fine-Tuned RAG")

    chatbot = gr.Chatbot(label="Fine-Tuned RAG Chat")
    user_input = gr.Textbox(placeholder="Ask a question...", label="Your Question")
    send_button = gr.Button("Send")

    def chat_with_finetuned_rag(user_msg):
        answer_stream = generate_stream_response(rag_chain, user_msg)
        for chunk in answer_stream:
            yield [(user_msg, chunk)]

    send_button.click(
        fn=chat_with_finetuned_rag,
        inputs=[user_input],
        outputs=[chatbot],
        concurrency_limit=3
    )

# === LAUNCH ===
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
