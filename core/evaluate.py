# core/evaluate.py

from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer
from typing import Dict

def evaluate_ir_model(
    model: SentenceTransformer,
    queries: Dict[str, str],
    corpus: Dict[str, str],
    relevant_docs: Dict[str, str],
    name: str = "IR Evaluation",
    k_values: list = [1, 3, 5, 10]
) -> None:
    """
    Evaluate a fine-tuned embedding model using IR metrics (Recall@K).

    Args:
        model (SentenceTransformer): The model to evaluate.
        queries (dict): {question_id: question}
        corpus (dict): {doc_id: document text}
        relevant_docs (dict): {question_id: doc_id}
        name (str): Optional name of the evaluation run.
        k_values (list): Top-Ks to evaluate Recall@K.
    """
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        show_progress_bar=True,
        name=name,
        mrr_at_k=k_values,
        ndcg_at_k=k_values,
        accuracy_at_k=k_values,
        precision_recall_at_k=k_values,
        map_at_k=[max(k_values)],
    )

    print(f"\nRunning IR Evaluation: {name}")
    evaluator(model, output_path=None)
