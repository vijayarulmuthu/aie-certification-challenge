"""
core/evaluate.py

This module defines a utility to evaluate a fine-tuned SentenceTransformer embedding model
using standard Information Retrieval (IR) metrics such as:

- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Accuracy@K
- Precision/Recall@K
- Mean Average Precision (MAP)

The evaluation uses the `InformationRetrievalEvaluator` provided by SentenceTransformers.
"""

from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer
from typing import Dict, List


def evaluate_ir_model(
    model: SentenceTransformer,
    queries: Dict[str, str],
    corpus: Dict[str, str],
    relevant_docs: Dict[str, str],
    name: str = "IR Evaluation",
    k_values: List[int] = [1, 3, 5, 10]
) -> None:
    """
    Evaluates the quality of a SentenceTransformer model in a retrieval setting.
    
    Each query is matched against a corpus of documents, and the model’s ability
    to rank the correct relevant document highly is measured using standard IR metrics.

    Args:
        model (SentenceTransformer): A trained SentenceTransformer embedding model.
        queries (Dict[str, str]): Mapping from query ID to query text.
        corpus (Dict[str, str]): Mapping from document ID to document content.
        relevant_docs (Dict[str, str]): Mapping from query ID to the ID of its relevant document.
        name (str): A label to identify this evaluation run (e.g., for experiment tracking).
        k_values (List[int]): The cutoffs at which to compute metrics like Recall@K, MRR@K, etc.

    Prints:
        A full IR evaluation report with metrics at each specified `k` value.
    """
    # Initialize SentenceTransformers built-in IR evaluator
    evaluator = InformationRetrievalEvaluator(
        queries=queries,                # Query set
        corpus=corpus,                  # Corpus to search from
        relevant_docs=relevant_docs,    # Ground truth (query_id → correct doc_id)
        show_progress_bar=True,         # Show a progress bar for the evaluation
        name=name,                      # Label for tracking/logging purposes

        # Metrics to compute
        mrr_at_k=k_values,
        ndcg_at_k=k_values,
        accuracy_at_k=k_values,
        precision_recall_at_k=k_values,
        map_at_k=[max(k_values)],       # MAP is computed at the largest `k`
    )

    print(f"\n📊 Running IR Evaluation: {name}")
    
    # Run the evaluation and print results to stdout (no output file written)
    evaluator(model, output_path=None)
