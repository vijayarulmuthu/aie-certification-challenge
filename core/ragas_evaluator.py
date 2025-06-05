"""
core/ragas_evaluator.py

This module defines the RAGAS evaluation pipeline for assessing a Retrieval-Augmented Generation (RAG) system.
It uses a golden testset to run queries through a RAG chain (LangGraph) and evaluates the generated answers 
and retrieved contexts using RAGAS metrics, such as:

- LLMContextRecall
- Faithfulness
- FactualCorrectness
- ResponseRelevancy
- ContextEntityRecall
- NoiseSensitivity
"""

import ast
import pandas as pd

from config import settings

from ragas import evaluate, RunConfig, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall,
    NoiseSensitivity
)
from langchain_openai import ChatOpenAI


def evaluate_ragas(baseline_df: pd.DataFrame, rag_graph, timeout: int = 360) -> pd.DataFrame:
    """
    Runs a complete RAGAS evaluation workflow:
    1. Feeds user queries from the test set into the RAG pipeline.
    2. Collects generated answers and retrieved documents.
    3. Evaluates the results using a set of LLM-based metrics.

    Args:
        baseline_df (pd.DataFrame): Testset containing at least:
            - 'user_input': The query/question.
            - 'reference_contexts': Ground-truth context list (may be strified).
        rag_graph (StateGraph): The RAG pipeline implemented via LangGraph or similar framework.
        timeout (int): Timeout in seconds for each metric evaluation step.

    Returns:
        pd.DataFrame: DataFrame of RAGAS metric results (e.g., faithfulness, recall, relevancy).
    """

    # ── Step 1: Convert stringified reference contexts to Python list, if needed ──
    if isinstance(baseline_df["reference_contexts"].iloc[0], str):
        baseline_df["reference_contexts"] = baseline_df["reference_contexts"].apply(ast.literal_eval)

    # ── Step 2: Initialize RAGAS-compatible LLM wrapper ──
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o", temperature=0)
    )

    # ── Step 3: Run the RAG pipeline over testset to get responses and retrieved docs ──
    generated_answers = []
    retrieved_contexts = []

    for i, row in baseline_df.iterrows():
        print(f"Processing query {i+1}/{len(baseline_df)}: {row['user_input']}")
        
        # Run one query through the pipeline
        state = rag_graph.invoke({"query": row["user_input"]})

        # Extract the final answer from the chain
        generated_answers.append(state["final_answer"])

        # Collect all retrieved document page contents
        context_texts = [
            doc.page_content
            for result in state["retrieval_results"]
            for doc in result["docs"]
        ]
        retrieved_contexts.append(context_texts)

    # ── Step 4: Store RAG outputs in testset dataframe ──
    baseline_df["response"] = generated_answers
    baseline_df["retrieved_contexts"] = retrieved_contexts

    # ── Step 5: Convert the testset into RAGAS-compatible EvaluationDataset ──
    eval_dataset = EvaluationDataset.from_pandas(baseline_df)

    # ── Step 6: Define evaluation metrics and configuration ──
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=settings.OPENAI_EVAL_MODEL))
    custom_run_config = RunConfig(timeout=timeout)

    metrics = [
        LLMContextRecall(),
        Faithfulness(),
        FactualCorrectness(),
        ResponseRelevancy(),
        ContextEntityRecall(),
        NoiseSensitivity()
    ]

    # ── Step 7: Run RAGAS evaluation using selected metrics ──
    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        run_config=custom_run_config
    )

    # ── Step 8: Convert to pandas DataFrame for visualization or persistence ──
    return results.to_pandas()
