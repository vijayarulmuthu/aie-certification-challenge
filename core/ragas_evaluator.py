# core/ragas_evaluator.py

import ast

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

def evaluate_ragas(baseline_df, rag_graph, timeout: int = 360) -> dict:
    """
    Run RAGAS evaluation over a Pandas DataFrame that includes:
      - user_input
      - response
      - retrieved_contexts (List[str])
    
    Args:
        baseline_df (pd.DataFrame): The golden testset.
        rag_graph (StateGraph): The RAG pipeline graph.
        timeout (int): Timeout per metric call.

    Returns:
        dict: Metric → score
    """

    # Fix stringified context column if needed
    if isinstance(baseline_df["reference_contexts"].iloc[0], str):
        baseline_df["reference_contexts"] = baseline_df["reference_contexts"].apply(ast.literal_eval)

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o", temperature=0)
    )

    # Run RAG pipeline and fill in answers + contexts
    generated_answers = []
    retrieved_contexts = []

    for i, row in baseline_df.iterrows():
        print(f"Processing query {i+1}/{len(baseline_df)}: {row['user_input']}")
        state = rag_graph.invoke({"query": row["user_input"]})

        generated_answers.append(state["final_answer"])
        context_texts = [
            doc.page_content
            for result in state["retrieval_results"]
            for doc in result["docs"]
        ]
        retrieved_contexts.append(context_texts)

    # Add predictions to the dataset
    baseline_df["response"] = generated_answers
    baseline_df["retrieved_contexts"] = retrieved_contexts

    # === Build EvaluationDataset ===
    eval_dataset = EvaluationDataset.from_pandas(baseline_df)

    # === RAGAS Evaluation ===
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

    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        run_config=custom_run_config
    )

    # Combine metrics with dataset
    return results.to_pandas()
