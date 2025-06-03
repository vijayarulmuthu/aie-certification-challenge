# core/train.py

import wandb

from typing import List, Dict

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator


def prepare_examples(
    questions: Dict[str, str],
    relevant_contexts: Dict[str, str],
    corpus: Dict[str, str]
) -> List[InputExample]:
    examples = []
    print(f"Questions count: {len(questions)}")
    print(f"Relevant contexts count: {len(relevant_contexts)}")
    print(f"Corpus size: {len(corpus)}")

    missing_docs = 0
    for q_id, question in questions.items():
        doc_id = relevant_contexts.get(q_id)
        if not doc_id:
            print(f"Warning: No relevant context found for question ID {q_id}")
            continue
        doc = corpus.get(doc_id)
        if doc:
            examples.append(InputExample(texts=[question, doc]))
        else:
            missing_docs += 1

    if missing_docs > 0:
        print(f"Warning: {missing_docs} documents referenced in relevant_contexts were missing from corpus")

    print(f"Created {len(examples)} examples")
    return examples


def train_model(
    model_name: str,
    output_path: str,
    train_examples: List[InputExample],
    val_queries: Dict[str, str],
    val_corpus: Dict[str, str],
    val_relevant_docs: Dict[str, str],
    batch_size: int = 10,
    epochs: int = 10,
    matryoshka_dims: List[int] = [768, 512, 256, 128, 64],
    wandb_mode: str = "disabled",
) -> SentenceTransformer:
    """
    Fine-tune a SentenceTransformer model using Matryoshka Loss.

    Args:
        model_name (str): Hugging Face model ID.
        output_path (str): Directory to save the model.
        train_examples (List[InputExample]): Training triplets.
        val_queries, val_corpus, val_relevant_docs: For IR evaluation.
        batch_size (int)
        epochs (int)
        matryoshka_dims (List[int])
        wandb_mode (str): "online" or "disabled"

    Returns:
        SentenceTransformer: The fine-tuned model
    """
    # Check if we have training examples
    if not train_examples:
        raise ValueError(
            "No training examples available. Check that your questions, relevant_contexts, "
            "and corpus dictionaries are properly populated and aligned."
        )
    
    # Check validation data
    if not val_queries or not val_corpus or not val_relevant_docs:
        raise ValueError("Validation data is empty. Check your validation dataset generation.")
    
    wandb.init(mode=wandb_mode)

    model = SentenceTransformer(model_name)

    train_loader = DataLoader(train_examples, batch_size=batch_size, shuffle=True)

    inner_loss = losses.MultipleNegativesRankingLoss(model)
    train_loss = losses.MatryoshkaLoss(
        model,
        inner_loss,
        matryoshka_dims=matryoshka_dims,
    )

    # Create IR evaluator
    evaluator = InformationRetrievalEvaluator(
        queries=val_queries,
        corpus=val_corpus,
        relevant_docs=val_relevant_docs,
        name="validation",
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 3, 5, 10],
        precision_recall_at_k=[1, 3, 5, 10],
        map_at_k=[100],
        show_progress_bar=False,
    )

    warmup_steps = int(len(train_loader) * epochs * 0.1)

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        evaluator=evaluator,
        evaluation_steps=50,
    )

    return model
