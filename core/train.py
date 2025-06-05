"""
core/train.py

This module handles training a SentenceTransformer model using contrastive learning 
with Matryoshka Loss. It includes utilities to:
- Convert question-context pairs into InputExample format
- Fine-tune the model with evaluation on IR metrics using SentenceTransformers
- Log progress using wandb (optional)

Dependencies:
- SentenceTransformers
- PyTorch
- wandb
"""

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
    """
    Prepares training examples by pairing each question with its relevant context document.

    Args:
        questions (Dict[str, str]): Mapping from question ID to question text
        relevant_contexts (Dict[str, str]): Mapping from question ID to relevant doc ID
        corpus (Dict[str, str]): Mapping from doc ID to actual content

    Returns:
        List[InputExample]: Formatted examples for SentenceTransformer training
    """
    examples = []
    print(f"Questions count: {len(questions)}")
    print(f"Relevant contexts count: {len(relevant_contexts)}")
    print(f"Corpus size: {len(corpus)}")

    missing_docs = 0
    for q_id, question in questions.items():
        doc_id = relevant_contexts.get(q_id)
        if not doc_id:
            print(f"⚠️  Warning: No relevant context found for question ID {q_id}")
            continue
        doc = corpus.get(doc_id)
        if doc:
            examples.append(InputExample(texts=[question, doc]))
        else:
            missing_docs += 1

    if missing_docs > 0:
        print(f"⚠️  Warning: {missing_docs} referenced documents were missing from corpus")

    print(f"✅ Created {len(examples)} training examples")
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
    Fine-tunes a SentenceTransformer model using contrastive learning with Matryoshka Loss,
    and evaluates retrieval quality on validation data using IR metrics.

    Args:
        model_name (str): HuggingFace model name (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        output_path (str): Directory to save the fine-tuned model
        train_examples (List[InputExample]): Formatted training pairs (query, doc)
        val_queries (Dict[str, str]): Validation queries (query_id -> text)
        val_corpus (Dict[str, str]): Validation corpus (doc_id -> passage)
        val_relevant_docs (Dict[str, str]): Validation labels (query_id -> relevant doc_id)
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        matryoshka_dims (List[int]): Vector dimensionalities used in Matryoshka Loss
        wandb_mode (str): "online", "offline", or "disabled"

    Returns:
        SentenceTransformer: The trained embedding model
    """
    if not train_examples:
        raise ValueError(
            "No training examples found. Ensure prepare_examples() ran correctly."
        )

    if not val_queries or not val_corpus or not val_relevant_docs:
        raise ValueError("Validation dataset is incomplete. Cannot evaluate.")

    # Initialize Weights & Biases logging
    wandb.init(mode=wandb_mode, project="bible-embedder", name="matryoshka-finetune")

    # Load pre-trained base model
    model = SentenceTransformer(model_name)

    # Create PyTorch dataloader for training
    train_loader = DataLoader(train_examples, batch_size=batch_size, shuffle=True)

    # Apply Matryoshka Loss using nested representation learning
    base_loss = losses.MultipleNegativesRankingLoss(model)
    train_loss = losses.MatryoshkaLoss(
        model=model,
        loss=base_loss,
        matryoshka_dims=matryoshka_dims
    )

    # Set up information retrieval evaluation
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

    # Warm-up phase to stabilize training
    warmup_steps = int(len(train_loader) * epochs * 0.1)

    # Begin training loop
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        evaluator=evaluator,
        evaluation_steps=50,  # Evaluate every N steps
    )

    return model
