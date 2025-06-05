"""
embed/chunking.py

This module provides functionality to segment (chunk) a Bible verse dataset into semantically meaningful passages.
It groups verses by book and chapter, then chunks adjacent verses while respecting a configurable verse count 
and token-length limit, ensuring compatibility with transformer models.
"""

import pandas as pd
from transformers import AutoTokenizer
from typing import List, Dict
from config import settings

def chunk_verses(
    df: pd.DataFrame,
    verses_per_chunk: int = settings.VERSES_PER_CHUNK,
    max_tokens: int = settings.MAX_TOKENS,
    text_col: str = "Text",
    tokenizer_name: str = settings.TOKENIZER_NAME
) -> List[Dict]:
    """
    Chunk adjacent Bible verses by book and chapter into passages optimized for transformer-based embeddings.

    This function ensures that each chunk:
    - Includes at most `verses_per_chunk` consecutive verses.
    - Does not exceed `max_tokens` in length when tokenized by the specified tokenizer.
    - Preserves chapter boundaries (no cross-chapter chunks).

    Parameters:
        df (pd.DataFrame): DataFrame with at least ['Book', 'Chapter', 'Verse', text_col] columns.
        verses_per_chunk (int): Maximum number of verses per chunk (default: from settings).
        max_tokens (int): Maximum token length per chunk (default: from settings).
        text_col (str): Name of the column containing verse text (default: "Text").
        tokenizer_name (str): Name of Hugging Face tokenizer for token length estimation.

    Returns:
        List[Dict]: List of chunked passages with:
            - 'chunk_text' (str): Combined passage text
            - 'book' (str): Book name (e.g., "Genesis")
            - 'chapter' (int): Chapter number
            - 'verse_range' (str): Start-to-end verse range (e.g., "1-5")
    """

    # Initialize tokenizer to compute token lengths
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    grouped = []

    # Group all verses by book and chapter
    for _, chapter_df in df.groupby(["Book", "Chapter"]):
        verses = chapter_df.to_dict("records")
        i = 0

        # Iterate over verses in the chapter
        while i < len(verses):
            # Take up to N verses from the current position
            chunk = verses[i:i + verses_per_chunk]
            text = " ".join([v[text_col] for v in chunk])
            tokens = tokenizer.encode(text)

            # If token length exceeds max_tokens, shrink chunk size
            while len(tokens) > max_tokens and len(chunk) > 1:
                chunk = chunk[:-1]
                text = " ".join([v[text_col] for v in chunk])
                tokens = tokenizer.encode(text)

            # Save the valid chunk
            grouped.append({
                "chunk_text": text,
                "book": chunk[0]["Book"],
                "chapter": chunk[0]["Chapter"],
                "verse_range": f"{chunk[0]['Verse']}-{chunk[-1]['Verse']}"
            })

            # Move index forward by the size of the accepted chunk
            i += len(chunk)

    return grouped
