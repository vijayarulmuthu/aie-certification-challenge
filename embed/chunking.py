# embed/chunking.py

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
    Chunk adjacent Bible verses by chapter into semantically meaningful passages.

    Parameters:
        df (pd.DataFrame): DataFrame containing columns ['Book', 'Chapter', 'Verse', text_col]
        verses_per_chunk (int): Maximum number of verses per chunk.
        max_tokens (int): Maximum token length per chunk.
        text_col (str): Name of the column containing verse text.
        tokenizer_name (str): Tokenizer to use for length estimation.

    Returns:
        List[Dict]: List of chunks with keys: 'chunk_text', 'book', 'chapter', 'verse_range'
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    grouped = []

    for _, chapter_df in df.groupby(["Book", "Chapter"]):
        verses = chapter_df.to_dict("records")
        i = 0
        while i < len(verses):
            chunk = verses[i:i+verses_per_chunk]
            text = " ".join([v[text_col] for v in chunk])
            tokens = tokenizer.encode(text)

            # Reduce chunk size until it fits within token limit
            while len(tokens) > max_tokens and len(chunk) > 1:
                chunk = chunk[:-1]
                text = " ".join([v[text_col] for v in chunk])
                tokens = tokenizer.encode(text)

            grouped.append({
                "chunk_text": text,
                "book": chunk[0]["Book"],
                "chapter": chunk[0]["Chapter"],
                "verse_range": f"{chunk[0]['Verse']}-{chunk[-1]['Verse']}"
            })
            i += len(chunk)

    return grouped
