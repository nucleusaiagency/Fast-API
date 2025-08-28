import re
from typing import List, Dict

WORD_PATTERN = re.compile(r"\w+[’']?\w*|\S", re.UNICODE)

def word_tokenize(text: str) -> list:
    # Lightweight tokenizer: words / punctuation
    return WORD_PATTERN.findall(text)

def chunk_text(text: str, target_words: int = 500, overlap_words: int = 100):
    tokens = word_tokenize(text)
    n = len(tokens)
    if n == 0:
        return []

    chunks = []
    start = 0
    while start < n:
        end = min(start + target_words, n)
        chunk_tokens = tokens[start:end]
        chunk_text_str = " ".join(chunk_tokens)
        chunks.append({
            "chunk_id": len(chunks),
            "chunk_start_word": start,
            "chunk_end_word": end,
            "chunk_word_count": len(chunk_tokens),
            "overlap_words": overlap_words,
            "text": chunk_text_str
        })
        if end == n:
            break
        start = max(0, end - overlap_words)
    return chunks
