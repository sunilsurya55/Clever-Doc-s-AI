# app/chunking.py

from typing import List
import re

try:
    import tiktoken
    _TIKTOKEN_IMPORT_ERROR = None
except Exception as _e:
    tiktoken = None
    _TIKTOKEN_IMPORT_ERROR = _e


class _SimpleWordEncoder:
    """
    Fallback 'tokenizer' used only if tiktoken's BPE ranks file can't be
    obtained (no internet on first run AND no local tiktoken cache). This
    approximates tokens as whitespace-separated words, which is good enough
    for chunk-sizing purposes and keeps the app fully usable offline.
    """
    def encode(self, text: str):
        return text.split()

    def decode(self, tokens) -> str:
        return " ".join(tokens)


def _load_encoder():
    if tiktoken is None:
        print(
            f"[chunking] tiktoken package is not installed ({_TIKTOKEN_IMPORT_ERROR}). "
            "Falling back to a simple whitespace tokenizer so chunking still works. "
            "Run `pip install tiktoken` for more accurate token-based chunk sizing."
        )
        return _SimpleWordEncoder()
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(
            f"[chunking] Could not load tiktoken's 'cl100k_base' encoding ({e}). "
            "This normally requires a one-time internet download. Falling back to "
            "a simple whitespace tokenizer so chunking still works fully offline."
        )
        return _SimpleWordEncoder()


ENC = _load_encoder()


def clean_text(text: str) -> str:
    """
    Remove noisy sections like references, bibliography, and excessive blank lines.
    """
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(Bibliography|References)[\s\S]*$', '', text, flags=re.I)
    return text.strip()


def count_tokens(text: str) -> int:
    """
    Count tokens with the active encoder (tiktoken if available, else the
    whitespace fallback).
    """
    return len(ENC.encode(text))


def chunk_text(
    text: str,
    max_tokens: int = 900,
    overlap_tokens: int = 150
) -> List[str]:
    """
    Splits text into overlapping chunks based on token length.
    Good defaults:
      - 700-1200 tokens per chunk
      - 100-200 token overlap
    """
    if overlap_tokens >= max_tokens:
        # Guard against a configuration that would make the sliding window
        # never advance (start = end - overlap would stay <= previous start),
        # which previously could loop indefinitely.
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    if not text or not text.strip():
        return []

    tokens = ENC.encode(text)
    if not tokens:
        return []

    chunks = []
    start = 0
    end = max_tokens

    while start < len(tokens):
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            break
        piece = ENC.decode(chunk_tokens)
        chunks.append(piece)

        if end >= len(tokens):
            break

        # next window with overlap
        start = end - overlap_tokens
        if start < 0:
            start = 0
        end = start + max_tokens

    return chunks


def prepare_chunks(raw_text: str) -> List[str]:
    """
    Full pipeline: clean -> chunk
    """
    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned)
    return chunks
