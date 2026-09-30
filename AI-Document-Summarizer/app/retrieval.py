# app/retrieval.py
from typing import List, Tuple
import numpy as np
from app.embeddings import get_model, load_index_and_chunks, load_embeddings_if_exists, embed_chunks
from pathlib import Path

def embed_query(text: str) -> np.ndarray:
    """
    Embed and normalize a single query string to float32 vector.
    """
    model = get_model()
    vec = model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
    vec = vec.astype("float32")
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm


def retrieve_top_k(prefix: str, query: str, top_k: int = 5) -> Tuple[List[str], List[int]]:
    """
    Load FAISS index and chunks, compute query embedding and return top_k chunks + indices.
    """
    index, chunks = load_index_and_chunks(prefix)
    qvec = embed_query(query).reshape(1, -1).astype("float32")
    D, I = index.search(qvec, top_k)
    indices = [int(i) for i in I[0] if i != -1]
    selected = [chunks[i] for i in indices]
    return selected, indices


def assemble_context(chunks: List[str], max_chars: int = 4000) -> str:
    """
    Concatenate chunks in order until max_chars reached.
    """
    out = []
    cur_len = 0
    for c in chunks:
        if cur_len + len(c) + 3 > max_chars:
            break
        out.append(c)
        cur_len += len(c) + 3
    return "\n\n---\n\n".join(out)