# app/embeddings.py
from pathlib import Path
import os
import numpy as np
import pickle
import faiss
import torch

# -------------------------------------------------------------------
# Output folder: MUST match app.py's OUTPUT_FOLDER exactly, regardless
# of the process's current working directory. Both app.py and this file
# live in <project_root>/app/, so parent.parent resolves to the same
# <project_root> in both places.
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model configuration.
# 1. If a local copy of the model exists at LOCAL_MODEL_PATH, use it (fully offline).
# 2. Otherwise, fall back to downloading the model by name from Hugging Face Hub
#    the first time it's needed (requires internet on first run only; cached after).
LOCAL_MODEL_PATH = os.environ.get(
    "EMBEDDING_MODEL_LOCAL_PATH",
    os.path.expanduser("~/pdf_summariser_models/all-MiniLM-L6-v2"),
)
HUB_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = 384

# Lazy model singleton
_MODEL = None


def get_model():
    """
    Lazy-load SentenceTransformer, preferring a local offline copy but
    falling back to a Hugging Face Hub download so the app works out of
    the box on a fresh machine without any manual model setup.
    """
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        if os.path.isdir(LOCAL_MODEL_PATH):
            print(f"[embeddings] Loading local embedding model from {LOCAL_MODEL_PATH}")
            _MODEL = SentenceTransformer(LOCAL_MODEL_PATH)
        else:
            print(
                f"[embeddings] No local model found at {LOCAL_MODEL_PATH}. "
                f"Downloading '{HUB_MODEL_NAME}' from Hugging Face Hub (cached after first run)."
            )
            _MODEL = SentenceTransformer(HUB_MODEL_NAME)

        # attempt to use MPS on mac M1/M2, or CUDA if available
        try:
            if torch.cuda.is_available():
                _MODEL.to(torch.device("cuda"))
            elif torch.backends.mps.is_available():
                _MODEL.to(torch.device("mps"))
        except Exception:
            pass
    return _MODEL


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (in-place copy safe)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-10)
    return x / norms


def embed_chunks(chunks: list, batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """
    Embed list of text chunks into a (N, D) float32 numpy array.
    Returns normalized vectors (float32) suitable for IndexFlatIP (cosine).
    """
    if not chunks:
        return np.zeros((0, EMBED_DIM), dtype="float32")

    model = get_model()
    embs_list = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        b_emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=show_progress)
        embs_list.append(b_emb)
    embs = np.vstack(embs_list).astype("float32")
    embs = _normalize_rows(embs)
    return embs


def build_faiss_index(emb_matrix: np.ndarray, use_ivf: bool = False, nlist: int = 128):
    """
    Build FAISS index. Default: IndexFlatIP on normalized vectors (fast cosine via inner product).
    If use_ivf=True, build an IVF index (faster for large corpora) -- remember to call index.train().
    Returns the FAISS index instance.
    """
    if emb_matrix.shape[0] == 0:
        raise ValueError("Cannot build a FAISS index from zero embeddings.")

    d = int(emb_matrix.shape[1])
    # IVF indexes need enough training vectors relative to nlist, otherwise
    # faiss throws an opaque error. Fall back to a flat index for small corpora.
    if use_ivf and emb_matrix.shape[0] >= nlist:
        quant = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_L2)
        index.train(emb_matrix)
        index.add(emb_matrix)
    else:
        index = faiss.IndexFlatIP(d)  # inner product on normalized vectors ~ cosine
        index.add(emb_matrix)
    return index


def persist_index(index, chunks: list, output_prefix: str, embeddings: np.ndarray = None):
    """
    Save faiss index, chunks list and (optionally) embeddings to <project_root>/outputs/<prefix>_*
    Returns paths (faiss_path, chunks_path, embeddings_path_or_None)
    """
    faiss_path = OUTPUT_DIR / f"{output_prefix}_faiss.idx"
    chunks_path = OUTPUT_DIR / f"{output_prefix}_chunks.pkl"
    emb_path = OUTPUT_DIR / f"{output_prefix}_embeddings.npy"

    faiss.write_index(index, str(faiss_path))

    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    if embeddings is not None:
        np.save(str(emb_path), embeddings)
        return str(faiss_path), str(chunks_path), str(emb_path)

    return str(faiss_path), str(chunks_path), None


def load_index_and_chunks(prefix: str):
    """
    Load FAISS index and chunks given a prefix.
    """
    faiss_path = OUTPUT_DIR / f"{prefix}_faiss.idx"
    chunks_path = OUTPUT_DIR / f"{prefix}_chunks.pkl"

    if not faiss_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"Index or chunks file not found for prefix '{prefix}'. "
            "Make sure the document has been indexed via /index_file first."
        )

    index = faiss.read_index(str(faiss_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def load_embeddings_if_exists(prefix: str):
    """
    Return embeddings np.ndarray if outputs/<prefix>_embeddings.npy exists, else None.
    """
    emb_path = OUTPUT_DIR / f"{prefix}_embeddings.npy"
    if emb_path.exists():
        return np.load(str(emb_path), mmap_mode=None)
    return None
