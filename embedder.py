import hashlib
from functools import lru_cache
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config import CFG
from logger import get_logger, Timer

log = get_logger("embedder")

_model: Optional[SentenceTransformer] = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info(f"Loading embedding model: {CFG.embedding.content_model}")
        _model = SentenceTransformer(
            CFG.embedding.content_model,
            device="cpu",
        )
        _model.encode(["warmup"], batch_size=1, show_progress_bar=False)
        log.info("Embedding model ready")
    return _model



def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_model()
    with Timer(log, "embed_batch", n=len(texts)):
        vecs = model.encode(
            texts,
            batch_size=CFG.embedding.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
    return vecs.tolist()


def build_colbert_tokens(text: str) -> list[dict]:
    model = get_model()
    DIM   = CFG.embedding.colbert_dim

    words = text.split()[:CFG.embedding.max_colbert_tokens]
    if not words:
        return []

    token_vecs_768 = model.encode(
        words,
        batch_size=min(len(words), CFG.embedding.batch_size),
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    projected = token_vecs_768[:, :DIM]
    norms     = np.linalg.norm(projected, axis=1, keepdims=True)
    norms     = np.where(norms == 0, 1.0, norms)
    projected = projected / norms

    return [{"v": row.tolist()} for row in projected]



def build_muvera_fde(colbert_token_vecs: list[dict]) -> list[float]:
    MUVERA_DIM  = CFG.embedding.muvera_dim
    COLBERT_DIM = CFG.embedding.colbert_dim
    K           = MUVERA_DIM // COLBERT_DIM  # = 8

    if not colbert_token_vecs:
        return [0.0] * MUVERA_DIM

    vecs = np.array(
        [t["v"] for t in colbert_token_vecs], dtype=np.float32
    )  # shape: (N, 128)

    buckets: list[list[np.ndarray]] = [[] for _ in range(K)]
    for vec in vecs:
        fingerprint = vec[:4].tobytes()
        bucket_idx  = int.from_bytes(
            hashlib.sha256(fingerprint).digest()[:2], "big"
        ) % K
        buckets[bucket_idx].append(vec)

    fde_parts = []
    for bucket in buckets:
        if bucket:
            stacked = np.stack(bucket)           # (m, 128)
            pooled  = stacked.max(axis=0)        # (128,)
        else:
            pooled = np.zeros(COLBERT_DIM, dtype=np.float32)

        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm
        fde_parts.append(pooled)

    fde  = np.concatenate(fde_parts).astype(np.float32)  # (1024,)
    norm = np.linalg.norm(fde)
    if norm > 0:
        fde = fde / norm

    return fde.tolist()


def embed_document(text: str) -> dict:
    content_vec    = embed_batch([text])[0]
    colbert_tokens = build_colbert_tokens(text)
    muvera_fde     = build_muvera_fde(colbert_tokens)
    return {
        "content_vector":  content_vec,
        "colbert_tokens":  colbert_tokens,
        "muvera_fde":      muvera_fde,
    }