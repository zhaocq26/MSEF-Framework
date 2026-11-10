"""Semantic encoders mapping memory content to dense vectors.

Produces fixed-dimension dense vectors v = E(content), v ∈ R^d used for
semantic similarity retrieval and consistency scoring. Default dimension is
384 (Sentence-BERT all-MiniLM-L6-v2). Two implementations:

- ``SemanticEncoder``: prefers sentence-transformers, emitting 384-dim
  embeddings, and falls back automatically when the package is missing.
- ``deterministic_embed``: feature hashing over character n-grams with
  PyTorch L2 normalization; reproducible offline with no external model.
"""
from __future__ import annotations

import hashlib
import re
from typing import List

import torch


class SemanticEncoder:
    """Unified interface for semantic encoding.

    Args:
        dim: Output dimension (default 384).
        use_sentence_transformers: Prefer Sentence-BERT when available (default True).
    """

    def __init__(self, dim: int = 384, use_sentence_transformers: bool = True) -> None:
        self.dim = dim
        self._st = None
        if use_sentence_transformers:
            self._st = self._try_load_st()

    def _try_load_st(self):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:  # pragma: no cover - silently fall back when dependency is missing
            return None

    def encode(self, texts) -> torch.Tensor:
        """Encode one or more texts, returning an [N, dim] tensor."""
        if isinstance(texts, str):
            texts = [texts]
        if self._st is not None:
            vecs = self._st.encode(
                texts, convert_to_tensor=True, normalize_embeddings=True
            )
            if vecs.shape[1] != self.dim:  # pragma: no cover - fallback on dimension mismatch
                vecs = torch.nn.functional.pad(
                    vecs, (0, self.dim - vecs.shape[1]), value=0.0
                )
            return vecs
        return deterministic_embed(texts, dim=self.dim)

    def cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Cosine similarity matrix [N_a, N_b] between two vector sets."""
        a = torch.nn.functional.normalize(a, dim=-1)
        b = torch.nn.functional.normalize(b, dim=-1)
        return torch.mm(a, b.t())

    @property
    def use_sentence_transformer(self) -> bool:
        return self._st is not None


def _text_features(text: str) -> List[str]:
    """Extract text features: character n-grams plus words (with weight tags).

    Features map into a fixed hash vocabulary (no external model), keeping the
    result reproducible offline. Feature weights: unigram 1, bigram 2, trigram
    3, English word/cluster 3 — longer shared features contribute more to
    similarity, pulling semantically close entries closer in vector space.
    """
    text = re.sub(r"\s+", " ", text.lower()).strip()
    chars = re.findall(r"[\u4e00-\u9fff]|[a-z0-9_]", text)
    feats: List[str] = []
    # unigram/bigram/trigram (Chinese-dominant; English by word)
    for i, ch in enumerate(chars):
        feats.append(f"1:{ch}")
        if i + 1 < len(chars):
            feats.append(f"2:{ch}{chars[i+1]}")
        if i + 2 < len(chars):
            feats.append(f"3:{ch}{chars[i+1]}{chars[i+2]}")
    # English/numeric word clusters
    for tok in re.findall(r"[a-z0-9_]{2,}", text):
        feats.append(f"w:{tok}")
        if len(tok) >= 3:
            feats.append(f"2:{tok[:2]}{tok[2:4]}")
    return feats


def deterministic_embed(texts: List[str], dim: int = 384) -> torch.Tensor:
    """Deterministic feature-hash embedding (PyTorch).

    Each feature is hashed to a fixed index and accumulated with positive
    weights (avoiding cancellation from sign flips); longer features
    (bigrams/trigrams/words) weigh more, then L2-normalized. More shared
    n-grams yield higher cosine similarity — a character-level TF cosine —
    bringing semantically close entries nearer in vector space.
    """
    weight = {"1:": 1.0, "2:": 2.0, "3:": 3.0, "w:": 3.0}
    out = []
    for t in texts:
        vec = torch.zeros(dim, dtype=torch.float32)
        feats = _text_features(t)
        if not feats:
            vec[0] = 1.0
            out.append(vec)
            continue
        for f in feats:
            h = hashlib.md5(f.encode("utf-8")).hexdigest()
            idx = int(h[:8], 16) % dim
            cat = f[:2] if f[:2] in weight else f[:3]
            vec[idx] += weight.get(cat, 1.0)
        norm = vec.norm()
        if norm > 1e-8:
            vec = vec / norm
        out.append(vec)
    return torch.stack(out)


def token_estimate(text: str) -> int:
    """Estimate the token count of a text (encoder helper)."""
    return len(re.findall(r"[\u4e00-\u9fff]", text)) + len(
        re.findall(r"[a-z0-9_]{2,}", text.lower())
    )