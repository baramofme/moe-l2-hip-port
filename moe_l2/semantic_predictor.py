"""Optional semantic domain predictor using sentence-transformers embeddings.

This module is only loaded when `sentence-transformers` is installed.
It provides a fallback for prompts that keyword matching can't classify,
using cosine similarity to domain prototype embeddings.

Usage:
    from moe_l2.predictor import SemanticPredictor
    sp = SemanticPredictor()
    sp.predict("write a sorting function")  # returns "codegen" or None
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.preprocessing import normalize

from .data.prototypes import DOMAIN_PROTOTYPES

__all__ = ["SemanticPredictor"]


class SemanticPredictor:
    """Embedding-based domain classifier using sentence-transformers + prototypes.

    Downloads all-MiniLM-L6-v2 (~80MB) on first instantiation.
    Inference: ~10-30ms on CPU.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._domain_names: list[str] = []
        self._domain_embeddings: list[np.ndarray] = []
        self._confidence_threshold = 0.35

        for domain, examples in DOMAIN_PROTOTYPES.items():
            embs = normalize(self._model.encode(examples))
            self._domain_names.append(domain)
            self._domain_embeddings.append(embs)

    def predict(self, prompt: str) -> Optional[str]:
        """Classify prompt to a domain, or None if below confidence threshold.

        Uses max-similarity over all prototypes for each domain.
        Returns None when no domain reaches the confidence threshold.
        """
        emb = normalize(self._model.encode([prompt]))

        best_domain = None
        best_score = -1.0
        for i, embs in enumerate(self._domain_embeddings):
            sims = emb @ embs.T
            max_sim = float(np.max(sims[0]))
            if max_sim > best_score:
                best_score = max_sim
                best_domain = self._domain_names[i]

        if best_score >= self._confidence_threshold:
            return best_domain
        return None

    def predict_with_scores(self, prompt: str) -> dict[str, float]:
        """Return all domain scores for debugging / threshold tuning."""
        emb = normalize(self._model.encode([prompt]))
        scores: dict[str, float] = {}
        for i, embs in enumerate(self._domain_embeddings):
            sims = emb @ embs.T
            scores[self._domain_names[i]] = float(np.max(sims[0]))
        return dict(sorted(scores.items(), key=lambda x: -x[1]))
