"""轻量 TF-IDF 领域分类器（P2 ①，2026-08-02）。

用 collect 种子数据（prototypes + DEFAULT_PROMPTS，111 条）训练的
TF-IDF + LinearSVC 分类器。仅当 scikit-learn 已安装且
domain_classifier.joblib 存在时才可用；否则 predict() 返回 None，
调用方（predictor.py）自动回退到关键词 / 语义方案。

推理延迟：~1-3ms（char_wb TF-IDF + 线性 SVC，无需下载模型文件）。

Usage:
    from moe_l2.tfidf_predictor import TfidfPredictor
    tp = TfidfPredictor()
    tp.predict("write a python function")  # → "codegen" or None
    tp.predict_with_scores("print hello")  # → {"codegen": 0.8, ...}
"""

from __future__ import annotations

import os
from typing import Optional

__all__ = ["TfidfPredictor", "is_tfidf_available"]

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "domain_classifier.joblib")


def is_tfidf_available() -> bool:
    """Check if the trained classifier artifact exists (sklearn assumed)."""
    return os.path.exists(_MODEL_PATH)


class TfidfPredictor:
    """TF-IDF + LinearSVC domain classifier (zero-download, ~237 KB model)."""

    def __init__(self, model_path: Optional[str] = None):
        import joblib

        path = model_path or _MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"domain_classifier.joblib not found at {path}. "
                "Run `python train_classifier.py` first."
            )
        self._pipe = joblib.load(path)
        # LinearSVC exposes decision_function → confidence margin per class
        self._classes = list(self._pipe.classes_)

    def predict(self, prompt: str, confidence_threshold: float = 0.0) -> Optional[str]:
        """Classify prompt to a domain, or None if below threshold.

        Uses the LinearSVC decision margin (distance to hyperplane).
        Returns None when the top class margin is below confidence_threshold.
        """
        pred = self._pipe.predict([prompt])[0]
        if confidence_threshold <= 0.0:
            return pred
        margins = self._pipe.decision_function([prompt])[0]
        top_margin = float(max(margins))
        if top_margin >= confidence_threshold:
            return pred
        return None

    def predict_with_scores(self, prompt: str) -> dict[str, float]:
        """Return all domain scores (decision margins) for threshold tuning."""
        margins = self._pipe.decision_function([prompt])[0]
        return dict(sorted(
            zip(self._classes, (float(m) for m in margins)),
            key=lambda kv: -kv[1],
        ))
