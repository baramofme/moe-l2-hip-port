"""模式 B 数据飞轮（P2 ②，2026-08-02）。

proxy 边用边收集真实流量 → 累积 (prompt, 领域) 样本 → 增量重训
TF-IDF 分类器 → 越用越准。

设计：
- 样本库：~/.moe-l2/training_samples.jsonl（追加写，轻量）
- 触发：每 N 条新样本（默认 50）自动重训一次；也可手动 train 命令
- 重训：复用 train_classifier.py 的管线（TF-IDF + LinearSVC），
  用 种子数据（prototypes + DEFAULT_PROMPTS）+ 累积样本 一起训
- 输出：moe_l2/data/domain_classifier.joblib 原子替换
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("moe-l2-flywheel")

DEFAULT_SAMPLE_PATH = Path.home() / ".moe-l2" / "training_samples.jsonl"
# 触发重训的样本阈值：20 条（个人使用 2-3 天即可触发一次，重训秒级无感）
RETRAIN_EVERY_N = 20
MIN_SAMPLES_FOR_RETRAIN = 20

# 种子数据源（train_classifier.py 同款）
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "moe_l2" / "data"


def append_sample(
    prompt: str,
    domain: str,
    sample_path: Path = DEFAULT_SAMPLE_PATH,
) -> int:
    """记录一条 (prompt, domain) 样本，返回当前总样本数。

    domain 为 None / 空 / 未知标签时跳过（避免污染训练集）。
    """
    if not prompt or not domain:
        return 0
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sample_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": prompt, "domain": domain}, ensure_ascii=False) + "\n")
    return _count_samples(sample_path)


def _count_samples(sample_path: Path) -> int:
    if not sample_path.exists():
        return 0
    n = 0
    with open(sample_path, encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def load_samples(sample_path: Path = DEFAULT_SAMPLE_PATH) -> list[dict]:
    """读取全部累积样本（供重训）。"""
    if not sample_path.exists():
        return []
    samples = []
    with open(sample_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if d.get("prompt") and d.get("domain"):
                    samples.append(d)
            except json.JSONDecodeError:
                continue
    return samples


def maybe_retrain(
    sample_path: Path = DEFAULT_SAMPLE_PATH,
    force: bool = False,
) -> bool:
    """样本达到阈值时增量重训分类器。返回是否重训了。"""
    n = _count_samples(sample_path)
    if force:
        if n < MIN_SAMPLES_FOR_RETRAIN:
            logger.info("Only %d samples — skipping retrain", n)
            return False
    else:
        # 未到阈值 或 非整批点：跳过
        if n < MIN_SAMPLES_FOR_RETRAIN or n % RETRAIN_EVERY_N != 0:
            return False

    logger.info("Retraining classifier with %d accumulated samples...", n)
    try:
        _retrain(sample_path)
        logger.info("Classifier retrained OK")
        return True
    except Exception as e:
        logger.warning("Retrain failed (keeping old model): %s", e)
        return False


def _retrain(sample_path: Path) -> None:
    """用 种子 + 累积样本 重训，原子替换 joblib。"""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(DATA_DIR))
    sys.path.insert(0, str(ROOT / "moe_l2"))

    import joblib
    from collect import DEFAULT_PROMPTS  # type: ignore
    from prototypes import DOMAIN_PROTOTYPES  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    texts, labels = [], []
    for dom, protos in DOMAIN_PROTOTYPES.items():
        for p in protos:
            texts.append(p)
            labels.append(dom)
    for dom, prompts in DEFAULT_PROMPTS.items():
        for p in prompts:
            if p not in texts:
                texts.append(p)
                labels.append(dom)
    # 累积样本（真实流量，数据飞轮的核心）
    for s in load_samples(sample_path):
        texts.append(s["prompt"])
        labels.append(s["domain"])

    pipe = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1),
        LinearSVC(C=10, class_weight="balanced", max_iter=10000),
    )
    pipe.fit(texts, labels)

    out = DATA_DIR / "domain_classifier.joblib"
    tmp = out.with_suffix(".joblib.tmp")
    joblib.dump(pipe, tmp)
    os.replace(tmp, out)  # 原子替换，避免读到半成品
    logger.info("Model saved: %s (%d samples)", out, len(texts))


def training_stats(sample_path: Path = DEFAULT_SAMPLE_PATH) -> dict:
    """返回飞轮状态（供 /stats 展示）。"""
    n = _count_samples(sample_path)
    return {
        "flywheel_samples": n,
        "retrain_every": RETRAIN_EVERY_N,
        "next_retrain_at": RETRAIN_EVERY_N if n == 0 else ((n // RETRAIN_EVERY_N) + 1) * RETRAIN_EVERY_N,
    }
