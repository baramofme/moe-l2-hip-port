#!/usr/bin/env python3
"""轻量分类器骨架（P2 ①）— 2026-08-02
用 collect 种子数据（prototypes.py 87 条 + collect DEFAULT_PROMPTS 24 条）训 TF-IDF + 线性分类器。

设计：
- 特征：TF-IDF（char_wb 2-4 gram，中英混合友好；CJK 子串特征由 char n-gram 天然覆盖）
- 模型：LinearSVC（多分类，one-vs-rest）
- 评估：StratifiedKFold 5 折交叉验证（样本少，看稳定区间）
- 产物：moe_l2/data/domain_classifier.joblib（~20-50KB）

运行时：predict_tfidf(prompt) → (domain, confidence)；无 sklearn 时优雅降级回关键词。
"""
import os, sys, json
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
import joblib

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "moe_l2", "data")

# 种子数据 1：prototypes.py（87 条）
sys.path.insert(0, DATA_DIR)
from prototypes import DOMAIN_PROTOTYPES

# 种子数据 2：collect DEFAULT_PROMPTS（24 条，去重）
sys.path.insert(0, os.path.join(ROOT, "moe_l2"))
from collect import DEFAULT_PROMPTS

def build_dataset():
    texts, labels = [], []
    for dom, protos in DOMAIN_PROTOTYPES.items():
        for p in protos:
            texts.append(p); labels.append(dom)
    for dom, prompts in DEFAULT_PROMPTS.items():
        for p in prompts:
            if p not in texts:
                texts.append(p); labels.append(dom)
    return texts, labels

def evaluate():
    texts, labels = build_dataset()
    print(f"数据集: {len(texts)} 条, 域分布:")
    cnt = Counter(labels)
    for d in sorted(cnt): print(f"  {d}: {cnt[d]}")

    # char_wb 2-4 gram：中文按字切分，英文按词边界切分，混合友好
    pipe = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1),
        LinearSVC(C=10, class_weight="balanced", max_iter=10000),
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, texts, labels, cv=skf, scoring="accuracy")
    print(f"\n5 折交叉验证准确率: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")
    print(f"  各折: {[f'{s*100:.1f}%' for s in scores]}")
    return pipe, texts, labels

def train_full(pipe, texts, labels):
    pipe.fit(texts, labels)
    out = os.path.join(DATA_DIR, "domain_classifier.joblib")
    joblib.dump(pipe, out)
    size = os.path.getsize(out) / 1024
    print(f"\n模型已保存: {out} ({size:.1f} KB)")

    # 自测：train 集准确率 + 几个典型样本
    train_acc = pipe.score(texts, labels)
    print(f"训练集准确率: {train_acc*100:.1f}%")
    for q in ["print hello world in python", "计算一下 15 乘 37", "Fix this segfault",
              "什么是神经网络", "Write a poem", "Translate this to French",
              "If A implies B, what follows", "what is machine learning"]:
        pred = pipe.predict([q])[0]
        print(f"  [{pred:>13}] {q}")

if __name__ == "__main__":
    pipe, texts, labels = evaluate()
    train_full(pipe, texts, labels)
