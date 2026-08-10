#!/usr/bin/env python3
"""选择性 pin（selective pin）成果图（2026-08-10 实测）：
DeepSeek-V4-Flash 三种模式 RSS/速度对比——当前最新主路径。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import numpy as np

fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

BG = "#0d1117"; FG = "#e6edf3"; SUB = "#8b949e"; GRID = "#21262d"
BLUE = "#4dabf7"; RED = "#ff6b6b"; YELLOW = "#ffd43b"; GREEN = "#7ee787"; PURPLE = "#d2a8ff"

OUT = "/opt/data/moe-l2/docs/demo"

# 数据（2026-08-10 实测，RTX 4090，bins-v0.4.0，V4 UD-IQ2_M 85GB）
MODE = ["whole-pin\n（08-09 默认）", "selective pin\n（08-10 主路径）", "on-demand\n（08-10 兜底）"]
RSS = [84.0, 26.8, 17.5]
TPS = [30.9, 34.67, 35.96]
COLORS = [RED, GREEN, BLUE]

# ── 图1：RSS 对比（主图） ──────────────────────
fig, ax = plt.subplots(figsize=(10, 5.4))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
bars = ax.bar(MODE, RSS, color=COLORS, width=0.55, zorder=3)
for b, v, tps in zip(bars, RSS, TPS):
    ax.text(b.get_x()+b.get_width()/2, v+1.5, f"{v} GB", ha="center", color=FG,
            fontsize=15, fontweight="bold")
    ax.text(b.get_x()+b.get_width()/2, v-6, f"{tps} t/s", ha="center", color=BG,
            fontsize=12, fontweight="bold")
ax.axhline(32, color=YELLOW, linestyle="--", linewidth=1.2)
ax.text(2.42, 33.5, "16–32 GB 目标区间", color=YELLOW, fontsize=11, ha="right")
ax.set_ylim(0, 95)
ax.set_ylabel("常驻内存 RSS（GB）", color=FG, fontsize=12)
ax.set_title("DeepSeek-V4-Flash (157B) — selective pin 低内存（RTX 4090 实测 2026-08-10）",
             color=FG, fontsize=14, pad=12)
ax.grid(color=GRID, linewidth=0.6, alpha=0.5, axis="y")
ax.tick_params(colors=SUB, labelsize=11)
for s in ax.spines.values():
    s.set_color("#30363d")
ax.text(0.02, 0.95, "bins-v0.4.0 · 路由表驱动 top-K pin · RSS 84 → 26.8 GB（↓68%）· 速度反升 34.67 t/s",
        color=SUB, fontsize=9, transform=ax.transAxes, va="top")
fig.tight_layout()
fig.savefig(f"{OUT}/fig5-selective-pin-rss.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("fig5-selective-pin-rss.png 已生成")

# ── 图2：速度 vs 内存散点（双指标） ─────────────
fig, ax = plt.subplots(figsize=(8.5, 5.4))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
for rss, tps, c, m in zip(RSS, TPS, COLORS, ["whole-pin", "selective pin", "on-demand"]):
    ax.scatter(rss, tps, s=180, color=c, zorder=5, edgecolor=FG, linewidth=0.8)
    ax.annotate(f"{m}\n{RSS[RSS.index(rss)] if m=='whole-pin' else (RSS[1] if m=='selective pin' else RSS[2])} GB / {tps} t/s",
                (rss, tps), textcoords="offset points", xytext=(12, 8), color=FG, fontsize=10)
ax.set_xlabel("常驻内存 RSS（GB）", color=FG, fontsize=12)
ax.set_ylabel("生成速度（t/s）", color=FG, fontsize=12)
ax.set_title("selective pin — 内存降 68% 速度反升（V4 / 4090 / 08-10）", color=FG, fontsize=14, pad=12)
ax.set_xlim(0, 95); ax.set_ylim(25, 40)
ax.grid(color=GRID, linewidth=0.6, alpha=0.5)
ax.tick_params(colors=SUB, labelsize=11)
for s in ax.spines.values():
    s.set_color("#30363d")
ax.axvline(32, color=YELLOW, linestyle="--", linewidth=1)
ax.text(33, 26.5, "目标区间 16–32 GB", color=YELLOW, fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5b-selective-pin-speed-rss.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("fig5b-selective-pin-speed-rss.png 已生成")
