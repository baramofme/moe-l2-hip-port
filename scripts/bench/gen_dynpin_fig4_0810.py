#!/usr/bin/env python3
"""重新生成 README 低内存模式配图（2026-08-10 口径）：
动态 pin 实测曲线（08-09 数据）+ 选择性 pin/on-demand 稳定值 + whole-pin 基线。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import csv
import numpy as np
import os

fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

BG = "#0d1117"; FG = "#e6edf3"; SUB = "#8b949e"; GRID = "#21262d"
BLUE = "#4dabf7"; RED = "#ff6b6b"; YELLOW = "#ffd43b"; GREEN = "#7ee787"; PURPLE = "#d2a8ff"

# 数据
rows = list(csv.reader(open("/opt/data/moe-l2/demo-assets/dynpin-v4-20260809/dynpin_rss.csv")))
ts = [float(r[0])/60 for r in rows[1:]]          # 秒 → 分钟
rss = [float(r[1])/1024 for r in rows[1:]]        # MB → GB
WHOLE_PIN = 84.0                                   # whole-pin 全量驻留
SP_PIN = 26.8                                      # 选择性 pin（08-10，v4_top100.map）
SP_ONDEMAND = 17.5                                 # on-demand 兜底（08-10）

OUT = "/opt/data/moe-l2/docs/demo"

# ── 主图：RSS 曲线 ──────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

# whole-pin 基线（横线）
ax.axhline(WHOLE_PIN, color=RED, linestyle="--", linewidth=1.5, alpha=0.9)
ax.text(0.5, WHOLE_PIN+1.5, "whole-pin 84 GB（全量驻留）", color=RED, fontsize=11)

# 选择性 pin / on-demand 稳定值（横线）
ax.axhline(SP_PIN, color=GREEN, linestyle=":", linewidth=1.5)
ax.text(0.5, SP_PIN+1.2, "selective pin 26.8 GB", color=GREEN, fontsize=10)
ax.axhline(SP_ONDEMAND, color=PURPLE, linestyle=":", linewidth=1.5)
ax.text(0.5, SP_ONDEMAND+1.2, "on-demand 17.5 GB", color=PURPLE, fontsize=10)

# 动态 pin 实测曲线
ax.plot(ts, rss, color=BLUE, linewidth=2)
ax.fill_between(ts, rss, 0, color=BLUE, alpha=0.12)

ax.set_xlabel("时间（分钟）", color=FG, fontsize=12)
ax.set_ylabel("常驻内存 RSS（GB）", color=FG, fontsize=12)
ax.set_title("DeepSeek-V4-Flash (157B) — 低内存模式 RSS 对比（RTX 4090 实测）",
             color=FG, fontsize=14, pad=12)
ax.set_ylim(0, 95)
ax.grid(color=GRID, linewidth=0.6, alpha=0.5)
ax.tick_params(colors=SUB, labelsize=11)
for s in ax.spines.values():
    s.set_color("#30363d")

# 标注
ax.annotate("动态 pin 实测：105 轮跨话题\nRSS 9.8 → 45 GB 封顶（08-09）",
            xy=(50, 45), xytext=(22, 62), color=BLUE, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=BLUE))
ax.text(1, 4, "bins-v0.4.0 · -c 2048 · 实测（08-09 动态 pin 曲线 / 08-10 选择性 pin 稳定值）",
        color=SUB, fontsize=9)

fig.tight_layout()
fig.savefig(f"{OUT}/fig4-dynpin-rss-curve.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("fig4-dynpin-rss-curve.png 已生成")
