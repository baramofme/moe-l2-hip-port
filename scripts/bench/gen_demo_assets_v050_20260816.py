#!/usr/bin/env python3
"""生成 moe-l2 演示素材三图（2026-08-16 v0.5.0 C 方案实测口径，深色演示风）。

数据来源：RTX 4090 全链路实测（bins-v0.5.0 + C 方案按领域换表，2026-08-16）：
- DS-V2-Lite Q2_K：标准全 GPU 23.3 GB / 65 t/s；moe-l2 ~10.1 GB / 127-137 t/s（约 2× 全 GPU）
- Qwen3.6-35B-A3B UD-IQ2_M：标准 8 GB 卡 OOM；moe-l2 ~5.4 GB / 20-34 t/s（混合领域）
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

BG = "#0d1117"
BLUE = "#4dabf7"
RED = "#ff6b6b"
YELLOW = "#ffd43b"
GRAY = "#8b949e"
WHITE = "#e6edf3"
GREEN = "#7ee787"
PURPLE = "#d2a8ff"

OUT = "/opt/data/moe-l2/examples/demo-assets/"

def style_ax(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color("#30363d")
    ax.tick_params(colors=GRAY, labelsize=11)
    ax.yaxis.label.set_color(GRAY)
    ax.xaxis.label.set_color(GRAY)
    ax.title.set_color(WHITE)

# ── 图1：Qwen VRAM 对比（标准 8G OOM，moe-l2 5.4 GB）─────────────────
fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=BG)
models = ["标准全 GPU\n(8 GB 卡 OOM)", "moe-l2\n(selective pin + C 方案)"]
vram = [8.0, 5.4]   # 标准条画到 8G 上限并标注 OOM
colors = [RED, BLUE]
bars = ax.barh(models, vram, color=colors, height=0.55)
ax.axvline(8, color=YELLOW, linestyle="--", linewidth=1.5)
ax.text(8, 1.55, "8 GB 卡上限", color=YELLOW, fontsize=10, ha="center")
ax.text(vram[0]+0.1, 1, "OOM\n装不下", color=RED, fontsize=11, va="center")
ax.text(vram[1]+0.1, 0, "20-34 t/s\n生成速度", color=BLUE, fontsize=11, va="center")
ax.set_xlim(0, 12)
ax.set_xlabel("GPU 显存 (GB)")
ax.set_title("Qwen3.6-35B-A3B (32B MoE) — 标准 vs moe-l2（RTX 4090 实测，2026-08-16）", fontsize=13, pad=12)
style_ax(ax)
plt.tight_layout()
plt.savefig(OUT + "fig1-qwen-vram.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

# ── 图2：DS VRAM 对比（23.3 → 10.1 GB，127-137 t/s）─────────────────
fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=BG)
models = ["标准全 GPU", "moe-l2\n(selective pin + C 方案)"]
vram = [23.3, 10.1]
bars = ax.barh(models, vram, color=[RED, BLUE], height=0.55)
ax.axvline(8, color=YELLOW, linestyle="--", linewidth=1.5)
ax.text(8, -0.55, "8 GB 卡上限", color=YELLOW, fontsize=10, ha="center")
ax.text(vram[0]+0.2, 1, "65 t/s", color=WHITE, fontsize=11, va="center")
ax.text(vram[1]+0.2, 0, "127-137 t/s\n(约 2× 全 GPU)", color=BLUE, fontsize=11, va="center")
ax.set_xlim(0, 27)
ax.set_xlabel("GPU 显存 (GB)")
ax.set_title("DeepSeek-V2-Lite (16B MoE) — 显存省 57%，速度反超（RTX 4090 实测，2026-08-16）", fontsize=13, pad=12)
style_ax(ax)
plt.tight_layout()
plt.savefig(OUT + "fig2-ds-vram.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

# ── 图3：汇总卡 ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
ax.set_xlim(0, 10); ax.set_ylim(0, 4)
ax.axis("off")

def card(x, y, w, h, title, big, sub, color):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor="#161b22", edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x+w/2, y+h-0.55, title, ha="center", va="center", color=GRAY, fontsize=12)
    ax.text(x+w/2, y+h/2, big, ha="center", va="center", color=color, fontsize=26, fontweight="bold")
    ax.text(x+w/2, y+0.5, sub, ha="center", va="center", color=WHITE, fontsize=10)

card(0.3, 0.8, 2.9, 2.4, "显存节省", "-57%", "23.3 → 10.1 GB\n(DS-V2-Lite)", GREEN)
card(3.6, 0.8, 2.9, 2.4, "生成速度", "127-137 t/s", "DS-V2-Lite 约 2× 全 GPU\n(标准 65 t/s)", BLUE)
card(6.9, 0.8, 2.9, 2.4, "显存/速度双达标", "2.3×", "23.3 GB 模型 / 10.1 GB 显存\nQwen 8 GB OOM → 5.4 GB 跑", PURPLE)
ax.text(5, 3.4, "moe-l2 — 低显存跑大 MoE（RTX 4090 实测，2026-08-16）", ha="center", color=WHITE, fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT + "fig3-summary.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

print("三图生成完成（v0.5.0 C 方案口径）")
for f in ["fig1-qwen-vram.png", "fig2-ds-vram.png", "fig3-summary.png"]:
    p = os.path.join(OUT, f)
    print(f, os.path.getsize(p), "bytes")
