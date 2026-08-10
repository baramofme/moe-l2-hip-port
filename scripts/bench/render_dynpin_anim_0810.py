#!/usr/bin/env python3
"""dynpin GIF/MP4 动画（2026-08-10 口径）：105 轮 RSS 曲线逐帧绘制 + whole-pin/selective/on-demand 参考线。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.animation import FuncAnimation
import csv
import numpy as np
import os
import sys

fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

BG = "#0d1117"; FG = "#e6edf3"; SUB = "#8b949e"; GRID = "#21262d"
BLUE = "#4dabf7"; RED = "#ff6b6b"; YELLOW = "#ffd43b"; GREEN = "#7ee787"; PURPLE = "#d2a8ff"

OUT = "/opt/data/moe-l2/demo-assets/dynpin-v4-20260809"
CSV = sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/dynpin_rss.csv"

rows = list(csv.reader(open(CSV)))
t = [float(r[0])/60 for r in rows[1:]]       # 秒→分钟
rss = [float(r[1])/1024 for r in rows[1:]]   # MB→GB
T = max(t)
WHOLE_PIN = 84.0
SP_PIN = 26.8
SP_ONDEMAND = 17.5

N_FRAMES = 240   # 30s @ 8fps
FPS = 8

fig, ax = plt.subplots(figsize=(11, 6.2))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

def draw_baselines():
    ax.axhline(WHOLE_PIN, color=RED, linestyle="--", linewidth=1.5, alpha=0.9)
    ax.text(0.5, WHOLE_PIN+1.5, "whole-pin 84 GB", color=RED, fontsize=11)
    ax.axhline(SP_PIN, color=GREEN, linestyle=":", linewidth=1.5)
    ax.text(0.5, SP_PIN+1.2, "selective pin 26.8 GB", color=GREEN, fontsize=10)
    ax.axhline(SP_ONDEMAND, color=PURPLE, linestyle=":", linewidth=1.5)
    ax.text(0.5, SP_ONDEMAND+1.2, "on-demand 17.5 GB", color=PURPLE, fontsize=10)

draw_baselines()
ax.set_xlim(0, T)
ax.set_ylim(0, 95)
ax.set_xlabel("时间（分钟）", color=FG, fontsize=12)
ax.set_ylabel("常驻内存 RSS（GB）", color=FG, fontsize=12)
ax.set_title("DeepSeek-V4-Flash (157B) — 动态 pin 105 轮 RSS 曲线（RTX 4090 实测）",
             color=FG, fontsize=14, pad=12)
ax.grid(color=GRID, linewidth=0.6, alpha=0.5)
ax.tick_params(colors=SUB, labelsize=11)
for s in ax.spines.values():
    s.set_color("#30363d")

line, = ax.plot([], [], color=BLUE, linewidth=2, zorder=5)
status = ax.text(0.98*T, 6, "", color=FG, fontsize=12, ha="right",
                 bbox=dict(facecolor="#161b22", edgecolor="none", alpha=0.9))
fill = None

def animate(i):
    global fill
    idx = int((i+1) / N_FRAMES * len(t))
    idx = max(1, min(idx, len(t)))
    tt = t[:idx]; vv = rss[:idx]
    line.set_data(tt, vv)
    if fill is not None:
        fill.remove()
    fill = ax.fill_between(tt, vv, 0, color=BLUE, alpha=0.12, zorder=4)
    status.set_text(f"第 {idx} 个采样点 · RSS {vv[-1]:.1f} GB\n（峰值 {max(vv):.1f} GB）")
    return line, status

anim = FuncAnimation(fig, animate, frames=N_FRAMES, interval=125, blit=False)
anim.save(f"{OUT}/dynpin-curve.mp4", writer="ffmpeg", fps=FPS, dpi=100,
          bitrate=1000, savefig_kwargs={"facecolor": BG})
print(f"MP4: {OUT}/dynpin-curve.mp4")

# GIF（pillow writer，6fps，24s）
N_GIF = 144
gfig, gax = plt.subplots(figsize=(10, 5.2))
gfig.patch.set_facecolor(BG); gax.set_facecolor(BG)
draw_baselines()
gax.set_xlim(0, T); gax.set_ylim(0, 95)
gax.set_xlabel("时间（分钟）", color=FG, fontsize=12)
gax.set_ylabel("常驻内存 RSS（GB）", color=FG, fontsize=12)
gax.set_title("DeepSeek-V4-Flash (157B) — 动态 pin 105 轮 RSS 曲线", color=FG, fontsize=13, pad=10)
gax.grid(color=GRID, linewidth=0.6, alpha=0.5)
gax.tick_params(colors=SUB, labelsize=10)
for s in gax.spines.values():
    s.set_color("#30363d")
gline, = gax.plot([], [], color=BLUE, linewidth=2, zorder=5)

def ganimate(i):
    idx = max(1, min(int((i+1) / N_GIF * len(t)), len(t)))
    gline.set_data(t[:idx], rss[:idx])
    return gline,

ganim = FuncAnimation(gfig, ganimate, frames=N_GIF, interval=150, blit=False)
ganim.save(f"{OUT}/dynpin-curve.gif", writer="pillow", fps=6)
print(f"GIF: {OUT}/dynpin-curve.gif")
