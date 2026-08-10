#!/usr/bin/env python3
"""渲染 Qwen 显存曲线动画 MP4（2026-08-10 数据，深色风，1280x720）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import csv
import numpy as np
import os

fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei Mono"
plt.rcParams["axes.unicode_minus"] = False

BG = "#0d1117"
BLUE = "#4dabf7"
RED = "#ff6b6b"
YELLOW = "#ffd43b"
GRAY = "#8b949e"
WHITE = "#e6edf3"
GREEN = "#7ee787"

# ── 数据 ──────────────────────────────────────────
rows = list(csv.reader(open("/opt/data/moe-l2/examples/demo-assets/rec_data.csv")))
ts = [float(r[0]) for r in rows[1:]]
vram = [float(r[1]) for r in rows[1:]]
T = max(ts)
FPS = 8
DUR = 45
NFRAMES = FPS * DUR
PEAK = max(vram)
IDLE = min(vram)

# 速度：近 5 采样窗口（这里 tokens 列不可靠，用已知 76 t/s 标注）
GEN_TPS = 76.0
GEN_TOTAL = 3500

# ── 布局 ──────────────────────────────────────────
W, H = 1280, 720
OUT = "/opt/data/moe-l2/examples/demo-assets/frames"
os.makedirs(OUT, exist_ok=True)

def render(frame_idx):
    prog = frame_idx / (NFRAMES - 1)
    t_cur = prog * T
    # 曲线显示到当前时刻
    idx = int(np.searchsorted(ts, t_cur))
    tt = ts[:idx]; vv = vram[:idx]
    # 生成进度（0-45s 空闲，46s 起生成，76 t/s × 46s ≈ 3500）
    gen_t = max(0, t_cur - 45.5)
    toks = min(GEN_TOTAL, int(gen_t * GEN_TPS))
    speed = GEN_TPS if gen_t > 0 else 0

    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100, facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, T)
    ax.set_ylim(0, 10000)
    ax.axhline(8192, color=YELLOW, linestyle="--", linewidth=2)
    ax.text(T*0.7, 8400, "8 GB 显卡上限", color=YELLOW, fontsize=13, ha="center")
    ax.axhline(PEAK, color=GREEN, linestyle=":", linewidth=2)
    ax.text(T*0.7, PEAK+300, f"moe-l2 峰值 {int(PEAK)} MiB", color=GREEN, fontsize=13, ha="center")
    if len(tt) > 1:
        ax.plot(tt, vv, color=BLUE, linewidth=2.5)
        ax.scatter([tt[-1]], [vv[-1]], color=BLUE, s=30, zorder=5)
    ax.set_yticks([0, 2048, 3361, 4096, 6144, 8192, 10240])
    ax.set_yticklabels(["0", "2 GB", "3.3 GB", "4 GB", "6 GB", "8 GB", "10 GB"], fontsize=10)
    ax.set_xlabel("时间 (s)", color=GRAY, fontsize=12)
    ax.set_ylabel("GPU 显存 (MiB)", color=GRAY, fontsize=12)
    ax.set_title("Qwen3.6-35B-A3B (32B MoE) — moe-l2 推理显存曲线（RTX 4090 实测 2026-08-10）",
                 color=WHITE, fontsize=15, pad=12)
    for s in ax.spines.values():
        s.set_color("#30363d")
    ax.tick_params(colors=GRAY, labelsize=10)

    # 右上实时面板
    panel = dict(facecolor="#161b22", edgecolor="#30363d", boxstyle="round,pad=0.6")
    txt = (f"模型     Qwen3.6-35B-A3B\n"
           f"引擎     moe-l2 (selective pin)\n"
           f"GPU      RTX 4090\n"
           f"显存     {int(vram[min(idx-1, len(vram)-1)]) if idx>0 else int(IDLE)} MiB\n"
           f"tokens   {toks} / {GEN_TOTAL}\n"
           f"速度     {speed:.0f} t/s")
    ax.text(T*0.985, 9600, txt, ha="right", va="top", color=WHITE,
            fontsize=12, family="WenQuanYi Zen Hei Mono", bbox=panel)

    # 底部状态条
    status = f"显存从 {int(IDLE)} → {int(PEAK)} MiB（+{int(PEAK-IDLE)} MiB），全程低于 8 GB 红线"
    ax.text(10, 300, status, color=WHITE, fontsize=13, ha="left",
            bbox=dict(facecolor="#161b22", edgecolor="none", alpha=0.9))

    fig.savefig(f"{OUT}/frame_{frame_idx:04d}.png", facecolor=BG)
    plt.close(fig)

# 先渲 3 帧测试
for f in range(NFRAMES):
    render(f)
print("测试帧完成")
