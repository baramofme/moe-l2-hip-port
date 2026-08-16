#!/usr/bin/env python3
"""Render moe-l2 demo assets (fig1/fig2/fig3 + MP4) from new measured data.

Dark demo style per demo-assets-creation skill.
Usage: python3 render_demo.py <rec_qwen_dir> <rec_ds_dir> <out_dir>
"""
import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── CJK font ──
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

# ── palette (dark demo style) ──
BG = "#0d1117"
FG = "#e6edf3"
SUB = "#8b949e"
GRID = "#21262d"
MOE = "#4dabf7"
STD = "#ff6b6b"
Y8G = "#ffd43b"
GREEN = "#7ee787"
PURPLE = "#d2a8ff"


def load_rec(path):
    """Load rec_data.csv -> (t, vram_mib, tokens)."""
    t, v, toks = [], [], []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            t.append(float(parts[0]))
            v.append(float(parts[1]))
            toks.append(int(parts[2]))
    return np.array(t), np.array(v), np.array(toks)


def speed_at(t, toks, i, win=5):
    """Tokens/s over a window ending at index i."""
    if i < win:
        return 0.0
    dt = t[i] - t[i - win]
    if dt <= 0:
        return 0.0
    return (toks[i] - toks[i - win]) / dt


def compute_gen_speed(t, toks):
    """Overall generation speed: max_tokens / generation time.

    Handles both smooth and step-shaped token curves. The rec sampler writes
    one line per second reading the last token count from rec_gen.log; when
    the server streams the whole generation at once (buffered), the token
    curve is a step (0 -> max at one sample). In that case the first non-zero
    sample already sits at the completion time, so speed = total / t[first].
    """
    if len(toks) == 0 or toks[-1] <= 0:
        return 0.0
    first = np.argmax(toks > 0)
    if not any(toks > 0):
        return 0.0
    # step-shaped curve: first non-zero sample == completion point
    if t[first] > 0 and toks[first] == toks[-1]:
        return toks[first] / t[first]
    last = np.argmax(toks)
    dt = t[last] - t[first]
    if dt <= 0:
        return 0.0
    return toks[last] / dt


def fig_bar(ax, items, title, sub, unit_gb=True):
    """items: [(label, value_gb, color, is_std)] -> horizontal bars + 8GB line."""
    ax.set_facecolor(BG)
    labels = [it[0] for it in items]
    vals = [it[1] for it in items]
    colors = [it[2] for it in items]
    y = np.arange(len(items))[::-1]
    ax.barh(y, vals, color=colors, height=0.55, zorder=3)
    ax.axvline(8, color=Y8G, linestyle="--", linewidth=1.5, zorder=4)
    ax.text(8, len(items) - 0.35, " 8 GB 显卡上限", color=Y8G, fontsize=9, va="bottom")
    # over-8GB shading
    ax.axvspan(8, max(vals) * 1.08, color=STD, alpha=0.06, zorder=1)
    for i, (v, c) in enumerate(zip(vals, colors)):
        yy = len(items) - 1 - i
        txt = f"{v:.2f} GB" if unit_gb else f"{v:.0f} MiB"
        ax.text(v, yy, f"  {txt}", color=FG if v > 3 else "#0d1117",
                va="center", fontsize=11, fontweight="bold", zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=FG, fontsize=12)
    ax.set_xlim(0, max(max(vals) * 1.08, 10))
    ax.set_xlabel("显存占用 (GB)", color=SUB, fontsize=10)
    ax.tick_params(axis="x", colors=SUB)
    ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.set_title(title, color=FG, fontsize=16, fontweight="bold", pad=12)
    ax.text(0.0, 1.02, sub, transform=ax.transAxes, color=SUB, fontsize=9)


def render_figs(qwen_dir, ds_dir, out):
    # Qwen sample stats
    qt, qv, qtoks = load_rec(os.path.join(qwen_dir, "rec_data.csv"))
    dt_, dv, dtoks = load_rec(os.path.join(ds_dir, "rec_data.csv"))
    q_peak = qv.max()
    d_peak = dv.max()
    q_tps = compute_gen_speed(qt, qtoks)
    d_tps = compute_gen_speed(dt_, dtoks)

    # fig1: Qwen
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    fig.patch.set_facecolor(BG)
    fig_bar(ax, [
        ("标准形态（全专家上 GPU）", 8.0, STD, True),
        (f"moe-l2 C 方案（实测 {q_tps:.1f} t/s）", q_peak / 1024, MOE, False),
    ], "Qwen3.6-35B-A3B（32B MoE）显存对比",
        f"RTX 4090 实测 · v0.5.0 C 方案 · 标准 8GB 卡 OOM · moe-l2 峰值 {q_peak:.0f} MiB")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1-qwen-vram.png"), facecolor=BG,
                bbox_inches="tight")
    plt.close(fig)

    # fig2: DS
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    fig.patch.set_facecolor(BG)
    fig_bar(ax, [
        ("标准形态（全专家上 GPU）", 23.3, STD, True),
        (f"moe-l2 C 方案（实测 {d_tps:.1f} t/s）", d_peak / 1024, MOE, False),
    ], "DeepSeek-V2-Lite（16B MoE，64 experts）显存对比",
        f"RTX 4090 实测 · v0.5.0 C 方案 · 3200 tokens 长生成 · moe-l2 峰值 {d_peak:.0f} MiB")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2-ds-vram.png"), facecolor=BG,
                bbox_inches="tight")
    plt.close(fig)

    # fig3: summary (4 panels: vram bars + speed bars)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    fig.patch.set_facecolor(BG)
    fig_bar(axes[0], [
        ("Qwen 标准", 8.0, STD, True),
        ("Qwen moe-l2", q_peak / 1024, MOE, False),
        ("DS 标准", 23.3, STD, True),
        ("DS moe-l2", d_peak / 1024, MOE, False),
    ], "显存占用（越低越好）",
        f"标准形态用已发布数据（23.3 GB / ~12 GB）；moe-l2 为本次实测（{q_peak:.0f} / {d_peak:.0f} MiB）")
    # speed panel
    ax = axes[1]
    ax.set_facecolor(BG)
    spd = [
        ("Qwen 标准", 65.0 if False else None),  # standard Qwen: no speed (OOM on 8GB)
    ]
    labels = ["Qwen moe-l2", "DS moe-l2"]
    vals = [q_tps, d_tps]
    colors = [MOE, PURPLE]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=colors, height=0.5, zorder=3)
    for i, v in enumerate(vals):
        yy = len(labels) - 1 - i
        ax.text(v, yy, f"  {v:.1f} t/s", color="#0d1117", va="center",
                fontsize=12, fontweight="bold", zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=FG, fontsize=12)
    ax.set_xlim(0, max(vals) * 1.2)
    ax.set_xlabel("生成速度 (tokens/s)", color=SUB, fontsize=10)
    ax.tick_params(axis="x", colors=SUB)
    ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.set_title("生成速度（本次实测，越高越好）", color=FG, fontsize=16,
                 fontweight="bold", pad=12)
    ax.text(0.0, 1.02, "RTX 4090 · v0.5.0 C 方案 · 3200 tokens 长生成",
            transform=ax.transAxes, color=SUB, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3-summary.png"), facecolor=BG,
                bbox_inches="tight")
    plt.close(fig)
    print(f"figs done: q_peak={q_peak:.0f}MiB q_tps={q_tps:.1f} d_peak={d_peak:.0f}MiB d_tps={d_tps:.1f}")


def render_mp4(qwen_dir, out):
    """Qwen VRAM curve animation: 360 frames @ 8fps = 45s."""
    t, v, toks = load_rec(os.path.join(qwen_dir, "rec_data.csv"))
    with open(os.path.join(qwen_dir, "rec_full.txt")) as f:
        full = f.read()
    # skip thinking block for Qwen-style reasoning models
    body = full
    if "thinking" in body[:200].lower():
        idx = body.find("\n\n", 200)
        body = body[idx:] if idx != -1 else body[:800]
    n_frames = 360
    fps = 8
    frame_dir = os.path.join(out, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    v_max = max(8192, v.max() * 1.15)
    for k in range(n_frames):
        t_vid = k / fps
        i = int(np.searchsorted(t, t_vid) - 1)
        i = max(0, min(i, len(t) - 1))
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        fig.patch.set_facecolor(BG)
        # main curve
        ax = fig.add_axes([0.06, 0.10, 0.55, 0.78])
        ax.set_facecolor(BG)
        ax.plot(t[:i + 1], v[:i + 1], color=MOE, linewidth=2.5, zorder=3)
        ax.axhline(8192, color=Y8G, linestyle="--", linewidth=1.5, zorder=2)
        ax.text(0.02, 8300, "8 GB 显卡上限", color=Y8G, fontsize=10)
        ax.axhline(v.max(), color=GREEN, linestyle=":", linewidth=1.5, zorder=2)
        ax.text(0.02, v.max() + 80, f"moe-l2 峰值 {v.max():.0f} MiB", color=GREEN,
                fontsize=10)
        ax.fill_between(t[:i + 1], v[:i + 1], 0, color=MOE, alpha=0.12, zorder=1)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(0, v_max)
        ax.set_xlabel("时间 (s)", color=SUB, fontsize=10)
        ax.set_ylabel("显存 (MiB)", color=SUB, fontsize=10)
        ax.tick_params(colors=SUB)
        ax.grid(color=GRID, linewidth=0.5, zorder=0)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(GRID)
        ax.set_title("moe-l2 显存曲线 — Qwen3.6-35B-A3B 生成中（RTX 4090）",
                     color=FG, fontsize=14, fontweight="bold")
        # right panel
        ax2 = fig.add_axes([0.66, 0.10, 0.30, 0.78])
        ax2.set_facecolor(BG)
        ax2.axis("off")
        spd = speed_at(t, toks, i)
        info = [
            ("模型", "Qwen3.6-35B-A3B (32B MoE)"),
            ("引擎", "moe-l2 v0.5.0 (C 方案)"),
            ("GPU", "RTX 4090"),
            ("当前显存", f"{v[i]:.0f} MiB"),
            ("tokens", f"{toks[i]}"),
            ("速度", f"{spd:.1f} t/s"),
        ]
        for j, (k_, val) in enumerate(info):
            ax2.text(0.02, 0.92 - j * 0.14, f"{k_}：", color=SUB, fontsize=12,
                     va="top", fontfamily="WenQuanYi Zen Hei Mono")
            ax2.text(0.35, 0.92 - j * 0.14, val, color=FG, fontsize=12,
                     va="top", fontfamily="WenQuanYi Zen Hei Mono")
        # bottom text stream
        shown = int(toks[i] * len(body) / max(toks[-1], 1))
        window = body[max(0, shown - 150):shown]
        ax3 = fig.add_axes([0.06, 0.02, 0.90, 0.06])
        ax3.set_facecolor(BG)
        ax3.axis("off")
        ax3.text(0.0, 0.5, window, color=FG, fontsize=10, va="center",
                 fontfamily="WenQuanYi Zen Hei Mono")
        fig.savefig(os.path.join(frame_dir, f"frame_{k:04d}.png"),
                    facecolor=BG)
        plt.close(fig)
        if k % 90 == 0:
            print(f"  frame {k}/{n_frames}")
    print("frames done")
    os.system(
        f"ffmpeg -y -framerate {fps} -i {frame_dir}/frame_%04d.png "
        f"-c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "
        f"{os.path.join(out, 'demo-vram-animation.mp4')} 2>/dev/null"
    )
    print("mp4 done")


if __name__ == "__main__":
    qwen_dir, ds_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    render_figs(qwen_dir, ds_dir, out)
    render_mp4(qwen_dir, out)
