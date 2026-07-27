# moe-l2

[**中文**](README_zh.md) | English

[![PyPI version](https://img.shields.io/pypi/v/moe-l2)](https://pypi.org/project/moe-l2/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

**Got an 8 GB GPU? Run 32B MoE models — 91% less VRAM, one pip install.**

| Your GPU | Normally fits | **With moe-l2** |
|----------|--------------|-----------------|
| 4 GB | — | DeepSeek-V2-Lite (16B MoE) ✅ |
| **8 GB** | 7B dense | **Qwen2.5-32B-A3B (32B MoE) ✅** |
| 12 GB | 13B dense | DeepSeek-V2 (236B MoE) ✅ |
| 24 GB | 34B dense | DeepSeek-V2 (236B MoE) ✅ |

```bash
pip install moe-l2
moe-l2 download-bins
moe-l2 start --model model.gguf --l2-size 4GB
```

Your tools (curl, Open WebUI, LangChain) connect to `localhost:11435` — no client changes needed.

---

## What moe-l2 does for you

MoE (Mixture-of-Experts) models pack dozens to hundreds of "experts" but only use a few per token. Standard inference loads **every** expert into GPU VRAM — wasting 80-95% on idle weights.

**moe-l2 keeps only the active experts on your GPU. The rest stay in system RAM or SSD, swapped in on demand.**

### Real-world benchmark

Tested on **DeepSeek-V2-Lite** (16B, 64 experts, top-6, Q2_K):

| Mode | GPU VRAM | Speed | What it means |
|------|----------|-------|---------------|
| Standard (all experts on GPU) | 23.3 GB | 65 t/s | Needs a 24 GB card |
| **moe-l2** (hot-cached experts) | **2.2 GB** | **8.6 t/s** | **Fits in 4 GB cards** |
| **Savings** | **91% less** | 13% speed | ~20 GB freed for other work |

Without moe-l2, an 8 GB card **cannot load this model at all** — it OOMs immediately. With moe-l2, it uses 2.2 GB and leaves 5.8 GB for other tasks.

> 8.6 t/s is the current Phase 2 measurement (experts on CPU, loaded via PCIe every step). The GPU LRU cache (next phase) targets **40+ t/s** by keeping hot experts in VRAM.

---

## System architecture

```
                         ┌─────────────────────────┐
  Your prompt ──────────▶│  moe-l2 Proxy (:11435)    │
                         │                          │
                         │  ┌─────────────────────┐ │
                         │  │ Domain Predictor    │ │
                         │  │ (keyword + semantic) │ │
                         │  └────────┬────────────┘ │
                         │           │ predicted     │
                         │           ▼ domain        │
                         │  ┌─────────────────────┐ │
                         │  │ L2 Cache (RAM)      │ │
                         │  │ LRU · mmap · async  │ │
                         │  │ preload from SSD    │ │
                         │  └────────┬────────────┘ │
                         │           │ forward       │
                         └───────────┼───────────────┘
                                     ▼
                         ┌─────────────────────────┐
                         │  llama.cpp / ollama      │
                         │  (:11434, CUDA GPU)      │
                         │  Active experts only     │
                         └─────────────────────────┘
```

### The 4-tier storage model

```
L0 ─ CPU Router     Gate routing + domain classification (your CPU)
 ↑
L1 ─ GPU VRAM       Active inference — experts + KV cache (your GPU)
 ↑
L2 ─ RAM Hot Cache  Domain-Aware LRU cache, mmap shared memory (this project)
 ↑
L3 ─ SSD Cold Store  Full expert weights, demand-loaded from disk
```

Close to GPU = fast but small; far from GPU = slow but cheap. The scheduler keeps active data in fast tiers and pushes everything else down.

---

## When to use it

### ✅ Great for
- **Single-user local chat** with large MoE models on 4-12 GB GPUs
- **Testing and experimenting** with MoE architectures on budget hardware
- **Homelab / edge deployments** where every GB of VRAM counts
- **Research on expert caching**, tiered scheduling, and domain-aware preloading

### ❌ Not for
- High-throughput API serving (frequent expert swaps create I/O bottlenecks)
- Latency-sensitive real-time apps (SSD cache misses cause speed dips)
- Mechanical hard drives as storage — **NVMe SSD required**

---

## Quick start

### 1. Install

```bash
pip install moe-l2
```

### 2. Download GPU binaries (optional, only for `--gpu` mode)

```bash
moe-l2 download-bins
```
This fetches the pre-built CUDA-enabled llama-server from GitHub Releases (~530 MB).

### 3. Start

**CPU mode** (expert cache only, no GPU savings):
```bash
moe-l2 start --model /path/to/model.gguf --l2-size 4GB
```

**GPU mode** (A3-patched llama-server, saves 91% VRAM):
```bash
moe-l2 start --model /path/to/model.gguf --l2-size 4GB --gpu
```

The proxy starts on `localhost:11435`. Send requests as if it were a regular OpenAI / Ollama endpoint.

### 4. Check stats

```bash
moe-l2 stats
# → Hit rate: 85% · Slots used: 320/960 · Active domain: codegen
```

---

## CLI reference

| Command | Description |
|---------|-------------|
| `moe-l2 start --model <path> --l2-size <size>` | Start proxy + cache |
| `moe-l2 start --model <path> --gpu` | Start with GPU-accelerated llama-server |
| `moe-l2 stats --port <port>` | Show live cache stats |
| `moe-l2 download-bins [--release TAG]` | Download pre-built GPU binaries |
| `moe-l2 stop --port <port>` | Stop proxy |

Options:
- `--model auto`: scan `/opt/data/models/*.gguf`
- `--l2-size 4GB` / `--l2-size 512MB`: target cache size
- `--port 11435` (default)
- `--gpu`: enable GPU mode (requires CUDA + NVIDIA GPU)

> **GPU binaries**: Not tracked in git (~530 MB). Fetched at runtime via `moe-l2 download-bins`.

---

## How it works (brief)

1. Your prompt hits the moe-l2 proxy
2. The domain predictor classifies it (codegen → math → chinese_tech ...)
3. L2 cache preloads predicted experts from SSD into shared memory (`/dev/shm/`)
4. Request is forwarded to llama.cpp/ollama — hot experts load from RAM (~1150 µs) instead of cold SSD (~6500 µs)
5. Cache hit rate typically exceeds 85% within the same session

---

## Platform requirements

- **Linux x86_64 only** — pre-built binaries target Linux AMD64 (CUDA `.so` + `llama-server`)
- macOS, Windows, and ARM Linux are **not supported**
- **NVMe SSD strongly recommended**
- NVIDIA GPU required for `--gpu` mode (GPU LRU expert cache coming next)

---

## More data

| Metric | Standard | With moe-l2 |
|--------|----------|-------------|
| Prompt processing | 110 t/s | 110 t/s |
| Generation speed | 65 t/s | 8.6 t/s |
| VRAM used | 23.3 GB | 2.2 GB |
| Model size / VRAM ratio | 0.26× | **2.7×** |

The speed tradeoff is intentional: experts load from system RAM via PCIe. Perfect for users who prioritize memory efficiency over peak throughput — home labs, edge deployments, budget hardware.

---

## Related work

[TencentYoutuResearch/Palm-Infra](https://github.com/TencentYoutuResearch/Palm-Infra) / **mollm** is a C++ engine from Tencent for MoE models with SSD expert offload on Apple Silicon / ARM Linux (16.22 t/s, 122B MoE, 16 GB peak RSS).

| Dimension | mollm (Tencent) | moe-l2 |
|-----------|-----------------|--------|
| Platform | Apple Silicon / ARM Linux | **Linux x86_64 + GPU (NVIDIA)** |
| Install | Build from source (CMake + C++) | **pip install moe-l2** |
| Model support | Qwen-series only | **Any llama.cpp MoE** (DeepSeek, Qwen, Mixtral...) |
| Backend | Custom C++ engine | **llama.cpp proxy** — zero migration |
| GPU acceleration | CPU only (NEON) | **CUDA + GPU VRAM** |
| Target user | Mobile / edge developers | **Desktop homelab users** |

---

## Project status

- ✅ Domain predictor (keyword + optional semantic)
- ✅ L2 cache (mmap LRU, thread-safe, async preload)
- ✅ Transparent proxy (HTTP/SSE forwarding)
- ✅ CLI with auto model detection and GPU mode
- ✅ GPU mode verified on RTX 4090 (DS-V2-Lite, ~1.6 GiB VRAM, 95% savings)
- ✅ PyPI package (`moe-l2`)
- 🔲 GPU LRU expert cache (keep hot experts in VRAM, 8.6 → 40+ t/s)

---

## License

**Apache 2.0.** See [LICENSE](LICENSE) for details.
