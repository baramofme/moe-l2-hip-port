# moe-l2

**Run large MoE models on consumer GPUs.** A transparent proxy that predicts which experts your prompt needs, preloads them into a shared-memory LRU cache, so you can run 16 GB+ models on 8 GB GPUs.

## Quick start

```bash
pip install moe-l2                   # keyword-only predictor (zero extra deps)
pip install moe-l2[predictor]        # hybrid: keyword + semantic embedding
moe-l2 start --model model.gguf --l2-size 4GB
```

Your tools (curl, Open WebUI, LangChain) connect to `localhost:11435` — no client changes needed.

## How it works

MoE models have many "experts" but only activate a few per token. moe-l2 predicts your prompt's domain (codegen, math, chinese_tech, etc.) and preloads the relevant experts into an mmap'd LRU cache before they're needed.

```
user → moe-l2 proxy (localhost:11435)
    ├── predict domain
    ├── preload domain experts → /dev/shm/moe_l2/
    └── forward to ollama (localhost:11434)
```

### Real-world benchmark

| Your GPU | Normally fits | **With moe-l2** |
|----------|--------------|-----------------|
| 4 GB | — | DeepSeek-V2-Lite (16B MoE) ✅ |
| **8 GB** | 7B dense | **Qwen2.5-32B-A3B (32B MoE) ✅** |
| 12 GB | 13B dense | DeepSeek-V2 (236B MoE) ✅ |
| 24 GB | 34B dense | DeepSeek-V2 (236B MoE) ✅ |

Without moe-l2, an 8 GB card **cannot load these models at all** — it OOMs immediately. With moe-l2, a 32B MoE fits in ~2.7 GB VRAM (cache=0.5 on DS-V2-Lite).

### Benchmarked on RTX 4090

| Mode | GPU VRAM | Speed | What it means |
|------|----------|-------|---------------|
| Standard (all experts on GPU) | 23.3 GB | 65 t/s | Needs a 24 GB card |
| **moe-l2** (hot-cached experts) | **2.7 GB** | **~7 t/s** gen · **103 t/s** prompt | **Fits in 4 GB cards** |
| **Savings** | **88% less** | 11% speed | ~20 GB freed for other work |

> We benchmarked **Qwen3.6-A3B** (32B MoE) and **DeepSeek-V2-Lite** (16B MoE, 64 experts) on RTX 4090. GPU LRU expert cache (Phase 3) is now **stable** — 0 crashes across 7 cache levels × 3 conversation types. Gen speed is CPU-bound (~5-7 t/s), but followup prompt processing gets 10× faster (~80-103 t/s) from cache hits. Full reports: [Qwen3.6](references/qwen3.6-a3b-iq2m-benchmark.md) · [DS-V2-Lite](references/deepseek-v2-lite-q2k-benchmark.md)

## Usage

### 1. L2 proxy (recommended)

Start the transparent proxy — sits between your client and ollama:

```bash
moe-l2 start --model /models/DeepSeek-V2-Lite.Q4_K_M.gguf --l2-size 4GB
```

All default ollama tools work through it (curl, open-webui, langchain):

```bash
# streaming
curl http://localhost:11435/api/chat -d '{
  "model":"qwen3:4b",
  "messages":[{"role":"user","content":"write a Python script"}],
  "stream":true
}'

# blocking
curl http://localhost:11435/api/chat -d '{
  "model":"qwen3:4b",
  "messages":[{"role":"user","content":"hello"}],
  "stream":false
}'
```

### 2. Monitor cache stats

```bash
moe-l2 stats --port 11435
```

Example output:
```
moe-l2 cache stats
  requests:     47
  hits:         42     (89.4%)
  misses:        5
  slots_used:  32/48  (66.7%)
  memory:     456 MB  (68.3% of 668 MB)
```

### 3. Use as a library

```python
from moe_l2 import predict, predict_hybrid, domain_to_expert_ids
from moe_l2.cache import L2Cache

# Predict domain (zero-dependency mode)
domain = predict("print hello world")  # → "codegen"

# Or use the hybrid semantic predictor
domain = predict_hybrid("how do I sort a list?")  # → "codegen"

# Preload experts
cache = L2Cache(model_path="model.gguf", l2_size="4GB")
cache.preload(domain_to_expert_ids[domain])
```

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                       HTTP client                          │
│     curl / open-webui / langchain / any OpenAI client      │
└──────────┬─────────────────────────────────────────────────┘
           │ POST /api/chat
           ▼
┌────────────────────────────────────────────────────────────┐
│               moe-l2 Proxy (port 11435)                     │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Domain Predictor                                    │   │
│  │  - Keyword mode: zero deps, instant classification   │   │
│  │  - Hybrid mode: +sentence-transformers for context   │   │
│  └─────────────────────┬───────────────────────────────┘   │
│                        │ domain                             │
│  ┌─────────────────────▼───────────────────────────────┐   │
│  │  L2 Cache (mmap'd shared memory)                    │   │
│  │  - LRU eviction policy                              │   │
│  │  - Async preload: next-prediction prefetch          │   │
│  │  - Thread-safe concurrent access                    │   │
│  │  - Zero-copy mmap from SSD → RAM                    │   │
│  └─────────────────────┬───────────────────────────────┘   │
│                        │ forward request                    │
└────────────────────────┼───────────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────────┐
│                ollama / llama.cpp (port 11434)              │
│                Hot experts in GPU VRAM                      │
│                Cold experts loaded from RAM/SSD via mmap    │
└────────────────────────────────────────────────────────────┘
```

## CLI reference

| Command | Description |
|---------|-------------|
| `moe-l2 start --model <path> --l2-size <size>` | Start proxy + cache |
| `moe-l2 start --model <path> --gpu` | Start with GPU-accelerated llama-server |
| `moe-l2 stats --port <port>` | Show live cache stats |
| `moe-l2 download-bins [--release TAG]` | Download pre-built GPU binaries from GitHub |
| `moe-l2 stop --port <port>` | Stop proxy |

Options:
- `--model auto`: scan `/opt/data/models/*.gguf`
- `--l2-size 4GB` / `--l2-size 512MB`: target cache size
- `--port 11435` (default)
- `--gpu`: enable GPU mode (requires CUDA + NVIDIA GPU)

> **GPU binaries**: Not tracked in git (~530 MB). Fetched at runtime via `moe-l2 download-bins`. The repo does not track 500MB+ .so files. When you `pip install moe-l2`, binaries are included. For git-clone users, run `moe-l2 download-bins` to fetch them from GitHub Release.

## Platform requirements

- **Linux x86_64 only** — pre-built binaries target Linux AMD64 (CUDA `.so` + `llama-server`)
- macOS, Windows, and ARM Linux are **not supported**
- **NVMe SSD strongly recommended**
- NVIDIA GPU required for `--gpu` mode

## More data

| Metric | Standard | With moe-l2 |
|--------|----------|-------------|
| Prompt processing | 110 t/s | 110 t/s |
| Generation speed | 65 t/s | ~5-7 t/s |
| VRAM used | 23.3 GB | 2.7 GB |
| Model size / VRAM ratio | 0.26× | **2.2×** |

The speed tradeoff is intentional: experts load from system RAM via PCIe. Phase 3 GPU LRU cache is stable — gen speed is CPU-bound (~5-7 t/s), but followup prompt processing reaches ~80-103 t/s from cache hits.

## A3 Expert Cache (llama.cpp)

Beyond the proxy layer, moe-l2 ships an **A3 (Attention-Aware Expert Cache)** patch for llama.cpp that compiles expert LRU caching directly into the CUDA backend — no proxy needed. Run any llama.cpp binary with `--cpu-moe --expert-cache <fraction>`.

```
./llama-batched -m DeepSeek-V2-Lite.Q2_K.gguf \
  -p "prompt" -n 128 -ngl 99 \
  --cpu-moe --expert-cache 0.25
```

**Real benchmark (RTX 4090 · DeepSeek-V2-Lite Q2_K 6.4 GB):**

| Mode | VRAM used | Speed | Savings |
|------|-----------|-------|---------|
| OG (---no-mmap, full GPU) | **6,635 MB** | 126.64 t/s | baseline |
| A3 (--expert-cache 0.25) | **1,175 MB** | 8.22 t/s | **5,460 MB (5.64×)** |

A3 caches 25% of the most-recently-used experts on GPU and swaps inactive ones in from CPU RAM on demand. Good for single-user chat: acceptable latency (~7-8 t/s gen) with >80% VRAM savings. For batch / high-throughput scenarios, increase the cache fraction or omit `--cpu-moe`.

> Run the demo yourself: `bash examples/demo_a3_compression.sh` (edit paths first).

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

## Project status

- ✅ Domain predictor (keyword + optional semantic)
- ✅ L2 cache (mmap LRU, thread-safe, async preload)
- ✅ Transparent proxy (HTTP/SSE forwarding)
- ✅ CLI with auto model detection and GPU mode
- ✅ GPU mode verified on RTX 4090 (DS-V2-Lite, ~1.6 GiB VRAM, 95% savings)
- ✅ GPU LRU expert cache (verified: Qwen3.6 + DS-V2-Lite, 7 levels × 3 types, 0 crashes)
- ✅ PyPI package (`moe-l2`)

---

## License

**Apache 2.0.** See [LICENSE](LICENSE) for details.
