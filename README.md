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

### Benchmarked on RTX 4090 (2026-08-02, host-buffer GPU fast path)

| Mode | GPU VRAM | Gen speed | What it means |
|------|----------|-----------|---------------|
| Standard (all experts on GPU) | 23.3 GB | 65 t/s | Needs a 24 GB card |
| **moe-l2** (host-buffer experts, GPU compute) | **1.6 GB** | **DS 37.5 t/s · Qwen 46.8 t/s** | **Fits in 4-8 GB cards** |
| **Savings** | **93% less** | ~40% of full-GPU speed | Experts stay in CPU RAM, GPU reads them on demand |

> We benchmarked **Qwen3.6-A3B** (32B MoE) and **DeepSeek-V2-Lite** (16B MoE, 64 experts) on RTX 4090 with the host-buffer build: experts live in CPU pinned memory (zero VRAM), the scheduler copies only the **activated** experts to GPU each step. DS-V2-Lite **12.5 → 37.5 t/s** (+200%), Qwen3.6-A3B **10 → 46.8 t/s** (+370%), VRAM unchanged at 1.6 / 2.1 GB. Full report: `历史记录文档/lru-cache-档位对比测试-20260801.md`

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
| `moe-l2 collect --model <path>` | Collect MoE routing data → `~/.moe-l2/maps/domain_expert_map.json` |
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
| Prompt processing | 110 t/s | ~100 t/s |
| Generation speed (DS-V2-Lite) | 65 t/s | 37.5 t/s |
| Generation speed (Qwen3.6-A3B) | — | 46.8 t/s |
| VRAM used (DS-V2-Lite) | 23.3 GB | **1.6 GB** |
| Model size / VRAM ratio | 0.26× | **3.9×** |

The speed tradeoff is intentional and small: experts live in CPU pinned memory (host buffer, zero VRAM), and the scheduler copies only activated experts to GPU per step. On the 2026-08-02 host-buffer build, DS-V2-Lite reaches 37.5 t/s gen at 1.6 GB VRAM — ~40% of full-GPU speed at <7% of the VRAM.

## Expert Cache & host-buffer fast path (llama.cpp)

Beyond the proxy layer, moe-l2 ships llama.cpp patches that compile expert handling directly into the CUDA backend — no proxy needed. Two mechanisms:

**1. Host-buffer expert GPU fast path (recommended, 2026-08-02).** Expert tensors are loaded into a **CUDA host buffer** (CPU pinned memory, zero VRAM) instead of a plain CPU buffer. The scheduler then uses its MoE expert-copy optimization — it copies **only the activated experts** to GPU per step instead of the whole expert tensor — and the GPU runs the expert MUL_MAT_ID on the fast path. This is what the benchmark above measures (DS 37.5 / Qwen 46.8 t/s at 1.6 / 2.1 GB VRAM).

**2. A3 LRU expert cache (historical, `--expert-cache`).** An LRU cache that keeps recent experts on GPU. Verified on RTX 4090 with DS-V2-Lite: VRAM 6.6 GB → 1.2 GB (5.64×) at 8.2 t/s. This path is only useful when experts are CPU-hosted; it is now superseded by the host-buffer fast path, which is faster at the same VRAM.

### When the cache helps (and when it doesn't)

The expert cache only pays off when experts actually live in **CPU RAM** (default `mmap` mode). Verified on RTX 4090 with Mixtral 8x7B:

| Mode | Expert location | Gen speed | Cache value |
|------|----------------|-----------|-------------|
| `--no-mmap` (all weights on GPU) | GPU VRAM | **3.7 t/s** | ❌ cache is a no-op layer |
| `--no-mmap` + cache | GPU VRAM | 3.4-3.5 t/s | ❌ slower (cache forces the generic expert path, ~3 ms/token fixed overhead regardless of hit rate) |
| default `mmap` + cache | CPU RAM | ~1 t/s (scheduler copy bound) | ⚠️ not recommended |
| default `mmap` (no cache) | CPU RAM | 0.9 t/s | baseline |

Key findings (2026-08-01, per-segment CUDA timing):

- With `--no-mmap`, experts are **already fully resident in GPU VRAM** (copies are device-to-device, 100% on-GPU) — the cache adds nothing but an extra layer.
- Turning the cache on forces the generic `MUL_MAT_ID` pipeline (host-side id sorting + two stream syncs ≈ 3.1 ms/token) **independent of hit rate** — hit rate 27% vs 49% both showed identical ~3.1 ms. The fast expert path only runs with the cache **off**.
- Conclusion: use the cache only when experts are CPU-hosted (mmap). The bundled CLI defaults to this configuration (`mmap` default, no `--no-mmap` flag).

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
- ✅ CLI with auto model detection, GPU mode, and `collect` (routing data → expert map)
- ✅ Host-buffer expert GPU fast path (2026-08-02): DS-V2-Lite 12.5 → 37.5 t/s, Qwen3.6-A3B 10 → 46.8 t/s at 1.6 / 2.1 GB VRAM — experts in CPU pinned memory, only activated experts copied to GPU
- ✅ Expert cache boundary verified on Mixtral 8x7B / RTX 4090: under `--no-mmap` experts are already fully resident in VRAM, so the cache is a no-op layer that adds ~3 ms/token overhead (3.7 → 3.4 t/s); it only pays off when experts are CPU-hosted (mmap). CLI defaults to the mmap configuration.
- ✅ PyPI package (`moe-l2`)

---

## License

**Apache 2.0.** See [LICENSE](LICENSE) for details.
