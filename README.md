# moe-l2

[English](README.md) | [**中文**](README_zh.md)

**MoE expert offload for low-VRAM GPUs — run 100B+ MoE models (DeepSeek, Qwen, Mixtral) on 8 GB cards.** A transparent, OpenAI-compatible proxy that predicts which experts your prompt needs and preloads them into a shared-memory LRU cache, so you can run 16 GB+ MoE models on 8 GB GPUs with up to 91% VRAM savings.

> ⭐ **Found this useful? Give us a star** — it helps others discover the project. [★ Star on GitHub](https://github.com/yalun753/moe-l2)

### Real-world benchmark

| Your GPU | Normally fits | **With moe-l2** | **Measured speed** (RTX 4090) |
|----------|--------------|-----------------|-------------------------------|
| 4 GB | — | DeepSeek-V2-Lite (16B MoE) ✅ | **37.9 t/s** |
| **8 GB** | 7B dense | **Qwen3.6-A3B (32B MoE) ✅** | **50.2 t/s** |
| 10-11 GB | — | **DeepSeek-V4-Flash (157B MoE, 85 GB file) ✅** | **10.1 t/s** |

> Speed = RTX 4090 measured (2026-08-07, on-demand pin + A3 cache 2048, multi-arch build); 2080 Ti: Qwen 24.5 t/s, DS 6.89 t/s, V4 0.89-1.07 t/s. See [models-benchmark.md](references/models-benchmark.md).

Without moe-l2, an 8 GB card **cannot load these models at all** — it OOMs immediately. With moe-l2, a 32B MoE fits in ~2.9 GB VRAM (on-demand pin experts on Qwen3.6-A3B, GPU compute). **DeepSeek-V4-Flash (157B params / 85 GB file, 256 experts, top-6) runs on a 10-11 GB card at 8.3-9.1 GB VRAM** — verified on 2080 Ti (0.89-1.07 t/s) and RTX 3080 (2.11-2.22 t/s), with expert-page eviction keeping RSS capped. Full report: [deepseek-v4-flash-verify-20260805.md](references/deepseek-v4-flash-verify-20260805.md) · **All measured models: [models-benchmark.md](references/models-benchmark.md)**

### Benchmarked on RTX 4090 (2026-08-07, on-demand pin main path)

| Mode | GPU VRAM | Gen speed | What it means |
|------|----------|-----------|---------------|
| Standard (all experts on GPU) | 23.3 GB | 65 t/s | Needs a 24 GB card |
| **moe-l2** (on-demand pin experts, GPU compute) | **1.6-2.9 GB** | **DS 37.9 t/s · Qwen 50.2 t/s** | **Fits in 4-8 GB cards** |
| **Savings** | **93% less** | ~58% of full-GPU speed | Experts stay in CPU RAM, GPU reads them on demand |

> We benchmarked **Qwen3.6-A3B** (32B MoE) and **DeepSeek-V2-Lite** (16B MoE, 64 experts) on RTX 4090. **2026-08-07 main path upgraded to on-demand pin** (lazy mmap load + first-touch merge-registration of the whole expert tensor + A3 cache 2048 slots): experts stay in CPU RAM (zero VRAM), the scheduler copies only the **activated** experts to GPU each step, hot experts are cached in VRAM. DS-V2-Lite **12.5 → 37.9 t/s** (+200%), Qwen3.6-A3B **10 → 50.2 t/s** (+400%, beats pre-lazy 46.5). Full reports: [qwen3.6-a3b-iq2m-benchmark.md](references/qwen3.6-a3b-iq2m-benchmark.md) · [deepseek-v2-lite-q2k-benchmark.md](references/deepseek-v2-lite-q2k-benchmark.md) · [cache-sched-layer-benchmark.md](references/cache-sched-layer-benchmark.md) · [models-benchmark.md](references/models-benchmark.md) · **Why host-buffer? Full approach history: [design-decisions_EN.md](references/design-decisions_EN.md) / [design-decisions.md (中文)](references/design-decisions.md)**

### Multi-architecture binaries (bins-v0.3.1, 2026-08-05)

One binary for **all NVIDIA consumer GPUs** — GTX 1080 (sm_61) through RTX 50-series (sm_120a). Built with CUDA 12.8; no per-GPU compilation needed. `moe-l2 download-bins` fetches it automatically. bins-v0.3.1 includes the **on-demand pin main path** + expert-page eviction v3.1 (`MOE_L2_LRU_MAX_EXPERTS=N`) + A3 cache 2048 slots + cuda-libs (no libnccl — not needed for single-GPU).

| GPU | Architecture | DS-V2-Lite gen | Qwen3.6-A3B gen | VRAM |
|-----|-------------|----------------|-----------------|------|
| RTX 2080 Ti | sm_75 (Turing) | 6.89 t/s | 11.15 t/s | ~1.0-2.4 GB |
| RTX 3080 Ti | sm_86 (Ampere) | 12.25 t/s | 13.28 t/s | ~1.1-2.2 GB |
| RTX 5090 | sm_120a (Blackwell) | 16.63 t/s | 9.71 t/s | ~1.3-2.5 GB |
| RTX 4090* | sm_89 (Ada) | 39.0 t/s | 51.5 t/s | 1.6-2.9 GB |

\* 4090 measured with the multi-arch package (2026-08-07, on-demand pin + cache 2048, CUDA 12.8); 2080 Ti / 3080 Ti / 5090 rows are v3.1 multi-arch (bins-v0.3.0). Qwen single-turn 24.5 t/s on 2080 Ti (bins-v0.3.1, 2x vs old host-buffer 11.15).

> Verified on 2080 Ti (SM75), 3080 Ti (SM86) and 5090 (SM120a) with the multi-arch build. The 3080 Ti run was **+55% faster** than the previous CUDA 11.8 single-arch build (12.25 vs 7.88 t/s). Note: SM120a (RTX 50) kernel efficiency in llama.cpp 76f46ad is not yet mature — RTX 5090 shows only +36% over 3080 Ti on DS and −27% on Qwen; a newer llama.cpp rebuild should improve 50-series speed. Full report: [multi-arch-three-gpu-benchmark.md](references/multi-arch-three-gpu-benchmark.md) · **DeepSeek V4 Flash (157B) dual-GPU run: [deepseek-v4-flash-verify-20260805.md](references/deepseek-v4-flash-verify-20260805.md)**

## Quick start

**One-line install (Linux x86_64 + NVIDIA GPU):**

```bash
curl -fsSL https://raw.githubusercontent.com/yalun753/moe-l2/main/scripts/install.sh | bash
```

The installer checks your GPU/driver/Python, installs moe-l2 from PyPI,
downloads the pre-built CUDA binaries, optionally downloads a demo model
(Qwen3.6-35B-A3B, ~11.5 GB, resumable), then runs a self-check.

**Manual install:**

```bash
pip install moe-l2                   # keyword-only predictor (zero extra deps)
pip install moe-l2[predictor]        # hybrid: keyword + semantic embedding
moe-l2 download-bins                 # pre-built CUDA llama-server (on-demand pin patched)
moe-l2 model download --model qwen3.6-35b   # optional demo model (~11.5 GB)
moe-l2 start --model model.gguf --gpu
```

Useful commands:

```bash
moe-l2 doctor                        # environment self-check (GPU/CUDA/Python/disk)
moe-l2 model list                    # list downloadable models
moe-l2 model download --model <name> # download model (resumable, via hf-mirror)
```

Your tools (curl, Open WebUI, LangChain) connect to `localhost:11435` — no client changes needed.

## How it works

MoE models have many "experts" but only activate a few per token. moe-l2 predicts your prompt's domain (codegen, math, chinese_tech, etc.) and preloads the relevant experts into an mmap'd LRU cache before they're needed.

```
user → moe-l2 proxy (localhost:11435)
    ├── predict domain (keyword → TF-IDF → semantic)
    ├── on-demand pin experts (lazy mmap + register, zero VRAM)
    ├── hot experts cached in VRAM (A3 LRU)
    └── forward to llama-server (localhost:11436, CUDA GPU)
        └── GPU reads pinned experts via PCIe DMA; cold pages evicted
```

### Visual demo (RTX 4090, 2026-08-02)

| Qwen3.6-35B-A3B (32B MoE) — standard vs moe-l2 | DeepSeek-V2-Lite (16B MoE) — 8 GB card vs 24 GB card |
|---|---|
| ![Qwen VRAM comparison](examples/demo-assets/fig1-qwen-vram.png) | ![DS VRAM comparison](examples/demo-assets/fig2-ds-vram.png) |

Summary: **93% less VRAM · 58% of full-GPU speed · 3.1× model-per-GB ratio** — an 8 GB card runs what used to need 24 GB (measured 2026-08-02 host-buffer build; 2026-08-07 on-demand pin: DS 37.9 t/s @ 2.0 GB / Qwen 50.2 t/s @ 2.9 GB):

![moe-l2 summary](examples/demo-assets/fig3-summary.png)

Live capture: Qwen3.6-35B-A3B generating **3,200 tokens with VRAM pinned at ~2.4 GB** (41.6 t/s) — watch the VRAM curve stay flat below the 8 GB line the whole run:

[`examples/demo-assets/demo-vram-animation.mp4`](examples/demo-assets/demo-vram-animation.mp4) (45 s, 1280×720) · raw telemetry: [`examples/demo-assets/rec_data.csv`](examples/demo-assets/rec_data.csv) · full generated text: [`examples/demo-assets/rec_full.txt`](examples/demo-assets/rec_full.txt)

## Usage

### 1. L2 proxy with GPU on-demand pin (recommended)

Start the transparent proxy with the bundled on-demand pin llama-server:

```bash
moe-l2 start --model /models/DeepSeek-V2-Lite.Q4_K_M.gguf --gpu
```

The proxy exposes OpenAI-compatible endpoints — all your tools work through it (curl, open-webui, langchain):

```bash
# streaming
curl http://localhost:11435/v1/chat/completions -d '{
  "model":"qwen3:4b",
  "messages":[{"role":"user","content":"write a Python script"}],
  "stream":true
}'

# blocking
curl http://localhost:11435/v1/chat/completions -d '{
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
│         llama-server (port 11436, CUDA GPU)                │
│         on-demand pin experts: lazy mmap, zero VRAM       │
│         GPU reads pinned experts via PCIe DMA             │
│         hot experts cached in VRAM (A3 LRU, 2048 slots)   │
│         cold expert pages evicted (v3.1, RSS capped)      │
└────────────────────────────────────────────────────────────┘
```

## CLI reference

| Command | Description |
|---------|-------------|
| `moe-l2 start --model <path> --gpu` | Start proxy + on-demand pin llama-server (recommended) |
| `moe-l2 start --model <path> --l2-size <size>` | Start proxy + cache only (no GPU) |
| `moe-l2 stats --port <port>` | Show live cache stats |
| `moe-l2 download-bins [--release TAG]` | Download pre-built GPU binaries from GitHub |
| `moe-l2 collect --model <path>` | Collect MoE routing data → `~/.moe-l2/maps/domain_expert_map.json` |
| `moe-l2 stop --port <port>` | Stop proxy |

Options:
- `--model auto`: scan `/opt/data/models/*.gguf`
- `--l2-size 4GB` / `--l2-size 512MB`: target cache size (proxy-only mode)
- `--port 11435` (default)
- `--gpu`: enable GPU mode (requires CUDA + NVIDIA GPU; spawns bundled on-demand pin llama-server on 11436)

> **GPU binaries**: Not tracked in git (bundled as `llama_bins.tar.gz`, ~1.9 GB multi-architecture on the `bins-v0.3.1` release — sm_61/75/86/89/120a, one binary for all NVIDIA consumer GPUs, ships cuda-libs). Fetched at runtime via `moe-l2 download-bins`. When you `pip install moe-l2`, binaries are included. For git-clone users, run `moe-l2 download-bins` to fetch them from GitHub Release.

## Platform requirements

- **Linux x86_64 only** — pre-built binaries target Linux AMD64 (CUDA `.so` + `llama-server`)
- macOS, Windows, and ARM Linux are **not supported**
- **NVMe SSD strongly recommended**
- NVIDIA GPU required for `--gpu` mode

## More data

| Metric | Standard | With moe-l2 |
|--------|----------|-------------|
| Prompt processing (DS-V2-Lite) | 110 t/s | 99 t/s · **308 t/s** (sched-cache=0.25) |
| Generation speed (DS-V2-Lite) | 65 t/s | 37.9 t/s · 39.2 t/s (sched-cache=0.25) |
| Generation speed (Qwen3.6-A3B) | — | 50.2 t/s |
| VRAM used (DS-V2-Lite) | 23.3 GB | **2.0 GB** |
| Model size / VRAM ratio | 0.26× | **3.1×** |

The speed tradeoff is intentional and small: expert weights live in CPU RAM (lazy mmap, zero VRAM) and are pinned on first touch — the GPU reads them directly via PCIe DMA, hot experts are cached in VRAM (A3 LRU), and cold pages are evicted to keep RSS capped. On the 2026-08-07 on-demand pin build, DS-V2-Lite reaches 37.9 t/s gen at 2.0 GB VRAM — ~58% of full-GPU speed at <9% of the VRAM.

## Expert offload & cache fast path (llama.cpp)

Beyond the proxy layer, moe-l2 ships llama.cpp patches that compile expert handling directly into the CUDA backend — no proxy needed. Two mechanisms:

**1. On-demand pin expert GPU fast path (recommended, 2026-08-07).** Expert tensors live in CPU RAM via lazy mmap (zero VRAM). On first touch during inference the whole expert tensor is merge-registered as pinned (`cudaHostRegister`), so the GPU reads it directly via PCIe DMA with no per-step copies. Hot experts are cached in VRAM (A3 LRU, 2048 slots) and cold pages are evicted (v3.1) to keep RSS capped. This is what the benchmark above measures (DS 37.9 / Qwen 50.2 t/s at 2.0 / 2.9 GB VRAM).

**2. A3 LRU expert cache (historical, `--expert-cache`).** An LRU cache that keeps recent experts on GPU. In the old `--cpu-moe` CPU-compute architecture it cut VRAM from 6.6 GB → 1.2 GB (5.64×) at 8.2 t/s. In the current on-demand pin architecture the cache is hooked into the scheduler copy layer (`GGML_CUDA_EXPERT_CACHE`) and only pays off for small, frequently-hit experts (see below).

### When the cache helps (and when it doesn't)

The sched-cache only pays off when experts are **small and frequently hit**. Verified on RTX 4090 (host-buffer, 2026-08-02):

| Model | Expert size | Top-k | Cache value |
|-------|------------|-------|-------------|
| DS-V2-Lite | 1.55 MB | top-6 | ✅ **Prompt +211%, Gen +5%** (cache=0.25) |
| Qwen3.6-A3B | ~1 MB | top-8 | ❌ no gain (experts too small, copy cost already trivial) |
| Mixtral-8x7B | 252 MB | top-2 | ❌ no gain, +660 MiB VRAM (top-2 hit rate too low) |

Key findings (2026-08-02, cache hooked into the scheduler input-copy layer):

- The cache sits in `copy_experts`: on hit it does a D2D copy (no PCIe round-trip), on miss it falls back to the pinned-host CPU→GPU path and writes back. It only intercepts single-expert groups.
- Benefit = **expert size × hit rate**. DS (1.55 MB, top-6) wins big; Qwen (~1 MB) pays for itself at best; Mixtral (252 MB, top-2) never hits enough to pay for its VRAM slots.
- Recommended: `GGML_CUDA_EXPERT_CACHE=0.25` for DS-class models (16 slots/layer cover all hot experts, VRAM unchanged). Leave it off for Qwen/Mixtral.

> Run the demo yourself: `bash examples/demo_a3_compression.sh` (edit paths first).

## Related work

### AirLLM (lyogavin/airllm, ~29k stars)

AirLLM is a general-purpose layer-offload scheme for very large models. Its scheduling granularity is the **full Transformer layer**: during inference only one layer's weights stay in VRAM, everything else is swapped to/from disk — giving an extreme low-VRAM floor (4GB GPU runs 70B). But it has three weaknesses: ① every generated token requires reading/writing a full layer to/from disk, so IO cost is huge and interactive speed is very low; ② no MoE-specific routing prediction or expert hot cache (per-expert streaming only started in 2026-07, with Kimi K3), so repeated prompts keep triggering heavy disk reads; ③ built on native Hugging Face Transformers, with no OpenAI-compatible serving interface out of the box, making it awkward to wire into Open WebUI, LangChain, etc.

| Dimension | AirLLM | moe-l2 |
|-----------|--------|--------|
| Scheduling unit | Full Transformer layer | **Per-expert (sparse-optimal)** |
| Target models | All models (dense + MoE) | **MoE-optimized (DeepSeek / Qwen / Mixtral)** |
| Weight format | Native Hugging Face weights | **GGUF (llama.cpp ecosystem)** |
| Platform | Windows / macOS / Linux, incl. CPU | Linux x86_64 + NVIDIA GPU |
| MoE memory | Whole-layer disk swap, no hot cache | **85GB V4: 8.3GB VRAM + 11-12GB RSS cap (measured)** |
| MoE speed | Per-layer disk thrash, batch-offline only | Hot-expert cache cuts disk IO, real-time chat (Qwen full-chain 9.3 t/s measured) |
| Serving API | Python-code only, no web service | **Built-in OpenAI-compatible proxy (:11435), drop-in** |
| GPU support | Native transformers, manual CUDA setup | **Multi-arch kernels via download-bins, GTX10xx–RTX50xx** |
| Multi-shard GGUF | No specific support | **Fixed multi-shard metadata parsing, 85GB 3-shard V4 stable** |

**Which to choose**: pick moe-l2 if you run MoE models (DeepSeek/Qwen) locally for chat, have an 8–12GB older NVIDIA card, want an OpenAI API for tooling, or use multi-shard giant GGUFs. Pick AirLLM if you need dense (non-MoE) models, use Windows/macOS/AMD or CPU-only environments (moe-l2 currently requires Linux + NVIDIA), only do one-shot batch generation, or must stay with native HF weights.

## Project status

- ✅ Domain predictor (keyword + optional semantic)
- ✅ L2 cache (mmap LRU, thread-safe, async preload)
- ✅ Transparent proxy (HTTP/SSE forwarding)
- ✅ CLI with auto model detection, GPU mode, and `collect` (routing data → expert map)
- ✅ Host-buffer expert GPU fast path (2026-08-02): DS-V2-Lite 12.5 → 37.5 t/s, Qwen3.6-A3B 10 → 46.8 t/s at 1.6 / 2.1 GB VRAM — experts in CPU pinned memory, only activated experts copied to GPU
- ✅ **On-demand pin main path (2026-08-07)**: lazy mmap load + first-touch merge-registration of the whole expert tensor + A3 cache 2048 slots → Qwen **50.2** / DS **37.9** / V4 **10.1** t/s on 4090 (V4 5× faster than 1.7-2.0); fixes CUDA 11.8 cross-register-range copy crash
- ✅ Expert cache boundary verified on Mixtral 8x7B / RTX 4090 (2026-08-02, sched-cache): cache benefit = expert size × hit rate — DS-V2-Lite (1.55 MB, top-6) gets Prompt +211% / Gen +5% at cache=0.25; Qwen (~1 MB) and Mixtral (252 MB, top-2) get no gain. Recommended: cache=0.25 for DS-class, off otherwise.
- ✅ **DeepSeek-V4-Flash (157B MoE) verified (2026-08-05)**: 85 GB 3-shard GGUF runs on 2080 Ti (11 GB) and RTX 3080 (10 GB) — VRAM 8.3-9.1 GB, RSS capped by expert-page eviction v3.1 (fixed-expert-count LRU, `MOE_L2_LRU_MAX_EXPERTS`), multi-shard GGUF parsing fix shipped. Speed 0.89-2.22 t/s (GPU compute bound). [Full report](references/deepseek-v4-flash-verify-20260805.md)
- ✅ PyPI package (`moe-l2`)

---

## License

**Apache 2.0.** See [LICENSE](LICENSE) for details.
