# moe-l2

**Run large MoE models on consumer GPUs.** A transparent expert-caching proxy that lets you run 16GB+ models on 8GB (or even 4GB) GPUs — saving up to 91% VRAM.

## Why

Mixture-of-Experts (MoE) models have dozens to hundreds of "experts" but only activate a handful per token. Yet the standard inference stack loads *all* expert weights into GPU VRAM, wasting 80-95% of available memory on idle weights.

moe-l2 predicts which domain your prompt belongs to (code, math, Chinese tech, etc.), preloads the relevant experts into a fast mmap'd LRU cache, and keeps the rest on CPU/system memory. The GPU only holds the active expert set.

## Benchmark

Tested on **DeepSeek-V2-Lite** (16B params, 64 experts, top-6) in Q2_K quantization:

| Mode | GPU VRAM | Generation speed |
|------|----------|-----------------|
| Standard (all experts on GPU) | 23.3 GB | 65 t/s |
| **moe-l2** (hot-cached experts) | **2.2 GB** | **8.6 t/s** |
| **Savings** | **91% less VRAM** | 13% of original speed |

With a 24 GB GPU you'd normally need for this model, moe-l2 brings it down to **2.2 GB** — leaving 22 GB free for other workloads, or making the model runnable on an 8 GB / 4 GB card.

## Quick start

```bash
pip install moe-l2
moe-l2 start --model model.gguf --l2-size 4GB
```

The proxy starts on `localhost:11435` — all your existing tools (curl, Open WebUI, LangChain) work without changes.

## How it works

```
your client → moe-l2 proxy (:11435) → ollama/llama.cpp (:11434)
                  ├── Domain predictor (keyword + optional semantic)
                  ├── L2 cache (LRU, mmap /dev/shm/, async preload)
                  └── Transparent forwarding (SSE streaming)
```

1. User sends a prompt
2. Domain predictor classifies it (codegen, math, chinese_tech, ...)
3. L2 cache preloads the predicted experts into shared memory
4. Request is forwarded to the backend — experts already hot in cache
5. Cache hit rate typically exceeds 85% within the same session

## Features

- **Zero-code setup** — install, point to a model, done
- **Transparent proxy** — no client changes, works with any OpenAI-compatible tool
- **Dual predictor** — keyword-based (zero extra deps) or hybrid keyword + semantic embedding
- **LRU eviction** — configurable cache size (`--l2-size 512MB` to `16GB`)
- **GPU mode** — A3-patched llama-server with expert-offloading, verified on RTX 4090
- **Library API** — `from moe_l2 import predict, L2Cache` for embedded use

## More data

| Metric | Standard | With moe-l2 |
|--------|----------|-------------|
| Prompt processing | 110 t/s | 110 t/s |
| Generation speed | 65 t/s | 8.6 t/s |
| VRAM used | 23.3 GB | 2.2 GB |
| Model size / VRAM ratio | 0.26× | **2.7×** |

The speed tradeoff is predictable: experts load from system RAM via PCIe. This is an intentional design choice for users who prioritize memory efficiency over peak throughput — ideal for home labs, edge deployments, and budget hardware.

## Project status

- ✅ Domain predictor (keyword + optional semantic)
- ✅ L2 cache (mmap LRU, thread-safe, async preload)
- ✅ Transparent proxy (HTTP/SSE forwarding)
- ✅ CLI (start/stats with auto model detection, --gpu mode)
- ✅ GPU mode verified on RTX 4090 (DS-V2-Lite, ~1.6 GiB VRAM)
- ✅ PyPI package (moe-l2)
- 🔲 GPU LRU expert cache (keep hot experts in VRAM, reduce PCIe transfers) — next

## License

**All Rights Reserved.** This software is proprietary and confidential.
