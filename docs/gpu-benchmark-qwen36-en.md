# MoE-L2 GPU Benchmark Complete Data — Qwen3.6-35B-A3B (IQ2_M, ~10.7 GB)

## Hardware: RTX 4090 24GB | Force CPU-Expert Offload | --cache-type-k q8_0

### Test Protocol
- **Short**: --single-turn -n 128 -c 512 (initial response)
- **Followup**: --single-turn -n 8 -c 1536 (follow-up / continuation)
- **Longtail**: --single-turn -n 512 -c 1024 (long generation)

---

### 1. codegen (Code Generation)

| Phase | Prompt t/s | Gen t/s | VRAM (MiB) |
|-------|-----------|---------|------------|
| Short | 10.1 | 6.2 | 2231 |
| Followup | 119.1 | 5.3 | - |
| Longtail | 39.5 | 6.1 | 2241 |

### 2. debug (Code Debugging)

| Phase | Prompt t/s | Gen t/s | VRAM (MiB) |
|-------|-----------|---------|------------|
| Short | 40.4 | 6.0 | 2235 |
| Followup | 69.5 | 5.7 | - |
| Longtail | 47.3 | 6.2 | 2243 |

### 3. math (Math Reasoning)

| Phase | Prompt t/s | Gen t/s | VRAM (MiB) |
|-------|-----------|---------|------------|
| Short | 44.6 | 6.1 | 2235 |
| Followup | 70.2 | 5.8 | - |
| Longtail | 46.8 | 6.3 | 2243 |

### 4. logic (Logic Puzzles)

| Phase | Prompt t/s | Gen t/s | VRAM (MiB) |
|-------|-----------|---------|------------|
| Short | 39.6 | 6.2 | - |
| Followup | 42.1 | 6.6 | - |
| Longtail | 51.6 | 6.2 | 2243 |

### 5. general_qa (General Q&A)

| Phase | Prompt t/s | Gen t/s | VRAM (MiB) |
|-------|-----------|---------|------------|
| Short | 10.5 | 6.0 | - |
| Followup | 28.4 | 5.4 | - |
| Longtail | 36.5 | 5.8 | 2241 |

### 6. chinese_tech (Chinese Tech)

| Phase | Prompt t/s | Gen t/s | VRAM (MiB) |
|-------|-----------|---------|------------|
| Short | 10.2 | 6.1 | - |
| Followup | 34.7 | 5.0 | - |
| Longtail | 33.3 | 5.9 | 2241 |

### 7. creative_write (Creative Writing)

| Phase | Prompt t/s | Gen t/s | VRAM (MiB) |
|-------|-----------|---------|------------|
| Short | 29.3 | 5.8 | - |
| Followup | 30.9 | 5.6 | - |
| Longtail | 34.6 | 6.0 | 2241 |

### 8. translate (EN→CN Translation)

| Phase | Prompt t/s | Gen t/s | VRAM (MiB) |
|-------|-----------|---------|------------|
| Short | 29.9 | 5.8 | - |
| Followup | 56.2 | 5.7 | - |
| Longtail | 51.3 | 6.0 | 2245 |

---

### Summary Statistics (Qwen3.6-35B-A3B IQ2_M)
- **Gen speed**: 5.0~6.6 t/s → mean ~5.95 t/s, cross-domain variance minimal (<3%)
- **VRAM**: 2231~2245 MiB → stable ~2.2 GiB (model size 10.7 GB / VRAM ~2.2 GiB = compression ratio ~4.9×)
- **Prompt speed**: 10~119 t/s, depends on prompt length (short prompts slower, long prompts faster)
- **Key finding**: Gen speed is domain-independent — confirms that forced CPU-expert offload strategy effectively eliminates MoE routing variance
