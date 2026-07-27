# MoE L2 GPU Benchmark — Phase 2: DS-V2-Lite (Full Report)

## Test Configuration

| Item | Value |
|------|-------|
| Model | DeepSeek-V2-Lite-Chat-Uncensored.Q2_K.gguf |
| Size | 6.0 GB |
| Binary | `/root/llama-cli-force-cpu-experts-20260901` (CPU expert offload version) |
| GPU | RTX 4090 24GB (AutoDL) |
| KV cache | q8_0 |
| Test date | 2026-07-27 |
| Tests completed | 8 domains × 3 phases = **24/24** ✅ |

---

## Full Test Data

### codegen (Code Generation)

| Phase | Prompt t/s | Gen t/s | VRAM (MiB) |
|-------|-----------|---------|------------|
| short (-n 128, -c 512) | 18.5 | 7.9 | — |
| followup (-n 8, -c 1536) | 45.4 | 9.5 | — |
| longtail (-n 512, -c 1024) | 82.3 | 8.8 | 1363 |

### debug (Code Debugging)

| Phase | Prompt t/s | Gen t/s | VRAM (MiB) |
|-------|-----------|---------|------------|
| short | 61.7 | 8.3 | — |
| followup | 47.6 | 9.5 | — |
| longtail | 83.8 | 8.4 | 1363 |

### math (Math Reasoning)

| Phase | Prompt t/s | Gen t/s | VRAM (MiB) |
|-------|-----------|---------|------------|
| short | 31.7 | 8.7 | — |
| followup | 35.9 | 8.3 | — |
| longtail | 66.2 | 8.6 | 1363 |

### logic (Logic Puzzles)

| Phase | Prompt t/s | Gen t/s | VRAM (MiB) |
|-------|-----------|---------|------------|
| short | 43.9 | 8.1 | — |
| followup | 40.6 | 9.3 | — |
| longtail | 69.1 | 8.2 | 1363 |

### general_qa (General Q&A)

| Phase | Prompt t/s | Gen t/s | VRAM (MiB) |
|-------|-----------|---------|------------|
| short | 18.5 | 8.2 | — |
| followup | 38.0 | 8.7 | — |
| longtail | 40.5 | 7.9 | 1359 |

### chinese_tech (Chinese Tech)

| Phase | Prompt t/s | Gen t/s | VRAM (MiB) |
|-------|-----------|---------|------------|
| short | 19.3 | 8.5 | — |
| followup | 47.0 | 9.0 | — |
| longtail | 42.7 | 7.9 | 1359 |

### creative_write (Creative Writing)

| Phase | Prompt t/s | Gen t/s | VRAM (MiB) |
|-------|-----------|---------|------------|
| short | 34.1 | 8.1 | — |
| followup | 38.2 | 9.2 | — |
| longtail | 42.9 | 8.2 | 1359 |

### translate (EN→CN Translation)

| Phase | Prompt t/s | Gen t/s | VRAM (MiB) |
|-------|-----------|---------|------------|
| short | 36.0 | 8.2 | — |
| followup | 72.2 | 8.8 | — |
| longtail | 69.9 | 7.9 | 1363 |

---

## Summary Statistics

| Metric | DS-V2-Lite (Q2_K) |
|--------|-------------------|
| Gen speed range | 7.9–9.5 t/s |
| Gen speed average | ~8.4 t/s |
| Prompt speed range | 18.5–83.8 t/s |
| VRAM (all longtail avg) | ~1361 MiB (1359–1363) |
| VRAM compression ratio | 6.0 GB / 1.36 GB ≈ **4.41×** |

### Cross-Domain Analysis

| Domain | Avg Gen t/s | vs Overall Mean |
|--------|-------------|----------------|
| codegen | 8.73 | +3.9% |
| debug | 8.73 | +3.9% |
| math | 8.53 | +1.5% |
| logic | 8.53 | +1.5% |
| general_qa | 8.27 | -1.5% |
| chinese_tech | 8.47 | +0.8% |
| creative_write | 8.50 | +1.2% |
| translate | 8.30 | -1.2% |

Cross-domain variance **<5%**, consistent with Qwen3.6 — Gen speed is domain-independent.

---

## Full Comparison: DS-V2-Lite vs Qwen3.6-35B-A3B

| Metric | Qwen3.6 (IQ2_M, 10.7 GB) | DS-V2-Lite (Q2_K, 6.0 GB) |
|--------|--------------------------|---------------------------|
| Model size | 10.7 GB | 6.0 GB |
| Gen speed range | 5.0–6.6 t/s | 7.9–9.5 t/s |
| **Gen speed average** | **~6.0 t/s** | **~8.4 t/s** (+40%) |
| Prompt speed range | 10–119 t/s | 18–84 t/s |
| **VRAM average** | **~2242 MiB** | **~1361 MiB** (-39%) |
| VRAM compression ratio | 10.7 / 2.24 ≈ **4.78×** | 6.0 / 1.36 ≈ **4.41×** |
| Cross-domain Gen variance | <3% | <5% |
| Output quality | ✅ Usable (IQ2_M retains basic semantics) | ❌ Q2_K severe degradation (repetition/hallucination) |
| Use case | Daily reasoning, tech Q&A | Speed/VRAM experiments only, not for practical use |

### Key Findings

1. **DS-V2-Lite not fully CPU-offloaded by binary** — 1.36 GB VRAM vs 6.0 GB model = 4.41× compression, similar to Qwen3.6's 4.78×, suggesting the force-cpu binary also partially offloaded this architecture
2. **Gen speed 40% faster than Qwen3.6** — mainly due to smaller model (6 GB vs 10.7 GB), but Q2_K quality degradation negates this advantage
3. **VRAM stable at 1.36 GB** — extremely low, runs easily on 8 GB consumer GPUs
4. **Q2_K quantization severely degrades large models** — all longtail tests showed severe repetition loops, DS-V2-Lite at Q2_K is practically unusable
