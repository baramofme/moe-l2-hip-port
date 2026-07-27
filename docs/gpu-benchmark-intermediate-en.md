# MoE-L2 GPU Memory Benchmark — Intermediate Results

**Model**: Qwen3.6-35B-A3B-UD-IQ2_M (10.7 GB)  
**Hardware**: RTX 4090 24GB (AutoDL)  
**Config**: Force-CPU-Experts (baseline VRAM already compressed)  
**Binary**: `/root/llama-cli-force-cpu-experts-20260901`  
**Date**: 2026-07-26

---

## Summary Table

| Domain | Short Prompt | Short Gen | Followup Prompt | Followup Gen | Longtail Prompt | Longtail Gen | VRAM (MiB) |
|--------|:-----------:|:---------:|:-------------:|:-----------:|:--------------:|:-----------:|:----------:|
| **codegen** | 10.1 | 6.2 | 119.1 | 5.3 | 39.5 | 6.1 | 2231~2241 |
| **debug** | 40.4 | 6.0 | 69.5 | 5.7 | 47.3 | 6.2 | 2235~2243 |
| **math** | 44.6 | 6.1 | 70.2 | 5.8 | 46.8 | 6.3 | 2243 |

**Progress**: 3/8 domains × 3 phases = 9/24 tests completed

---

## Raw Per-Test Data

### codegen (Code Generation)

**Short** (-n 128, -c 512)
```
[ Prompt: 10.1 t/s | Generation: 6.2 t/s ]
VRAM: 2231 MiB
```

**Followup** (-n 8, -c 1536) — long context + short response
```
[ Prompt: 119.1 t/s | Generation: 5.3 t/s ]
VRAM: ~2235 MiB (too fast to capture)
```

**Longtail** (-n 512, -c 1024)
```
[ Prompt: 39.5 t/s | Generation: 6.1 t/s ]
VRAM: 2241 MiB
```

### debug (Code Debugging)

**Short**
```
[ Prompt: 40.4 t/s | Generation: 6.0 t/s ]
VRAM: 2235 MiB
```

**Followup**
```
[ Prompt: 69.5 t/s | Generation: 5.7 t/s ]
VRAM: ~2235 MiB (too fast to capture)
```

**Longtail**
```
[ Prompt: 47.3 t/s | Generation: 6.2 t/s ]
VRAM: 2243 MiB
```

### math (Math Reasoning)

**Short**
```
[ Prompt: 44.6 t/s | Generation: 6.1 t/s ]
VRAM: ~2235 MiB (nvidia-smi didn't capture post-load)
```

**Followup**
```
[ Prompt: 70.2 t/s | Generation: 5.8 t/s ]
VRAM: ~2235 MiB (too fast to capture)
```

**Longtail**
```
[ Prompt: 46.8 t/s | Generation: 6.3 t/s ]
VRAM: 2243 MiB
```

---

## Initial Observations

### VRAM
- All tests stable at **2231~2243 MiB** (~2.2 GiB)
- Short vs longtail VRAM difference minimal (<12 MiB), model weights dominate
- Context length impact on VRAM is negligible

### Generation Speed
- Stable **5.3 ~ 6.3 t/s** range
- Followup Gen slightly slower (5.3~5.8), possibly due to longer context increasing KV cache access overhead
- Low cross-domain variance

### Prompt Processing Speed
- **Short prompts** (10~40 tokens): 10~45 t/s
- **Long context** (300~700 tokens): 69~119 t/s — longer prompts benefit from better parallelization
- Followup and longtail prompt speeds significantly higher than short

---

## Remaining

Remaining 5 domains × 3 phases = 15 tests:

| Domain | Status |
|--------|--------|
| logic | ⏳ Not run |
| general_qa | ⏳ Not run |
| chinese_tech | ⏳ Not run |
| creative_write | ⏳ Not run |
| translate | ⏳ Not run |

Prompt files already uploaded to cloud GPU at `/root/prompts/`. Test pattern is verified — can continue directly.
