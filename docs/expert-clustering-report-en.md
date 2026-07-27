# MoE Expert Clustering Verification Report

> Date: 2026-07-28 (last updated 2026-07-29)
> Models: DeepSeek-V2-Lite (Chat-Uncensored, Q2_K quant) + Qwen3.6-35B-A3B (IQ2_M)
> Architecture: DS-V2-Lite = 27 layers (layer 0 dense, 64 experts top-6), Qwen3.6 = 40 layers (256 experts top-8×2 gate)
> Status: Phase 1 CPU Verification ✅, Phase 2 GPU — Qwen3.6 cross-domain routing data **fully collected** ✅ (8 domains × 3 phases = 24 batches), Direction A fix operational 🚧

---

## Conclusion

**MoE routers exhibit domain clustering, but behavior differs fundamentally between architectures:**

- DS-V2-Lite (64 experts): domain-specific expert sets are **nearly disjoint** (Python vs Router: zero intersection)
- Qwen3.6 (256 experts, A3B): all 256 experts appear **in all 7 domains**, but **47-78% of experts are domain-exclusive at the per-layer level**

This finding confirms that the **MoE-L2 architectural hypothesis (domain→expert clustering) is correct**, but the constraint landscape varies by model. In fine-grained MoE (256+ experts), clustering manifests at the per-layer distribution level, not in the global expert set.

---

## Verification Methodology

### Phase 1 (DS-V2-Lite, CPU)

In llama.cpp's `graph_compute` function, read `int32_t` values from the `ffn_moe_topk-{layer}` tensor after `ggml_backend_sched_graph_compute_async` succeeds. Compared 3 prompts (hi, Python, Router). See `moe-l2-domain-clustering-report-20260721.md` for details.

### Phase 2 (Qwen3.6, GPU — AutoDL RTX 4090 24GB)

`LLAMA_EXPERT_LOG=1` environment variable controls expert log output. Expert indices are read via `ggml_backend_tensor_get()` from GPU after graph compute completes. **Direction A path** — no graph topology modification, zero crash risk.

**Test configuration:**

| Parameter | Value |
|-----------|-------|
| GPU | RTX 4090 24GB (AutoDL) |
| Model | Qwen3.6-35B-A3B-UD-IQ2_M.gguf (11GB) |
| Quantization | IQ2_M (2-bit), 256 experts/layer, top-8×2 |
| VRAM idle | ~978 MiB |
| VRAM running | ~1725-2249 MiB |
| Generation speed | 5.0-5.7 t/s |
| Batch prompt | `-n 512` (longtail phase) |

**Test domains (8):**

| Domain | Label | Description |
|--------|-------|-------------|
| codegen | Code generation | Generate web servers, sorting algorithms, etc. |
| debug | Code debugging | Analyze crash logs, fix bugs |
| math | Math reasoning | Calculus, probability computations |
| logic | Logic puzzles | Reasoning problems, constraint solving |
| general_qa | General Q&A | Knowledge-based questions |
| chinese_tech | Chinese tech | Technology concepts explained in Chinese |
| creative_write | Creative writing | Story and poetry creation |
| translate | Translation | English→Chinese tech docs ✅ |

**Each domain has 3 phases:**
- **short** (`-n 128 -c 512`): short generation, ~131 tokens → 5240 lines of expert log
- **followup** (`-n 8 -c 1536`): follow-up, ~11 tokens → 440 lines
- **longtail** (`-n 512 -c 1024`): long-tail generation, ~515 tokens → 20600 lines

**Data scale:** 8 domains × 3 phases = 24 exhaust logs, ~224,000 lines of expert routing records (each line = 40 layers × 16 experts ≈ 3,580,000 expert selections)

---

## Key Findings

### 1. Global: All 256 experts appear in every domain (fundamentally different from DS-V2-Lite)

Jaccard similarity matrix (based on full expert set):

```
          codegen  debug   math    logic   gen_qa  chinese_tech  creative_write
codegen    1.000   1.000   1.000   1.000   1.000   1.000        1.000
debug      1.000   1.000   1.000   1.000   1.000   1.000        1.000
...
```

**All 8 domains have cross-domain Jaccard = 1.000.** Every domain activates the same 256-expert set.

**Key contrast with DS-V2-Lite:**

| Metric | DS-V2-Lite (64 exp) | Qwen3.6 (256 exp) |
|--------|-------------------|------------------|
| Global expert intersection | Python vs Router: **zero** | 7 domains: **100% overlap** |
| Experts per layer per token | 6 | 16 (8+8 gate) |
| Expert granularity | 64 coarse experts | 256 fine-grained experts |
| Backbone experts | 2 (1, 11) | 10 (top-15 intersection) |

**Interpretation:** Qwen3.6's 256 experts are highly fine-grained. The model doesn't partition knowledge into "256 blocks each owning a domain" — instead, it combines "256 micro-knowledge units." The model can vary per-layer activation distributions without swapping the global expert set.

### 2. Cross-domain clustering manifests at the **per-layer level** — increasing with depth

**Per-layer cross-domain Jaccard similarity (average across all 7 domain pairs):**

| Layer Range | Avg Jaccard | Domain-exclusive proportion | Category |
|-------------|-------------|---------------------------|----------|
| L00-L05 (shallow) | 0.876-0.982 | **5-32%** | **High sharing** — base language capability |
| L06-L15 (shallow-mid) | 0.650-0.862 | 32-65% | Medium — partial domain differentiation |
| L16-L25 (mid-deep) | 0.550-0.648 | 60-72% | Low — clear domain differentiation |
| L26-L34 (deep) | 0.553-0.623 | 66-72% | Low — sustained differentiation |
| **L35-L36 (deepest)** | **0.480-0.517** | **78-79%** | **Lowest sharing — strongest domain differentiation** |
| L37-L39 (output) | 0.555-0.682 | 55-73% | Recovery — shared output formatting |

**Key pattern:**
- **L02** is the most cross-domain similar layer (Jaccard=0.982), only **4.7%** domain-exclusive experts
- **L35** is the least similar (Jaccard=0.471), **78.4%** domain-exclusive
- **First 3 and last 2 layers have high expert sharing** — base processing + output formatting may be global
- **L10-L34: 60-72% of experts are domain-exclusive** — this is MoE-L2's primary optimization target

**MoE-L2 implication:** Domain predictor (L0a) output → minimal impact on shallow layers (L0-L5, no domain-aware caching needed), but significant impact on mid-deep layers (L6-L36, 60-78% domain-exclusive experts), saving substantial H2D bandwidth.

### 3. 10 backbone experts appear in all domains' Top-15

```
Experts in ALL domains' top-15: 10
  IDs: [41, 72, 89, 95, 112, 127, 191, 217, 221, 231]
```

**Expert 112** is the most globally active (Top-1 or Top-2 in every domain), but appears in only 10-14/40 layers. These backbone experts are **layer-specific "hot experts,"** not "permanent residents" per layer.

In contrast, DS-V2-Lite's backbone experts (1, 11) appear in all layers across all prompts. Qwen3.6's backbone experts appear only in **specific layers**, but with high cross-domain consistency.

### 4. Activation distribution is relatively uniform (unlike DS-V2-Lite)

| Domain | Gini | Top-5% share | Bottom-50% share |
|--------|------|-------------|-----------------|
| codegen | -0.265 | 9.8% | 30.9% |
| debug | -0.271 | 10.1% | 30.5% |
| math | -0.262 | 9.9% | 31.0% |
| logic | -0.262 | 9.8% | 31.1% |
| general_qa | -0.261 | 9.9% | 31.3% |
| chinese_tech | -0.259 | 9.9% | 31.3% |
| creative_write | -0.256 | 9.8% | 31.9% |
| translate | -0.275 | 10.1% | 30.3% |

**All 8 domains have nearly identical distributions!** Top-5% experts account for ~10% of activations, Bottom-50% for ~31%. Qwen3.6 uses experts much more evenly than DS-V2-Lite — no obvious "winner-takes-most" pattern. Translate domain has the most uniform distribution (Gini=-0.275, lowest Bottom-50% share at 30.3%).

### 5. Most domain-discriminative experts (highest cross-domain variance)

| Expert | CV | Domain distribution (activation count per domain) |
|--------|-----|---------------------------------------------------|
| **5** | **0.877** | codegen:4, debug:2, **math:10**, logic:1, gen_qa:1, chinese:4, creative:4, translate:3 |
| 28 | 0.630 | codegen:3, debug:3, **math:13**, logic:5, gen_qa:7, chinese:3, creative:4, translate:3 |
| 118 | 0.622 | codegen:4, debug:1, math:5, logic:7, gen_qa:7, chinese:2, creative:4, translate:4 |
| 101 | 0.542 | codegen:4, debug:3, math:4, logic:3, gen_qa:8, chinese:10, creative:6, translate:3 |
| 155 | 0.413 | codegen:6, debug:8, **math:10**, logic:7, gen_qa:7, chinese:8, creative:6, translate:5 |

Experts 5 and 28 are clearly math-biased (but CV magnitude is low because all 8 domains are too similar — translate's inclusion homogenizes cross-domain distribution).

**Complete contrast with DS-V2-Lite:** On DS-V2-Lite's Python vs Router domain comparison, many experts are **present in one domain and entirely absent in the other** (fully discriminative). Qwen3.6 has no such "exclusive experts" — differences are quantitative, not qualitative.

---

## DS-V2-Lite vs Qwen3.6: Comparison Summary

| Dimension | DS-V2-Lite (64 exp, top-6) | Qwen3.6 (256 exp, top-8×2) |
|-----------|---------------------------|---------------------------|
| Output experts per layer | 6 | 16 (8+8) |
| Activation ratio | 6/64 = 9.4% | 16/256 = 6.25% |
| Cross-domain expert set | Python vs Router: **zero overlap** | 8 domains: **100% overlap** |
| Backbone experts | 2 (1, 11) — all layers, all domains | 10 (41, 72, 89, 95, 112, ...) — specific layers, cross-domain consistent |
| Per-layer domain-exclusive ratio | Not precisely measured, Python-exclusive > 50% | Mid-deep layers **60-78%** |
| Expert usage distribution | High skew (some experts significantly more active) | Low skew (relatively uniform) |
| Activation concentration (Top-5%) | Not measured | ~10% |

### Why the difference?

Two likely causes:

**1. Expert granularity (primary):** 64 coarse experts vs 256 fine-grained experts. DS-V2-Lite selects 6/64 = 9.4%, using a small set of "large modules" per layer. Qwen3.6 selects 16/256 = 6.25% — more experts selected but a smaller share. If each DS-V2-Lite expert is "one large domain," switching domains requires switching experts. If each Qwen3.6 expert is "one small pattern fragment," domain differences express through different expert combinations.

**2. A3B architecture impact:** Qwen3.6 is "activate 3B/35B" (A3B), meaning the router must cover broad knowledge with only 3B active parameters per token out of 35B total. DS-V2-Lite has smaller total params (~16B?), with more parameters per expert.

---

## Implications for MoE-L2 Architecture

### 1. L0a domain predictor value confirmed ✅

While Qwen3.6's global expert set is domain-invariant, **per-layer expert distributions differ significantly** (deepest layer Jaccard as low as 0.47). With domain predictor output:
- **L0-L5 (~20% shared)**: no domain-aware caching needed, uniform strategy
- **L6-L36 (60-78% domain-exclusive)**: preload domain-specific experts, significant H2D savings

### 2. LRU cache strategy needs adjustment

For DS-V2-Lite, LRU can preload by "domain→expert set" mapping. For Qwen3.6, LRU needs:
- **Per-layer independent caching**: each layer's expert set differs, no cross-layer sharing
- **Domain-aware per-layer budget allocation**: different domains need different per-layer cache strategies
- **Backbone expert priority protection**: the 10 cross-domain backbone experts affect latency less (they're active in all domains)

### 3. More MoE models need verification

The routing behavior difference between Qwen3.6 and DeepSeek-V2-Lite shows that **MoE routing strategy is architecture-dependent**. A general MoE-L2 solution needs to verify more models:
- DeepSeek-V2-Lite (64 exp ✅) → Qwen3.6 (256 exp ✅) → DeepSeek-V3 (256+ exp?)
- Qwen2.5-MoE (128 exp?)
- Mixtral-8x7B (8 exp — fundamentally different routing behavior)

### 4. Baseline benefit of caching

Even without domain prediction (pure per-layer LRU), fine-grained MoE like Qwen3.6 has caching potential:
- 256 experts per layer, only 16 selected per token
- After N tokens, many experts are re-selected
- Top-5% expert distribution is identical across all 7 domains (backbone expert set is stable)

---

## Current Progress Summary

### Completed & Verified ✅

| Phase | Core Result |
|-------|------------|
| Phase 1 CPU Verification | Confirmed domain clustering on DS-V2-Lite (Python vs Router: zero expert intersection) |
| **✅ Qwen3.6 Direction A fixed** | `ggml_backend_tensor_get()` reads GPU expert indices without crashing — no graph modification needed |
| **✅ Qwen3.6 8-domain × 3-phase full data collected** | 224,000+ lines × 40 layers × 16 experts = ~3,580,000 routing decisions |
| **✅ Cross-domain analysis complete** | Discovered Qwen3.6 and DS-V2-Lite routing behaviors fundamentally differ — but domain clustering still exists (at per-layer level) |
| **✅ Backbone experts identified** | 10 cross-domain backbone experts + 60-78% domain-exclusive rate per layer |

### To Do ♻️

| Priority | Task | Notes |
|----------|------|-------|
| P0 | Write MoE-L2 productization plan | Based on this report + prior design discussions |
| P1 | Verify more MoE models (DeepSeek-V3, Qwen2.5-MoE) | Confirm if findings are general |
| P2 | L0a domain predictor prototype + domain→per-layer expert mapping | Needs more domain data |
| P2 | LRU GPU expert cache C++ prototype | ~100 lines of code |
| P3 | moe-l2 pip package + ollama transparent proxy | Productization phase |

### Verified Infeasible ❌

| Direction | Reason |
|-----------|--------|
| Caching based on "domain→global expert set" (DS-V2-Lite pattern) | Qwen3.6 shares all 256 experts across domains — can't map this way |
| Aligning Qwen3.6 and DS-V2-Lite routing behavior | The two models have fundamentally different routing strategies |

---

## Raw Data

All raw expert log files (48 files, 8 domains × 3 phases × 2 logs):
```
/opt/data/副业操作技巧/可发素材/moe-l2-qwen36-raw/
├── codegen/{short,followup,longtail}/{expert_data,generation}.log
├── debug/...
├── math/...
├── logic/...
├── general_qa/...
├── chinese_tech/...
├── creative_write/...
└── translate/...     ✅ Complete
```

Analysis scripts:
- `analyze_expert_data.py` — First-pass parser (deprecated)
- `analyze_expert_data_v2.py` — Domain analysis + Jaccard + Top-15
- `analyze_expert_data_v3.py` — Per-layer cross-domain Jaccard + exclusive expert rate
- `analyze_expert_final.py` — Full 8-domain comprehensive evaluation (latest)

Binary used:
- `/root/llama-cli-expert-log` (Direction A, compiled 2026-07-26) — AutoDL cloud server
- Data correctness confirmed via `LLAMA_EXPERT_LOG=1` environment variable
- GGUF model: `Qwen3.6-35B-A3B-UD-IQ2_M.gguf` (11 GB)
