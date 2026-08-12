# MoE Expert Locality Study — Domain Clustering in Router Activation

> Date: 2026-07-28 (updated 2026-07-29)
> Models: DeepSeek-V2-Lite (Chat-Uncensored, Q2_K) + Qwen3.6-35B-A3B (IQ2_M)
> Structure: DS-V2-Lite = 27 layers (layer 0 dense, 64 experts top-6); Qwen3.6 = 40 layers (256 experts top-8×2)
> Method: capture actual expert routing (per-layer expert IDs) across 8 domains × 3 phases (short / follow-up / long-tail) = 24 runs, ~3.58M routing decisions

---

## TL;DR

**MoE routers do exhibit domain locality, but the manifestation differs fundamentally by model:**

- **DS-V2-Lite (64 experts)**: domain expert sets are **almost mutually exclusive** (Python vs router prompts share zero experts)
- **Qwen3.6 (256 experts, A3B)**: all 256 experts appear in **every** domain, but **47-78% of experts per layer are domain-specific**

The core MoE-L2 hypothesis (domain → expert locality) holds, but the constraints are model-dependent. In fine-grained MoE (256+ experts), locality manifests at the **per-layer distribution** level, not the global expert-set level.

---

## Method

Expert routing was captured with an instrumentation build of llama.cpp that logs per-layer expert IDs (via an `LLAMA_EXPERT_LOG`-style hook) during real generation on an RTX 4090.

**Test domains (8):** codegen, debug, math, logic, general_qa, chinese_tech, creative_write, translate

**Per-domain phases:**
- short (`-n 128`): ~131 tokens
- followup (`-n 8`): ~11 tokens, conversational follow-up
- longtail (`-n 512`): ~515 tokens

**Data scale:** 8 domains × 3 phases = 24 runs, ~224,000 log lines × 40 layers × 16 experts ≈ **3.58M expert routing decisions**

---

## Key Findings

### 1. Globally: all 256 experts appear in all domains (opposite of DS-V2-Lite)

Cross-domain Jaccard similarity of expert *sets*: **1.000 for all 8 domains**. Every domain uses the identical 256-expert universe.

| Metric | DS-V2-Lite (64 exp) | Qwen3.6 (256 exp) |
|--------|--------------------|--------------------|
| Global expert overlap | Python vs Router: **zero** | All 7 domains: **100%** |
| Experts/token | 6 | 16 (8+8) |
| Expert granularity | 64 coarse experts | 256 fine-grained experts |

**Interpretation:** Qwen3.6's 256 experts are highly fine-grained knowledge units. The model doesn't partition knowledge into "one expert per domain" — it recombines the same expert pool with different per-layer activation *distributions* per domain.

### 2. Cross-domain locality appears at the **per-layer** level — increasing with depth

Per-layer cross-domain Jaccard (average over all domain pairs):

| Layer range | Avg Jaccard | Domain-specific expert ratio | Category |
|-------------|------------|------------------------------|----------|
| L00-L05 (shallow) | 0.876-0.982 | **5-32%** | High sharing — base language |
| L06-L15 | 0.650-0.862 | 32-65% | Medium — partial divergence |
| L16-L25 | 0.550-0.648 | 60-72% | Low — clear divergence |
| L26-L34 (deep) | 0.553-0.623 | 66-72% | Low — sustained divergence |
| **L35-L36 (deepest)** | **0.480-0.517** | **78-79%** | **Lowest sharing — strongest domain divergence** |
| L37-L39 (output) | 0.555-0.682 | 55-73% | Recovers — shared output format |

- **L02** is the most domain-similar layer (Jaccard=0.982), only **4.7%** domain-specific experts
- **L35** is the most domain-divergent layer (Jaccard=0.471), **78.4%** domain-specific experts
- Middle-deep layers (L10-L34) are **60-72% domain-specific** — the primary region where domain-aware caching can save H2D transfers

### 3. 10 backbone experts appear in every domain's top-15

```
Experts in ALL domains' top-15: 10
  IDs: [41, 72, 89, 95, 112, 127, 191, 217, 221, 231]
```

Expert 112 is the most globally active (top-1/2 in every domain) but only appears in 10-14 of 40 layers — these are **layer-specific "hot experts"**, not per-layer constants. Unlike DS-V2-Lite's backbone experts (1, 11) which appear in all layers of all prompts, Qwen3.6's backbone experts are layer-specific but cross-domain consistent.

### 4. Activation distribution is relatively uniform (unlike DS-V2-Lite)

| Domain | Gini | Top-5% share | Bottom-50% share |
|--------|------|--------------|------------------|
| codegen | -0.265 | 9.8% | 30.9% |
| debug | -0.271 | 10.1% | 30.5% |
| math | -0.262 | 9.9% | 31.0% |
| logic | -0.262 | 9.8% | 31.1% |
| general_qa | -0.261 | 9.9% | 31.3% |
| chinese_tech | -0.259 | 9.9% | 31.3% |
| creative_write | -0.256 | 9.8% | 31.9% |
| translate | -0.275 | 10.1% | 30.3% |

All 8 domains look nearly identical: top-5% experts carry ~10% of activations, bottom-50% carry ~31%. No strong "rich-get-richer" pattern — expert usage is much more uniform than DS-V2-Lite.

### 5. Most domain-discriminative experts

| Expert | CV | Domain distribution (activation counts) |
|--------|-----|------------------------------------------|
| **5** | 0.877 | codegen:4, debug:2, **math:10**, logic:1, gen_qa:1, chinese:4, creative:4, translate:3 |
| 28 | 0.630 | codegen:3, debug:3, **math:13**, logic:5, gen_qa:7, chinese:3, creative:4, translate:3 |
| 118 | 0.622 | codegen:4, debug:1, math:5, logic:7, gen_qa:7, chinese:2, creative:4, translate:4 |
| 101 | 0.542 | codegen:4, debug:3, math:4, logic:3, gen_qa:8, chinese:10, creative:6, translate:3 |

Experts 5 and 28 lean strongly toward math. But absolute CV values are modest — the 8 domains are too similar overall for "exclusive" experts; differences are quantitative, not qualitative.

---

## DS-V2-Lite vs Qwen3.6: Summary

| Dimension | DS-V2-Lite (64 exp, top-6) | Qwen3.6 (256 exp, top-8×2) |
|-----------|----------------------------|----------------------------|
| Activated experts/layer | 6 | 16 (8+8) |
| Activation ratio | 6/64 = 9.4% | 16/256 = 6.25% |
| Cross-domain expert set | Python vs Router: **zero overlap** | All 8 domains: **100% overlap** |
| Backbone experts | 2 (1, 11) — all layers, all domains | 10 — layer-specific, cross-domain consistent |
| Per-layer domain-specific ratio | Python-specific > 50% | Middle-deep layers **60-78%** |
| Usage skew | High (some experts far more active) | Low (relatively uniform) |

**Likely causes:**
1. **Expert granularity (primary)**: 64 coarse experts vs 256 fine-grained ones. DS-V2-Lite picks 6/64 = 9.4% "big modules"; Qwen3.6 picks 16/256 = 6.25% "small pattern fragments". Domain shifts in DS-V2-Lite require swapping experts; in Qwen3.6 they're expressed by recombining the same pool.
2. **A3B architecture**: 3B of 35B activated per token. The router must cover knowledge more broadly with limited activation budget.

---

## Implications for MoE-L2 Architecture

1. **L0a domain predictor value confirmed ✅**: even though Qwen3.6's expert *universe* is shared across domains, per-layer distributions diverge strongly (deepest layers Jaccard=0.47). After a domain prediction:
   - L0-L5 (~20% shared): no domain-awareness needed, uniform cache policy
   - L6-L36 (60-78% domain-specific): domain-aware preloading saves significant H2D transfers

2. **LRU cache policy must adapt**:
   - **Per-layer independent caching**: expert sets differ per layer, no cross-layer sharing
   - **Domain-aware per-layer budget allocation**: different domains need different per-layer cache policies
   - **Backbone experts prioritized**: the 10 cross-domain backbone experts are stable across domains

3. **Router behavior is architecture-specific**: routing policies differ between models (64/256/8 experts). A general solution needs validation across more architectures (DeepSeek-V3, Qwen2.5-MoE, Mixtral).

4. **Baseline cache benefit exists even without domain prediction**: with fine-grained routing, per-token activation is 16/256 — repeated selection across tokens means a simple per-layer LRU still has cache headroom (top-5% expert distribution is identical across domains → stable backbone set).

---

## Appendix: Verified Dead Ends

| Direction | Why it fails |
|-----------|--------------|
| Domain→global-expert-set caching (DS-V2-Lite style) | Qwen3.6 shares all 256 experts across domains — can't map this way |
| Unifying Qwen3.6 and DS-V2-Lite routing behavior | The two models' routing policies are fundamentally different |
