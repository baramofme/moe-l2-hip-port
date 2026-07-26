<![CDATA[# moe-l2 Roadmap

## Phase 1 ✅ — Core Hypothesis Validation (Completed)

- [x] Pool mechanism exists in ggml_gallocr (confirmed)
- [x] Pool hit (memcpy ~1150µs) vs miss (mmap ~6500µs): ~5.6x difference
- [x] Pool size has zero impact on CPU generation speed
- [x] Minimum viable pool: 16 MB
- [x] DS-V2-Lite domain-expert affinity (Python vs router: zero intersection)
- [x] Qwen3.6 8-domain expert routing data collected (224K+ lines)
- [x] LRU cache simulation: 84.4% hit rate at 96 slots/layer

## Phase 1.5 ✅ — Feasibility Confirmation (Completed)

- [x] Soft Domain Preference confirmed across 8 domains
- [x] Domain-expert mapping theoretically viable
- [x] Decision gate passed: proceed to Phase 2

## Phase 2 🚧 — L2 Scheduler Prototype (In Progress)

**Goal:** Working L2 cache that accelerates MoE inference on consumer GPUs.

### Milestone 2.1 — Core Infrastructure
- [ ] L0a domain predictor (keyword matching) ← skeleton done
- [ ] L2 cache manager (mmap shared memory) ← skeleton done
- [ ] ollama transparent proxy ← skeleton done
- [ ] CLI: `moe-l2 start`, `moe-l2 stats` ← skeleton done

### Milestone 2.2 — Integration
- [ ] Generate domain→expert mapping from Phase 1 data
- [ ] Wire predictor → cache → proxy together
- [ ] Test with real model: single-domain chat
- [ ] Measure end-to-end latency vs bare ollama

### Milestone 2.3 — Hardening
- [ ] Docker deployment support
- [ ] Graceful degradation on prediction miss
- [ ] Error handling + logging

## Phase 3 📋 — Optimization (Planned)

- [ ] Upgrade predictor: keyword → lightweight classifier
- [ ] Embed domain→expert map in GGUF metadata
- [ ] Multi-model support (DS-V2, DBRX, others)
- [ ] Smooth domain transition strategy
- [ ] Performance tuning based on real usage data

## Future 📋 — Release & Ecosystem

- [ ] PyPI release: `pip install moe-l2`
- [ ] Comprehensive README + examples
- [ ] GitHub Discussions for community feedback
- [ ] ollama/llama.cpp integration PR/issue

---

## How to Contribute

Phase 2 is open for contributions. High-value first tasks:

1. **Domain→expert map**: process the 48 log files into a structured JSON mapping
2. **Real proxy forwarding**: replace the stub in `proxy.py` with actual HTTP forwarding
3. **Cache integration**: wire predictor output to trigger cache preload
4. **Testing**: test with real ollama + Qwen3-30B on 8GB GPU
]]>