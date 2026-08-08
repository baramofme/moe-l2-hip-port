# Contributing to moe-l2

Thanks for your interest in moe-l2! This project makes large MoE models runnable on low-VRAM consumer GPUs. Every contribution — bug reports, benchmark data, docs, code — helps.

## Quick links

- **Report a bug** → open an [issue](https://github.com/yalun753/moe-l2/issues)
- **Ask a question** → open a discussion (or comment on a relevant issue)
- **Share benchmark results** → open an issue with your GPU / model / measured numbers (see below)

## What we value

1. **Real measured data over claims.** This project is built on reproducible benchmarks. If you run moe-l2 on hardware we haven't tested (especially 4-12 GB consumer GPUs), please share your numbers.
2. **Minimal, focused changes.** Keep PRs small and tied to one concern.
3. **Backward compatibility.** Users run this on old GPUs (GTX 1080 → RTX 50). Don't break them.

## Development setup

```bash
# 1. Install the package (editable)
pip install -e ".[predictor]"

# 2. Fetch pre-built CUDA binaries (for --gpu mode)
moe-l2 download-bins

# 3. Run the unit tests
python -m pytest tests/ -v
```

Requirements: Linux x86_64, Python 3.9+, NVIDIA GPU + CUDA driver for GPU mode (CPU-only mode works without).

## Where things live

| Path | What it is |
|------|-----------|
| `moe_l2/` | Python package: predictor, L2 cache, proxy, CLI, classifier |
| `scripts/` | Installer (`install.sh`) and helper scripts |
| `references/` | Benchmark reports, design decisions, verification docs |
| `examples/` | Demo scripts and assets |

## Submitting code

1. Fork the repo and create a feature branch.
2. Make your change — keep it small and focused.
3. Run tests: `python -m pytest tests/ -v`
4. If you touch the C++/llama.cpp layer (under `moe_l2/bin` source or the referenced patch), note that **binaries are distributed via GitHub Release**, not the PyPI wheel — a source patch alone won't change the shipped behavior until a new `bins-vX.Y.Z` is cut.
5. Open a PR with a clear description: what changed, why, and (for performance changes) before/after numbers.

## Reporting benchmark results

We love new data points. When sharing benchmark numbers, include:

- GPU model + VRAM (e.g. "RTX 3060 12 GB")
- Model + quantization (e.g. "Qwen3.6-35B-A3B-UD-IQ2_M")
- Command used (e.g. `moe-l2 start --model ... --gpu`)
- Prompt length / context (`-c`), tokens generated
- VRAM peak + generation t/s (and prompt t/s if you have it)

## Code of conduct

Be respectful. This is a solo-maintainer project — patient, specific feedback gets answered; demands don't.
