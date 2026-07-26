<![CDATA[# Architecture

## Overview

moe-l2 is a **user-space L2 cache scheduler** for MoE inference. It sits between the user and ollama/llama.cpp, predicting which experts will be needed and preloading them into shared memory before the model requests them.

## Components

```
┌─────────────────────────────────────────────────────┐
│                    moe-l2 Scheduler                   │
│                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────┐ │
│  │ L0a Predictor │──>│ L2 Cache Mgr │──>│ Proxy    │ │
│  │ (keyword)     │   │ (mmap + LRU) │   │ (HTTP)   │ │
│  └──────────────┘   └──────────────┘   └────┬─────┘ │
│                                              │       │
│  ┌──────────────┐                            │       │
│  │ L3 Loader    │◄───────────────────────────┘       │
│  │ (SSD→RAM)    │                                    │
│  └──────────────┘                                    │
└─────────────────────────────────────────────────────┘
         │                                    │
         │ user connects here                  │ forwards to
         ▼                                    ▼
  ┌──────────────┐                  ┌──────────────────┐
  │ Client       │                  │ ollama/llama.cpp  │
  │ (localhost:  │                  │ (unchanged)       │
  │  11435)      │                  └──────────────────┘
  └──────────────┘
```

## Data Flow

1. **User sends prompt** to moe-l2 proxy (`localhost:11435`)
2. **L0a Predictor** analyzes prompt → outputs a domain label (e.g. "codegen")
3. **L2 Cache Manager** looks up domain→expert mapping, starts async preload of predicted experts from SSD into shared memory
4. **Proxy** forwards the request to ollama (`localhost:11434`)
5. **During inference**, when llama.cpp needs an expert:
   - **L2 hit**: expert already in shared memory → ~1150 µs memcpy
   - **L2 miss**: fall back to mmap from SSD → ~6500 µs

## Memory Hierarchy

| Level | Location | Contents | Latency |
|-------|----------|----------|---------|
| L0 (GPU VRAM) | GPU | Universal experts + LRU hot set | ~µs (native access) |
| L2 (RAM) | System memory | Preloaded cold experts (mmap SHM) | ~1150 µs (memcpy to GPU) |
| L3 (SSD) | Disk | Full model weights (GGUF mmap) | ~6500 µs (page fault) |

## Key Design Decisions

- **No llama.cpp modifications**: works as an external proxy
- **Shared memory (mmap)**: lets llama.cpp access preloaded weights without inter-process copy
- **Keyword-based predictor**: zero dependencies, fast, good enough for Phase 2
- **Per-layer LRU**: each of the 40 layers has its own cache with independent LRU order
- **Pinned experts**: universal experts (present across all domains) are never evicted

## Verification Results

See [benchmark.md](./benchmark.md) for detailed numbers.
]]>