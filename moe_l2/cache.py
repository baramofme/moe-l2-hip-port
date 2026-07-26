"""
L2 hot-cache manager — the core of moe-l2.

Manages a shared-memory pool of preloaded expert weights.
Experts are loaded from L3 (SSD/mmap) into L2 (RAM shared memory)
based on domain predictions from the L0a predictor.

Design:
  - mmap-backed shared memory for zero-copy access from llama.cpp
  - LRU eviction when the pool is full
  - Pinned (non-evictable) slots for universal experts
"""

import mmap
import os
import struct
from pathlib import Path
from typing import Optional

# Default cache size (per-layer slot count)
DEFAULT_SLOTS_PER_LAYER = 96

# Where expert weight files live
EXPERT_DIR = Path("/tmp/moe_l2_experts")


class L2Cache:
    """Per-layer LRU cache of expert weights in shared memory."""

    def __init__(
        self,
        n_layers: int = 40,
        slots_per_layer: int = DEFAULT_SLOTS_PER_LAYER,
        expert_size: int = 1_010_000,  # ~1.01 MB per expert (Qwen3.6)
        l2_dir: Optional[Path] = None,
    ):
        self.n_layers = n_layers
        self.slots_per_layer = slots_per_layer
        self.expert_size = expert_size
        self.l2_dir = l2_dir or EXPERT_DIR
        self.l2_dir.mkdir(parents=True, exist_ok=True)

        # Per-layer LRU: slot_id -> expert_id (or -1 if empty)
        # Track access order for eviction
        self._slots: list[list[int]] = [
            [-1] * slots_per_layer for _ in range(n_layers)
        ]
        self._lru_order: list[list[int]] = [
            [] for _ in range(n_layers)
        ]
        self._pinned: list[set[int]] = [set() for _ in range(n_layers)]

        # Shared memory handles (TODO: real mmap setup)
        self._shm_handles: dict[tuple[int, int], int] = {}  # (layer, slot) -> fd

    def pin_experts(self, layer: int, expert_ids: list[int]) -> None:
        """Mark experts as pinned (never evicted)."""
        self._pinned[layer].update(expert_ids)

    def load(self, layer: int, expert_id: int) -> bool:
        """Load an expert from L3 (SSD) into L2 cache.

        Returns True if the expert was loaded (or already present).
        Returns False if the cache is full and no evictable slot.
        """
        # Check if already cached
        if expert_id in self._slots[layer]:
            self._touch(layer, expert_id)
            return True

        # Find eviction slot
        slot = self._find_evictable_slot(layer)
        if slot is None:
            return False  # cache full, all pinned

        # Load from disk (placeholder)
        self._load_from_disk(layer, expert_id, slot)

        # Update LRU tracking
        self._slots[layer][slot] = expert_id
        self._touch_slot(layer, slot)
        return True

    def contains(self, layer: int, expert_id: int) -> bool:
        """Check if an expert is already in L2 cache."""
        return expert_id in self._slots[layer]

    def stats(self) -> dict:
        """Return cache statistics."""
        total_slots = self.n_layers * self.slots_per_layer
        used_slots = sum(
            1 for layer_slots in self._slots for s in layer_slots if s != -1
        )
        pinned_count = sum(len(p) for p in self._pinned)
        return {
            "total_slots": total_slots,
            "used_slots": used_slots,
            "pinned_experts": pinned_count,
            "utilization_pct": round(used_slots / total_slots * 100, 1) if total_slots else 0,
        }

    # ── Internal helpers ─────────────────────────────────────────

    def _touch(self, layer: int, expert_id: int) -> None:
        """Mark an expert as most recently used."""
        if expert_id in self._pinned[layer]:
            return  # pinned experts don't move in LRU
        for i, eid in enumerate(self._lru_order[layer]):
            if eid == expert_id:
                self._lru_order[layer].pop(i)
                self._lru_order[layer].append(expert_id)
                return

    def _touch_slot(self, layer: int, slot: int) -> None:
        expert_id = self._slots[layer][slot]
        if expert_id in self._pinned[layer]:
            return
        if expert_id in self._lru_order[layer]:
            self._lru_order[layer].remove(expert_id)
        self._lru_order[layer].append(expert_id)

    def _find_evictable_slot(self, layer: int) -> Optional[int]:
        """Find the LRU slot that isn't pinned. Returns None if all pinned."""
        for i, expert_id in enumerate(self._slots[layer]):
            if expert_id == -1:
                return i  # empty slot
            if expert_id not in self._pinned[layer]:
                # LRU candidate — first unpinned slot
                return i
        return None

    def _load_from_disk(self, layer: int, expert_id: int, slot: int) -> None:
        """Load expert weights from SSD into shared memory.

        Phase 2: placeholder — creates a zero-filled file.
        Phase 3: real mmap read from GGUF model file.
        """
        expert_path = self.l2_dir / f"L{layer}_E{expert_id}.bin"
        if not expert_path.exists():
            # Create sparse placeholder
            with open(expert_path, "wb") as f:
                f.write(b"\x00" * self.expert_size)
        # TODO: mmap the file into shared memory accessible by llama.cpp
