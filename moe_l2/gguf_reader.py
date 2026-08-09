"""GGUF weight reader for MoE-L2 cache.

Reads MoE expert weights directly from GGUF-format model files.
Supports quantized formats (Q2_K, Q3_K, etc.) via gguf-python's memmap.

The gguf-python library exposes quantized tensors as uint8 NDArrays
with the expert dimension as the *first* axis, e.g.:

    blk.0.ffn_gate_exps.weight  shape=(2, 8960, 504)
    blk.0.ffn_down_exps.weight  shape=(2, 1536, 3850)

So t.data[expert_id] gives the full quantized weights for one expert.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from gguf import GGUFReader as _GGUFReader


class MoEGGUFReader:
    """Read MoE expert weights from a GGUF model file.

    Provides:
      - Metadata: num_layers, num_experts, architecture
      - Per-expert byte sizes
      - Raw weight extraction per (layer, expert_id) for gate+up+down

    Thread-safe for concurrent reads (each call reads independently
    from the memory-mapped file).
    """

    # Recognised expert tensor patterns (in order: gate, up, down)
    EXPERT_PATTERNS = [
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
    ]

    # Model metadata keys to probe for layer/expert counts
    _LAYER_KEYS = [
        "qwen2moe.block_count",
        "deepseek2.block_count",
        "llama.block_count",
        "qwen2.block_count",
    ]
    _EXPERT_KEYS = [
        "qwen2moe.expert_count",
        "deepseek2.expert_count",
    ]

    def __init__(self, model_path: str | Path):
        self.path = Path(model_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Model not found: {self.path}")

        self._reader = _GGUFReader(str(self.path))
        self._tensors: dict[str, _GGUFReader.TensorInfo] = {}
        self._build_index()

        # Cache metadata
        self._num_layers: Optional[int] = None
        self._num_experts: Optional[int] = None
        self._arch: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _get_field_value(field) -> int | str | None:
        """Extract the actual value from a ReaderField.

        In gguf-python, each ReaderField has parts array. For metadata
        KV fields, the *last* part (-1) is always the value:
          - UINT32/UINT64: memmap with a single element → .item()
          - STRING:        uint8 bytes → decode('utf-8')
        """
        raw = field.parts[-1]
        if field.types[-1] == 8:  # GGUFValueType.STRING
            return bytes(raw).decode("utf-8", errors="replace")
        return int(raw.item())

    @property
    def num_layers(self) -> int:
        """Number of transformer layers (auto-detected from metadata or tensors)."""
        if self._num_layers is not None:
            return self._num_layers

        # 1. Try metadata fields
        for key in self._LAYER_KEYS:
            if key in self._reader.fields:
                val = self._get_field_value(self._reader.fields[key])
                if isinstance(val, int) and val > 0:
                    self._num_layers = val
                    return self._num_layers

        # 2. Fallback: count from tensor names
        layers: set[int] = set()
        for name in self._tensors:
            m = re.match(r"blk\.(\d+)\.", name)
            if m:
                layers.add(int(m.group(1)))
        if layers:
            self._num_layers = max(layers) + 1
            return self._num_layers

        self._num_layers = 28  # sensible default
        return self._num_layers

    @property
    def num_experts(self) -> int:
        """Number of routed MoE experts (auto-detected)."""
        if self._num_experts is not None:
            return self._num_experts

        # 1. Try metadata
        for key in self._EXPERT_KEYS:
            if key in self._reader.fields:
                val = self._get_field_value(self._reader.fields[key])
                if isinstance(val, int) and val > 0:
                    self._num_experts = val
                    return self._num_experts

        # 2. Infer from first expert tensor's shape
        for name, t in self._tensors.items():
            if "exps" in name or "experts" in name:
                self._num_experts = t.data.shape[0]
                return self._num_experts

        self._num_experts = 2
        return self._num_experts

    @property
    def architecture(self) -> str:
        """Model architecture string (e.g. 'qwen2moe', 'deepseek2')."""
        if self._arch is not None:
            return self._arch
        field = self._reader.fields.get("general.architecture")
        if field is not None:
            val = self._get_field_value(field)
            if isinstance(val, str):
                self._arch = val
            else:
                self._arch = str(val)
        else:
            self._arch = "unknown"
        return self._arch

    def _first_expert_layer(self) -> int:
        """Find the first layer index that has expert tensors.

        Some models (e.g. DeepSeek-V2 with leading_dense_block_count > 0)
        have dense initial layers without ``_exps`` tensors.
        """
        for name in self._tensors:
            if "ffn_gate_exps" in name:
                m = re.match(r"blk\.(\d+)\.", name)
                if m:
                    return int(m.group(1))
        raise KeyError(
            "No expert tensors found in this model "
            "(no tensor matching 'blk.N.ffn_gate_exps.weight')"
        )

    def per_expert_size(self, layer: int | None = None) -> int:
        """Total raw bytes for one expert (gate+up+down combined).

        Args:
            layer: Layer index to measure. Defaults to ``None``, which
                   auto-discovers the first layer with expert tensors.

        Returns:
            Total bytes per expert for this model.
        """
        if layer is None:
            layer = self._first_expert_layer()
        total = 0
        for pattern in self.EXPERT_PATTERNS:
            t = self._get_tensor(layer, pattern)
            # bytes per expert = total tensor bytes / num_experts
            total += t.n_bytes // t.data.shape[0]
        return total

    def read_expert_weights(self, layer: int, expert_id: int) -> bytes:
        """Read combined gate+up+down weights for one expert.

        Returns raw quantized bytes concatenated in order:
            gate_weights || up_weights || down_weights

        The returned length is always per_expert_size(layer) bytes.
        """
        result = bytearray()
        for pattern in self.EXPERT_PATTERNS:
            t = self._get_tensor(layer, pattern)
            # gguf-python arranges data as (n_experts, ...) — first dim = expert
            result.extend(t.data[expert_id].tobytes())
        return bytes(result)

    def expert_tensor_info(self, layer: int) -> list[dict]:
        """Debug info about expert tensors for a given layer."""
        info = []
        for pattern in self.EXPERT_PATTERNS:
            t = self._get_tensor(layer, pattern)
            info.append({
                "name": t.name,
                "logical_shape": [int(d) for d in t.shape],
                "data_shape": tuple(t.data.shape),
                "data_dtype": str(t.data.dtype),
                "total_bytes": t.n_bytes,
                "bytes_per_expert": t.n_bytes // t.data.shape[0],
                "ggml_type": t.tensor_type,
            })
        return info

    def close(self) -> None:
        """Close the GGUF reader (releases memmap)."""
        self._reader = None
        self._tensors.clear()

    # ── Internal ──────────────────────────────────────────────────

    def _build_index(self) -> None:
        """Build {tensor_name: tensor_info} lookup."""
        for t in self._reader.tensors:
            self._tensors[t.name] = t

    def _get_tensor(self, layer: int, pattern: str):
        """Find expert tensor by layer and name pattern."""
        name = f"blk.{layer}.{pattern}"
        if name not in self._tensors:
            # Show available tensors for this layer
            nearby = [
                k for k in self._tensors
                if k.startswith(f"blk.{layer}.")
            ]
            raise KeyError(
                f"Tensor not found: {name!r}. "
                f"Available for layer {layer} "
                f"({len(nearby)} tensors): {nearby[:10]}..."
            )
        return self._tensors[name]
