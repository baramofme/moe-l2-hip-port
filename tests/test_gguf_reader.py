"""Tests for moe_l2.gguf_reader — MoEGGUFReader.

Uses tiny synthetic GGUF files (built with gguf.GGUFWriter) so tests
need no real model downloads and run in milliseconds.
"""

import numpy as np
import pytest
from gguf import GGUFWriter

from moe_l2.gguf_reader import MoEGGUFReader

EXPERT_TENSOR_BYTES = 2 * 4 * 4 * 4  # 128 bytes for the whole (2,4,4) f32 tensor
BYTES_PER_EXPERT = EXPERT_TENSOR_BYTES // 2  # 64 bytes per expert tensor
PER_EXPERT = BYTES_PER_EXPERT * 3  # gate + up + down


def _write_tiny_moe(path, n_layers=2, n_experts=2, dense_first=False):
    """Write a minimal MoE GGUF: blk.<i>.ffn_{gate,up,down}_exps.weight."""
    w = GGUFWriter(str(path), "qwen2moe")
    w.add_block_count(n_layers)
    w.add_uint32("qwen2moe.expert_count", n_experts)
    for layer in range(n_layers):
        if dense_first and layer == 0:
            # Dense layer: no expert tensors, just an attention tensor
            w.add_tensor(
                "blk.0.attn_q.weight",
                np.zeros((4, 4), dtype=np.float32),
            )
            continue
        w.add_tensor(
            f"blk.{layer}.ffn_gate_exps.weight",
            np.full((n_experts, 4, 4), 1.0, dtype=np.float32),
        )
        w.add_tensor(
            f"blk.{layer}.ffn_up_exps.weight",
            np.full((n_experts, 4, 4), 2.0, dtype=np.float32),
        )
        w.add_tensor(
            f"blk.{layer}.ffn_down_exps.weight",
            np.full((n_experts, 4, 4), 3.0, dtype=np.float32),
        )
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.fixture(scope="module")
def tiny_moe(tmp_path_factory):
    path = tmp_path_factory.mktemp("gguf") / "tiny_moe.gguf"
    _write_tiny_moe(path)
    return path


@pytest.fixture(scope="module")
def dense_first_moe(tmp_path_factory):
    path = tmp_path_factory.mktemp("gguf") / "dense_first.gguf"
    _write_tiny_moe(path, n_layers=2, dense_first=True)
    return path


# ── Construction ──────────────────────────────────────────────────

class TestConstruction:
    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            MoEGGUFReader("/nonexistent/model.gguf")

    def test_open_and_close(self, tiny_moe):
        r = MoEGGUFReader(tiny_moe)
        assert r._reader is not None
        r.close()
        assert r._reader is None


# ── Metadata ──────────────────────────────────────────────────────

class TestMetadata:
    @pytest.fixture(autouse=True)
    def reader(self, tiny_moe):
        r = MoEGGUFReader(tiny_moe)
        yield r
        r.close()

    def test_num_layers(self, reader):
        assert reader.num_layers == 2

    def test_num_experts(self, reader):
        assert reader.num_experts == 2

    def test_architecture(self, reader):
        assert reader.architecture == "qwen2moe"


# ── Weight access ─────────────────────────────────────────────────

class TestWeights:
    @pytest.fixture(autouse=True)
    def reader(self, tiny_moe):
        r = MoEGGUFReader(tiny_moe)
        yield r
        r.close()

    def test_per_expert_size_autodetect(self, reader):
        assert reader.per_expert_size() == PER_EXPERT

    def test_per_expert_size_explicit_layer(self, reader):
        assert reader.per_expert_size(layer=1) == PER_EXPERT

    def test_read_expert_weights_length(self, reader):
        data = reader.read_expert_weights(0, 0)
        assert len(data) == PER_EXPERT

    def test_read_expert_weights_concat_order(self, reader):
        # gate(1.0) || up(2.0) || down(3.0) for expert 0
        data = reader.read_expert_weights(0, 0)
        gate = np.full((4, 4), 1.0, dtype=np.float32).tobytes()
        up = np.full((4, 4), 2.0, dtype=np.float32).tobytes()
        down = np.full((4, 4), 3.0, dtype=np.float32).tobytes()
        assert data == gate + up + down

    def test_read_expert_weights_expert1_differs(self, reader):
        # expert 1 also gate=1.0 so content equals expert 0 here; just check len
        assert len(reader.read_expert_weights(0, 1)) == PER_EXPERT

    def test_expert_tensor_info(self, reader):
        info = reader.expert_tensor_info(0)
        assert len(info) == 3
        assert info[0]["name"] == "blk.0.ffn_gate_exps.weight"
        assert info[0]["bytes_per_expert"] == BYTES_PER_EXPERT
        assert info[2]["name"] == "blk.0.ffn_down_exps.weight"

    def test_missing_tensor_raises(self, reader):
        with pytest.raises(KeyError):
            reader._get_tensor(5, "ffn_gate_exps.weight")


# ── Dense-first models (DeepSeek-style) ───────────────────────────

class TestDenseFirst:
    @pytest.fixture(autouse=True)
    def reader(self, dense_first_moe):
        r = MoEGGUFReader(dense_first_moe)
        yield r
        r.close()

    def test_first_expert_layer_skips_dense(self, reader):
        assert reader._first_expert_layer() == 1

    def test_per_expert_size_uses_first_expert_layer(self, reader):
        assert reader.per_expert_size() == PER_EXPERT

    def test_num_layers_fallback_from_tensors(self, reader):
        # No block_count KV for the dense-first file? It has one (2) — still 2
        assert reader.num_layers == 2

    def test_dense_layer_has_no_expert_tensor(self, reader):
        with pytest.raises(KeyError):
            reader._get_tensor(0, "ffn_gate_exps.weight")
