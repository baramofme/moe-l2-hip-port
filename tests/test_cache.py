"""Tests for moe_l2.cache — L2Cache LRU / pin / preload / stats.

Uses a tiny expert_size and a tmp l2_dir so tests are fast and never
touch /dev/shm. All loads are async (thread pool) — tests serialise by
calling wait_for_pending() after each batch of requests.
"""

import pytest

from moe_l2.cache import DEFAULT_EXPERT_SIZE, L2Cache


@pytest.fixture
def cache(tmp_path):
    c = L2Cache(
        n_layers=2,
        slots_per_layer=2,
        expert_size=8,
        l2_dir=tmp_path / "l2",
    )
    yield c
    c.close()


def _wait(cache, timeout=10.0):
    """Wait for all pending loads and assert they completed."""
    done = cache.wait_for_pending(timeout=timeout)
    return done


# ── Request / hit / miss ─────────────────────────────────────────

class TestRequest:
    def test_miss_then_hit(self, cache):
        assert cache.request(0, 1) is False  # miss → async load
        _wait(cache)
        assert cache.contains(0, 1) is True
        assert cache.request(0, 1) is True  # now hot
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate_pct"] == 50.0

    def test_out_of_range_layer(self, cache):
        # Invalid layer: _is_cached returns False → miss, load fails silently
        assert cache.request(99, 1) is False
        _wait(cache)
        assert cache.contains(99, 1) is False

    def test_request_batch(self, cache):
        result = cache.request_batch([(0, 1), (0, 2), (1, 3)])
        assert result["miss_count"] == 3
        assert result["hit_count"] == 0
        _wait(cache)
        assert cache.contains(0, 1) and cache.contains(0, 2) and cache.contains(1, 3)

        result = cache.request_batch([(0, 1), (0, 9)])
        assert result["hit_count"] == 1
        assert result["miss_count"] == 1

    def test_request_batch_empty(self, cache):
        result = cache.request_batch([])
        assert result["hit_count"] == 0
        assert result["miss_count"] == 0


# ── LRU eviction ──────────────────────────────────────────────────

class TestEviction:
    def test_lru_evicts_oldest(self, cache):
        cache.request(0, 1)
        _wait(cache)
        cache.request(0, 2)
        _wait(cache)
        cache.request(0, 3)  # evicts LRU = 1
        _wait(cache)

        assert cache.contains(0, 1) is False
        assert cache.contains(0, 2) is True
        assert cache.contains(0, 3) is True

    def test_hit_refreshes_lru_position(self, cache):
        cache.request(0, 1)
        _wait(cache)
        cache.request(0, 2)
        _wait(cache)
        cache.request(0, 1)  # touch → 1 becomes MRU
        _wait(cache)
        cache.request(0, 3)  # evicts 2, not 1
        _wait(cache)

        assert cache.contains(0, 1) is True
        assert cache.contains(0, 2) is False

    def test_pinned_never_evicted(self, cache):
        cache.pin_experts(0, [1])
        cache.request(0, 1)
        _wait(cache)
        cache.request(0, 2)
        _wait(cache)
        cache.request(0, 3)  # only evictable unpinned = 2
        _wait(cache)

        assert cache.contains(0, 1) is True
        assert cache.contains(0, 2) is False
        assert cache.contains(0, 3) is True

    def test_unpin_eventually_allows_eviction(self, cache):
        cache.pin_experts(0, [1])
        cache.request(0, 1)
        _wait(cache)
        cache.request(0, 2)
        _wait(cache)
        cache.unpin_experts(0, [1])
        cache.request(0, 1)  # touch → re-enters LRU tracking
        cache.request(0, 3)
        _wait(cache)
        assert cache.contains(0, 2) is False  # oldest unpinned evicted first
        assert cache.contains(0, 1) is True
        cache.request(0, 4)
        _wait(cache)
        assert cache.contains(0, 1) is False  # now evictable after unpin


# ── Domain preload / pin lifecycle ────────────────────────────────

EXPERT_MAP = {
    "domains": {
        "codegen": {"per_layer_top": {"0": [1, 2]}},
        "math": {"per_layer_top": {"0": [3]}},
    }
}


class TestPreload:
    def test_preload_domain(self, cache):
        cache.preload_domain("codegen", EXPERT_MAP)
        _wait(cache)
        assert cache.contains(0, 1) and cache.contains(0, 2)
        assert cache.stats()["active_domain"] == "codegen"

    def test_preload_unknown_domain_noop(self, cache):
        cache.preload_domain("nope", EXPERT_MAP)  # must not raise
        assert cache.stats()["active_domain"] is None

    def test_preload_empty_map_noop(self, cache):
        cache.preload_domain("codegen", {})  # must not raise

    def test_domain_switch_releases_old_pins(self, cache):
        cache.preload_domain("codegen", EXPERT_MAP)
        _wait(cache)
        cache.preload_domain("math", EXPERT_MAP)
        _wait(cache)

        # Old domain pin released: requesting a new expert evicts 1/2, keeps 3
        assert cache.contains(0, 3) is True
        cache.request(0, 9)
        _wait(cache)
        assert cache.contains(0, 1) is False
        assert cache.contains(0, 3) is True

    def test_promote_domain_touches_cached(self, cache):
        cache.preload_domain("codegen", EXPERT_MAP)
        _wait(cache)
        cache.promote_domain("math", EXPERT_MAP)
        _wait(cache)
        assert cache.stats()["active_domain"] == "math"

    def test_preload_batch(self, cache):
        cache.preload_batch([(0, 5), (1, 6)])
        _wait(cache)
        assert cache.contains(0, 5) and cache.contains(1, 6)

    def test_clear_domain_pins(self, cache):
        cache.preload_domain("codegen", EXPERT_MAP)
        _wait(cache)
        cache.clear_domain_pins()
        assert cache.stats()["active_domain"] is None
        assert cache._domain_pinned[0] == set()


# ── Slots / paths / stats ─────────────────────────────────────────

class TestSlotsAndStats:
    def test_get_slot_path_valid(self, cache):
        assert cache.get_slot_path(0, 1) == cache.l2_dir / "L0_S1"

    def test_get_slot_path_invalid(self, cache):
        assert cache.get_slot_path(99, 0) is None
        assert cache.get_slot_path(0, 99) is None

    def test_get_expert_slot(self, cache):
        assert cache.get_expert_slot(0, 42) is None
        cache.request(0, 42)
        _wait(cache)
        assert cache.get_expert_slot(0, 42) is not None

    def test_stats_fields(self, cache):
        stats = cache.stats()
        for key in (
            "hits", "misses", "total_requests", "hit_rate_pct",
            "total_slots", "used_slots", "pinned_experts",
            "pending_loads", "loads_completed", "utilization_pct",
            "memory_usage_mb", "active_domain", "per_layer_lines",
        ):
            assert key in stats, key
        assert stats["total_slots"] == 4  # 2 layers × 2 slots

    def test_stats_memory_usage(self, cache):
        cache.request(0, 1)
        _wait(cache)
        stats = cache.stats()
        assert stats["memory_usage_mb"] == round(1 * 8 / (1024 * 1024), 1)
        assert stats["utilization_pct"] == 25.0

    def test_close_cleans_up(self, tmp_path):
        l2_dir = tmp_path / "l2b"
        c = L2Cache(n_layers=1, slots_per_layer=2, expert_size=8, l2_dir=l2_dir)
        c.request(0, 1)
        _wait(c)
        c.close()
        assert not l2_dir.exists()

    def test_context_manager(self, tmp_path):
        with L2Cache(
            n_layers=1, slots_per_layer=1, expert_size=8,
            l2_dir=tmp_path / "l2c",
        ) as c:
            c.request(0, 1)
            _wait(c)
            assert c.contains(0, 1)


# ── Weight sourcing ───────────────────────────────────────────────

class TestWeightSource:
    def test_placeholder_zeros(self, cache):
        data = cache._read_expert_weights(0, 7)
        assert data == b"\x00" * 8

    def test_bin_file_preferred(self, tmp_path):
        bins = tmp_path / "bins"
        bins.mkdir()
        (bins / "L0_E9.bin").write_bytes(b"ABCDEFGH")
        c = L2Cache(
            n_layers=1, slots_per_layer=2, expert_size=8,
            l2_dir=tmp_path / "l2d", expert_data_dir=bins,
        )
        try:
            assert c._read_expert_weights(0, 9) == b"ABCDEFGH"
        finally:
            c.close()

    def test_bin_file_padded(self, tmp_path):
        bins = tmp_path / "bins"
        bins.mkdir()
        (bins / "L0_E9.bin").write_bytes(b"AB")  # shorter than expert_size
        c = L2Cache(
            n_layers=1, slots_per_layer=2, expert_size=8,
            l2_dir=tmp_path / "l2e", expert_data_dir=bins,
        )
        try:
            assert c._read_expert_weights(0, 9) == b"AB" + b"\x00" * 6
        finally:
            c.close()

    def test_defaults(self):
        # Sanity: defaults match the documented architecture
        assert DEFAULT_EXPERT_SIZE == 1_010_000
