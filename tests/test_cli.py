"""Tests for moe_l2.cli — pure helper functions only.

No subprocess / network / GPU code is exercised here; those paths are
covered by end-to-end verification (see references/ test reports).
"""

import pytest

from moe_l2 import cli

# ── _parse_l2_size ────────────────────────────────────────────────

class TestParseL2Size:
    def test_gb(self):
        assert cli._parse_l2_size("4GB", expert_size=1024) == 4 * 1024 * 1024

    def test_mb(self):
        assert cli._parse_l2_size("512MB", expert_size=1024 * 1024) == 512

    def test_lowercase(self):
        assert cli._parse_l2_size("2gb", expert_size=1024) == 2 * 1024 * 1024

    def test_whitespace(self):
        assert cli._parse_l2_size("  1GB  ", expert_size=1024) == 1024 * 1024

    def test_fractional(self):
        assert cli._parse_l2_size("1.5GB", expert_size=1024) == int(1.5 * 1024 * 1024)

    def test_min_one_slot(self):
        # A size smaller than one expert still reserves one slot
        assert cli._parse_l2_size("1MB", expert_size=10 * 1024 * 1024) == 1
        assert cli._parse_l2_size("0.5MB", expert_size=1024 * 1024) == 1

    def test_invalid(self):
        with pytest.raises(ValueError):
            cli._parse_l2_size("abc", expert_size=1024)
        with pytest.raises(ValueError):
            cli._parse_l2_size("", expert_size=1024)


# ── _hf_url ───────────────────────────────────────────────────────

class TestHfUrl:
    def test_basic(self):
        assert (
            cli._hf_url("unsloth/Qwen3.6-35B-A3B-GGUF", "model.gguf")
            == "https://hf-mirror.com/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/model.gguf"
        )


# ── _find_gguf ────────────────────────────────────────────────────

class TestFindGguf:
    def test_existing_hint(self, tmp_path):
        f = tmp_path / "m.gguf"
        f.write_bytes(b"x")
        assert cli._find_gguf(str(f)) == str(f)

    def test_missing_hint_returns_none_or_path(self):
        # Falls back to /opt/data/models/*.gguf — may or may not exist.
        result = cli._find_gguf("/nonexistent/model.gguf")
        assert result is None or result.endswith(".gguf")


# ── doctor checks ─────────────────────────────────────────────────

class TestDoctorChecks:
    def test_check_python(self):
        ok, msg = cli._check_python()
        assert isinstance(ok, bool)
        assert isinstance(msg, str)

    def test_check_nvidia(self):
        ok, msg = cli._check_nvidia()
        assert isinstance(ok, bool)
        assert isinstance(msg, str)

    def test_check_cuda_lib(self):
        ok, msg = cli._check_cuda_lib()
        assert isinstance(ok, bool)
        assert isinstance(msg, str)

    def test_check_disk(self, tmp_path):
        ok, msg = cli._check_disk(tmp_path)
        assert isinstance(ok, bool)
        assert isinstance(msg, str)

    def test_check_dynamic_libs(self):
        ok, msg = cli._check_dynamic_libs()
        assert isinstance(ok, bool)
        assert isinstance(msg, str)
