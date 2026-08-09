"""Tests for moe_l2.training_flywheel — data flywheel (mode B).

append_sample / load_samples / _count_samples are pure file logic and
run anywhere. maybe_retrain degrades gracefully when scikit-learn is
missing (CI) — it must return False, never raise.
"""

import json

from moe_l2 import training_flywheel as fw


class TestAppendSample:
    def test_empty_prompt_skipped(self, tmp_path):
        path = tmp_path / "samples.jsonl"
        assert fw.append_sample("", "math", path) == 0
        assert not path.exists()

    def test_empty_domain_skipped(self, tmp_path):
        path = tmp_path / "samples.jsonl"
        assert fw.append_sample("hello", "", path) == 0
        assert not path.exists()

    def test_append_counts(self, tmp_path):
        path = tmp_path / "samples.jsonl"
        assert fw.append_sample("solve x", "math", path) == 1
        assert fw.append_sample("implement fn", "codegen", path) == 2
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"prompt": "solve x", "domain": "math"}

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "samples.jsonl"
        fw.append_sample("hi", "general_qa", path)
        assert path.exists()


class TestLoadSamples:
    def test_missing_file(self, tmp_path):
        assert fw.load_samples(tmp_path / "nope.jsonl") == []

    def test_roundtrip(self, tmp_path):
        path = tmp_path / "samples.jsonl"
        fw.append_sample("a", "math", path)
        fw.append_sample("b", "codegen", path)
        samples = fw.load_samples(path)
        assert len(samples) == 2
        assert samples[0]["domain"] == "math"

    def test_bad_lines_skipped(self, tmp_path):
        path = tmp_path / "samples.jsonl"
        path.write_text('{"prompt":"ok","domain":"math"}\nnot-json\n\n{"prompt":"","domain":"x"}\n', encoding="utf-8")
        samples = fw.load_samples(path)
        assert len(samples) == 1
        assert samples[0]["prompt"] == "ok"


class TestMaybeRetrain:
    def test_below_threshold(self, tmp_path):
        path = tmp_path / "samples.jsonl"
        fw.append_sample("a", "math", path)
        assert fw.maybe_retrain(path) is False

    def test_force_but_few_samples(self, tmp_path):
        path = tmp_path / "samples.jsonl"
        fw.append_sample("a", "math", path)
        assert fw.maybe_retrain(path, force=True) is False

    def test_retrain_attempt_without_sklearn(self, tmp_path):
        # 20 samples triggers the retrain path; without scikit-learn the
        # whole retrain is caught and reported as False — never raises.
        path = tmp_path / "samples.jsonl"
        for i in range(fw.MIN_SAMPLES_FOR_RETRAIN):
            fw.append_sample(f"prompt {i}", "math", path)
        result = fw.maybe_retrain(path)
        assert result is False
