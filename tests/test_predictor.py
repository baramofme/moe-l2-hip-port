"""Tests for moe_l2.predictor — keyword domain prediction + mapping access.

These tests run without any optional dependencies (sentence-transformers,
scikit-learn). The TF-IDF / semantic tiers are skipped when unavailable.
"""

import pytest

from moe_l2 import predictor
from moe_l2.predictor import (
    DOMAINS,
    domain_to_expert_ids,
    get_backbone_experts,
    get_layer_specificity,
    get_preload_set,
    load_mapping,
    predict,
    predict_hybrid,
)

# ── Keyword matching (predict) ───────────────────────────────────

class TestPredictKeyword:
    """One positive hit per domain plus boundary cases."""

    @pytest.mark.parametrize(
        "prompt,expected",
        [
            # codegen
            ("implement a sorting algorithm", "codegen"),
            ("write a function that reverses a list", "codegen"),
            ("print hello world", "codegen"),
            # debug
            ("my program crashes with a traceback", "debug"),
            ("fix this keyerror please", "debug"),
            ("check the log for errors", "debug"),
            # math
            ("calculate the derivative of x^2", "math"),
            ("solve for x in this equation", "math"),
            # logic
            ("solve this logic puzzle", "logic"),
            ("logical reasoning exercise", "logic"),
            # general_qa
            ("what is the capital of france", "general_qa"),
            ("explain how photosynthesis works", "general_qa"),
            # chinese_tech
            ("怎么部署 docker 到 nas", "chinese_tech"),
            ("这个模型的原理是什么", "chinese_tech"),
            # creative_write
            ("write a story about a dragon", "creative_write"),
            ("帮我写一篇小说", "creative_write"),
            # translate
            ("translate this to english", "translate"),
            ("翻译这段话", "translate"),
        ],
    )
    def test_positive_hits(self, prompt, expected):
        assert predict(prompt) == expected

    def test_fallback_no_keyword(self):
        assert predict("hello world") == "general_qa"

    def test_custom_fallback(self):
        assert predict("zzz qqq", fallback="unknown") == "unknown"

    def test_case_insensitive(self):
        assert predict("IMPLEMENT A SORTING ALGORITHM") == "codegen"

    def test_cjk_substring_match(self):
        # CJK keywords use plain substring matching
        assert predict("我想了解大模型原理") == "chinese_tech"

    def test_word_boundary_no_false_positive(self):
        # "log " must not match inside "logic" / "logical"
        assert predict("this is a logic problem") == "general_qa"
        assert predict("logical reasoning") == "logic"
        assert predict("check the logic") == "general_qa"

    def test_longer_keyword_wins(self):
        # "write a story" is longer than "story" — both map to creative_write
        assert predict("write a story") == "creative_write"
        # "logic puzzle" beats "puzzle"
        assert predict("a logic puzzle") == "logic"

    def test_all_domains_covered(self):
        # Every declared domain should be reachable by at least one keyword
        for domain in DOMAINS:
            assert domain in predictor._KEYWORD_MAP.values(), domain


# ── Hybrid prediction (keyword + optional tiers) ─────────────────

class TestPredictHybrid:
    def test_keyword_fast_path(self):
        assert predict_hybrid("implement a sorting algorithm") == "codegen"

    def test_fallback_when_no_tiers(self):
        # No TF-IDF / semantic available in CI → keyword result or fallback
        assert predict_hybrid("hello world") == "general_qa"

    def test_returns_str(self):
        result = predict_hybrid("什么是模型量化")
        assert isinstance(result, str)
        assert result in DOMAINS


# ── Mapping access (uses the packaged domain_expert_map.json) ─────

class TestMapping:
    def test_load_mapping_default(self):
        mapping = load_mapping()
        assert "domains" in mapping
        assert "codegen" in mapping["domains"]

    def test_load_mapping_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_mapping(tmp_path / "nope.json")

    def test_domain_to_expert_ids_real_data(self):
        ids = domain_to_expert_ids("codegen", 0)
        assert isinstance(ids, list)
        assert all(isinstance(e, int) for e in ids)

    def test_domain_to_expert_ids_unknown_domain(self):
        assert domain_to_expert_ids("no_such_domain", 0) == []

    def test_get_preload_set_real_data(self):
        result = get_preload_set("math")
        assert isinstance(result, dict)
        for layer, experts in result.items():
            assert isinstance(layer, int)
            assert isinstance(experts, list)

    def test_get_preload_set_unknown_domain(self):
        assert get_preload_set("no_such_domain") == {}

    def test_get_backbone_experts(self):
        backbone = get_backbone_experts()
        assert isinstance(backbone, list)

    def test_get_layer_specificity(self):
        info = get_layer_specificity(0)
        assert info is None or isinstance(info, dict)
