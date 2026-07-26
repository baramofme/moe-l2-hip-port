"""
Domain predictor — L0a in the moe-l2 architecture.

Phase 2: keyword matching (simple, fast, zero-dependency).
Phase 3: upgrade to lightweight classifier.

The predictor receives the user's prompt and outputs a domain label,
which the L2 cache manager uses to decide which experts to preload.
"""

# Default domain list — matches the 8 tested domains
DOMAINS = [
    "codegen",
    "debug",
    "math",
    "logic",
    "general_qa",
    "chinese_tech",
    "creative_write",
    "translate",
]

# Keyword → domain mapping (Phase 2 baseline)
# Expand this based on real usage data.
_KEYWORD_MAP: dict[str, str] = {
    # codegen
    "write a function": "codegen",
    "implement": "codegen",
    "def ": "codegen",
    "class ": "codegen",
    "server": "codegen",
    "api": "codegen",
    "script": "codegen",
    # debug
    "error": "debug",
    "bug": "debug",
    "crash": "debug",
    "fix": "debug",
    "traceback": "debug",
    "exception": "debug",
    "not working": "debug",
    # math
    "calculate": "math",
    "equation": "math",
    "derivative": "math",
    "integral": "math",
    "probability": "math",
    "matrix": "math",
    "solve for": "math",
    # logic
    "puzzle": "logic",
    "reasoning": "logic",
    "if and only if": "logic",
    "constraint": "logic",
    "deduction": "logic",
    # general_qa
    "what is": "general_qa",
    "explain": "general_qa",
    "tell me about": "general_qa",
    "definition": "general_qa",
    # chinese_tech
    "是什么": "chinese_tech",
    "怎么": "chinese_tech",
    "原理": "chinese_tech",
    "教程": "chinese_tech",
    "配置": "chinese_tech",
    "部署": "chinese_tech",
    # creative_write
    "story": "creative_write",
    "poem": "creative_write",
    "write a story": "creative_write",
    "narrative": "creative_write",
    # translate
    "translate": "translate",
    "translation": "translate",
    "翻译": "translate",
}


def predict(prompt: str, fallback: str = "general_qa") -> str:
    """Predict domain from prompt text using keyword matching.

    Args:
        prompt: User input text.
        fallback: Domain label when no keywords match.

    Returns:
        Predicted domain label.
    """
    prompt_lower = prompt.lower()
    for keyword, domain in _KEYWORD_MAP.items():
        if keyword in prompt_lower:
            return domain
    return fallback


def domain_to_expert_ids(domain: str, layer: int) -> list[int]:
    """Return expert IDs for a given domain and layer.

    Phase 2: stub — returns empty list. Will be populated from
    the domain→expert mapping table (GGUF metadata or .json file)
    derived from the LLAMA_EXPERT_LOG=1 data collected in Phase 1.
    """
    # TODO: load from domain_expert_map.json (generated from expert log data)
    return []
