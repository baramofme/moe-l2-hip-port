"""门控在线自适应（P2 ④，2026-08-02）。

推理中读 llama-server 的 LLAMA_EXPERT_LOG 实时路由输出，动态调整
L2 缓存优先级（LRU 智能增强），并按会话累积路由画像。

与分类器的分工（产品化方案定稿）：
- 分类器（L0a）管冷启动：prompt → 领域，预测该预载哪些专家
- 门控管热循环：推理中 → 真实路由信号 → 动态调缓存优先级

信号流：
    llama-server stderr (LLAMA_EXPERT_LOG=1)
        → EXPERT|L3|T4: [12,45] [3,77] ...
        → RoutingProfiler.on_log_line(line)  解析 + 更新画像
        → 窗口内漂移检测 → cache.promote_domain() / promote 高频专家
        → 缓存优先级自适应

用法：
    from moe_l2.gate import RoutingProfiler
    profiler = RoutingProfiler(cache, expert_map)
    for line in server_stderr_lines:
        profiler.on_log_line(line)      # 实时路由 → 调缓存
    profiler.on_request(domain)         # 请求级信号 → 检测漂移
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict, deque

logger = logging.getLogger("moe-l2-gate")

# EXPERT|L3|T4: [12,45] [3,77] ...
_EXPERT_LINE_RE = re.compile(r"EXPERT\|L(\d+)\|T(\d+): (.*)")

# 滑动窗口：保留最近 N 层-行数的激活记录做漂移检测
_WINDOW_LINES = 400


class RoutingProfiler:
    """会话路由画像 + 在线自适应门控。

    - 画像：每层专家激活频率（滑动窗口，反映"最近在用谁"）
    - 漂移检测：当前窗口与基线窗口的 Jaccard 差异 → 路由漂移信号
    - 动作：漂移 → promote 目标域；高频专家 → 自动抬优先级
    """

    def __init__(
        self,
        cache=None,
        expert_map: dict | None = None,
        drift_threshold: float = 0.35,
    ):
        self.cache = cache
        self.expert_map = expert_map or {}
        self.drift_threshold = drift_threshold

        # 窗口内激活记录（FIFO，用于漂移对比）
        self._window: deque[tuple[int, int]] = deque(maxlen=_WINDOW_LINES)

        # 会话画像：layer -> Counter(expert_id -> 激活次数)
        self._profile: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # 上次漂移检测时的窗口快照（对比基线）
        self._baseline: set[tuple[int, int]] = set()

        self._log_lines = 0
        self._last_drift: float = 0.0

    # ── 实时信号入口 ──────────────────────────────────────────

    def on_log_line(self, line: str) -> None:
        """解析一条 EXPERT 日志行并更新画像 / 触发动作。"""
        m = _EXPERT_LINE_RE.match(line.strip())
        if not m:
            return
        layer = int(m.group(1))
        body = m.group(3)
        self._log_lines += 1

        expert_ids: list[int] = []
        for seg in re.findall(r"\[([0-9, ]+)\]", body):
            for x in seg.split(","):
                x = x.strip()
                if x:
                    expert_ids.append(int(x))

        for eid in expert_ids:
            self._window.append((layer, eid))
            self._profile[layer][eid] += 1

        # 每 40 行做一次漂移检测（避免每行都算）
        if self._log_lines % 40 == 0:
            self._maybe_promote()
            self._detect_drift()

    def on_request(self, domain: str) -> None:
        """请求级信号：领域切换 → 主动预热目标域。"""
        if self.cache is None or self.expert_map is None:
            return
        try:
            self.cache.promote_domain(domain, self.expert_map)
            logger.info("Gate: request domain=%s → promote_domain", domain)
        except Exception as e:
            logger.warning("Gate promote failed: %s", e)

    # ── 画像查询 ─────────────────────────────────────────────

    def top_experts(self, layer: int, k: int = 8) -> list[int]:
        """当前画像中该层激活最频繁的专家。"""
        cnt = self._profile.get(layer, {})
        return [e for e, _ in sorted(cnt.items(), key=lambda kv: -kv[1])[:k]]

    def profile_size(self) -> int:
        return sum(len(v) for v in self._profile.values())

    # ── 内部：在线自适应 ─────────────────────────────────────

    def _maybe_promote(self) -> None:
        """高频专家自动抬优先级（不 pin，靠 LRU 保留）。"""
        if self.cache is None:
            return
        # 取画像中每层 top-4 高频专家，promote（抬 MRU）
        with self.cache._lock:
            for layer, cnt in self._profile.items():
                top = [e for e, _ in sorted(cnt.items(), key=lambda kv: -kv[1])[:4]]
                for eid in top:
                    if self.cache._is_cached(layer, eid):
                        self.cache._touch(layer, eid)

    def _detect_drift(self) -> None:
        """检测路由漂移：当前窗口 vs 基线窗口的 Jaccard 差异。"""
        current = set(self._window)
        if not self._baseline or not current:
            # 首个窗口：建立基线
            self._baseline = current
            return

        inter = len(current & self._baseline)
        union = len(current | self._baseline)
        jaccard = inter / union if union else 1.0
        drift = 1.0 - jaccard  # 1 = 完全不同（漂移大），0 = 完全相同

        self._last_drift = drift
        self._baseline = current

        if drift >= self.drift_threshold:
            logger.info("Gate: routing drift %.2f detected (threshold %.2f)", drift, self.drift_threshold)
            # 漂移 → 主动把当前高频专家抬优先级（相当于"换挡"）
            self._maybe_promote()

    @property
    def last_drift(self) -> float:
        return self._last_drift

    def stats(self) -> dict:
        return {
            "gate_log_lines": self._log_lines,
            "gate_profile_experts": self.profile_size(),
            "gate_last_drift": round(self._last_drift, 3),
            "gate_drift_threshold": self.drift_threshold,
        }
