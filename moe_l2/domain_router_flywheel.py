"""路由表数据飞轮（2026-08-09 新增，不改动现有源码）。

原理：gate 已经在实时解析 EXPERT 路由日志（每 token 每层的 expert id）。
把这些真实路由按领域聚合，攒够阈值后自动重建领域路由表
（domain_router_map_*.json 同格式：domains.<domain>.per_layer_domain_preferred），
让 pretouch / 批量 pin 消费的路由表"越用越准"，而不是生成一次就固定。

设计：
- 新增独立模块，不修改 gate.py / proxy.py / cache.py 的既有逻辑
- 由调用方在已有 hook 点（gate.on_log_line / proxy 请求级）追加一行调用
- 聚合计数放内存（Counter），攒够 N 条 EXPERT 记录自动写盘（原子替换）
- 路由表格式与 load_mapping() 消费的 domain_expert_map.json 完全一致，
  所以 pretouch / cache.preload_domain / 批量 pin 无需改动即可消费新表
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger("moe-l2-router-flywheel")

# 默认输出：moe_l2/data/domain_router_map_flywheel_{model_id}.json
# [2026-08-13 flywheel B] flywheel 表按模型分文件：每个模型一张动态表，
# 各模型热度/收敛数据互不干扰（医院=模型，各医院独立收敛）。
# 旧单文件 domain_router_map_flywheel.json 退役（不再读写，留作 legacy）。
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "moe_l2" / "data"
DEFAULT_MAP_PATH = DATA_DIR / "domain_router_map_flywheel.json"  # legacy fallback（无 model_id 时）

# 每层每领域保留的高频专家数（与验证报告 top-75 对齐；Qwen 用 top-100 也兼容）
DEFAULT_TOP_K = 75
# 触发重建路由表的 EXPERT 记录阈值（个人使用约 1-2 天）
REBUILD_EVERY_N = 5000
# 至少需要多少条才开始重建（防止刚启动就用 3 条记录覆盖真实表）
MIN_RECORDS_FOR_REBUILD = 500


class DomainRouterFlywheel:
    """聚合真实门路由记录 → 按领域统计高频专家 → 原子更新路由表 JSON。

    线程安全：proxy 的 GateReaderThread 和请求线程都会调用，用锁保护。
    """

    def __init__(
        self,
        map_path: Path | None = None,
        model_id: str = "",
        top_k: int = DEFAULT_TOP_K,
        rebuild_every: int = REBUILD_EVERY_N,
        min_records: int = MIN_RECORDS_FOR_REBUILD,
    ) -> None:
        # [2026-08-13 flywheel B] 按模型分文件：map_path 未显式指定时，
        # 用 domain_router_map_flywheel_{model_id}.json；无 model_id 时回退 legacy 单文件。
        if map_path is None:
            if model_id:
                map_path = DATA_DIR / f"domain_router_map_flywheel_{model_id}.json"
            else:
                map_path = DEFAULT_MAP_PATH
        self.map_path = Path(map_path)
        self.top_k = top_k
        self.rebuild_every = rebuild_every
        self.min_records = min_records

        # 当前领域（由请求级信号设置，默认 general）
        self._domain = "general"
        # 聚合计数：domain -> layer -> Counter(expert_id -> 次数)
        self._agg: dict[str, dict[int, Counter]] = defaultdict(
            lambda: defaultdict(Counter))
        # 领域热度：[2026-08-13] domain -> 请求次数（热门领域依据）
        self._dom_freq: Counter = Counter()
        self._records = 0
        self._rebuilds = 0
        self._lock = __import__("threading").Lock()

    # ── 入口 ────────────────────────────────────────────────

    def set_domain(self, domain: str) -> None:
        """请求级：当前请求判定为哪个领域（proxy 判完 domain 后调用）。

        [2026-08-13] 同时累计领域请求次数（热门领域热度榜）。
        """
        if not domain:
            return
        with self._lock:
            self._domain = domain
            self._dom_freq[domain] += 1

    def hot_domains(self, n: int) -> list[str]:
        """[2026-08-13] 热度榜前 N 名领域（按请求次数降序）。"""
        with self._lock:
            return [d for d, _ in self._dom_freq.most_common(n)]

    def on_expert_line(self, line: str) -> None:
        """喂一条 EXPERT 路由日志行（gate.on_log_line 同源解析，格式一致）。

        EXPERT|L0|T2: [64,161,...] [130,...]
        """
        import re

        m = re.match(r"EXPERT\|L(\d+)\|T\d+:\s*(.*)$", line.strip())
        if not m:
            return
        layer = int(m.group(1))
        body = m.group(2)
        with self._lock:
            dom_counter = self._agg[self._domain][layer]
            for seg in re.findall(r"\[([0-9, ]+)\]", body):
                for x in seg.split(","):
                    x = x.strip()
                    if x:
                        dom_counter[int(x)] += 1
                        self._records += 1
            if self._records >= self.rebuild_every:
                self._rebuild_locked()

    def maybe_rebuild(self, force: bool = False) -> bool:
        """手动触发重建（或 force 强制）。返回是否重建了。"""
        with self._lock:
            if not force and self._records < self.min_records:
                logger.info(
                    "Router flywheel: %d records (< %d) — skip rebuild",
                    self._records, self.min_records)
                return False
            if force and self._records < self.min_records:
                logger.warning(
                    "Router flywheel: force rebuild with only %d records",
                    self._records)
            self._rebuild_locked()
            return True

    # ── 内部 ────────────────────────────────────────────────

    def _rebuild_locked(self) -> None:
        """用当前聚合计数重建路由表 JSON（原子替换）。调用方须持有锁。"""
        domains = {}
        for dom, layers in self._agg.items():
            per_layer = {}
            for layer in sorted(layers):
                counter = layers[layer]
                if not counter:
                    continue
                experts = [e for e, _ in counter.most_common(self.top_k)]
                per_layer[str(layer)] = experts
            if per_layer:
                domains[dom] = {"per_layer_domain_preferred": per_layer}

        if not domains:
            logger.warning("Router flywheel: no aggregated data, skip write")
            return

        out = {
            "description": (
                "Domain router map auto-built by flywheel from real gating "
                "traces ({} records). Rebuilt {} times. "
                "Per-domain per-layer top-{} experts.").format(
                    self._records, self._rebuilds, self.top_k),
            "source": "flywheel",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "top_k": self.top_k,
            "dom_freq": dict(self._dom_freq),  # [2026-08-13] 领域热度榜
            "domains": domains,
        }

        self.map_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.map_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.map_path)  # 原子替换
        self._rebuilds += 1
        self._records = 0
        # 注意：不清空 _agg —— 数据飞轮要跨 rebuild 累积，
        # 每次重建都是"全量历史 + 新增"的聚合，路由表越用越准。
        logger.info(
            "Router flywheel: rebuilt %s (%d domains, top-%d)",
            self.map_path, len(domains), self.top_k)

    # ── 状态 ────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            return {
                "router_flywheel_records": self._records,
                "router_flywheel_rebuilds": self._rebuilds,
                "router_flywheel_domains": sorted(self._agg.keys()),
                "router_flywheel_rebuild_every": self.rebuild_every,
                "router_flywheel_top_k": self.top_k,
            }
