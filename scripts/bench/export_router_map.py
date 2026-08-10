#!/usr/bin/env python3
"""从路由表 JSON 生成选择性 pin 用的 router map 文件（C++ MOE_L2_ROUTER_FILE 格式）。

格式：每行 "layer expert1 expert2 ..."（层号 + 该层应 pin 的专家 id 列表）
来源：优先 flywheel 表，回退分话题表（聚合并集），再回退聚合表。

用法：
    python3 -m scripts.bench.export_router_map --top-k 100 --output /tmp/router.map
"""
import argparse
import json
import os


def load_tables(data_dir: str):
    """返回 (flywheel_path_or_none, topics_path_or_none, agg_path_or_none)"""
    flywheel = os.path.join(data_dir, "domain_router_map_flywheel.json")
    topics = os.path.join(data_dir, "domain_router_map_v4_topics.json")
    agg = os.path.join(data_dir, "domain_router_map_v4.json")
    return (
        flywheel if os.path.exists(flywheel) else None,
        topics if os.path.exists(topics) else None,
        agg if os.path.exists(agg) else None,
    )


def parse_table(path: str):
    """解析路由表 JSON，返回 {layer: [experts]}。兼容 flywheel / topics / agg 三种格式。"""
    with open(path) as f:
        d = json.load(f)
    layers = {}
    # 格式 A: {"domains": {domain: {"per_layer_domain_preferred": {"0": [...], ...}}}}
    # 格式 B: {"per_layer_domain_preferred": {"0": [...]}} (flywheel 单领域)
    # 格式 C: {"layers": {"0": [...]}} (agg)
    def collect_plist(obj, out):
        if isinstance(obj, dict):
            # 找 per_layer_domain_preferred 或 layers
            for key in ("per_layer_domain_preferred", "layers", "per_layer"):
                if key in obj and isinstance(obj[key], dict):
                    for lk, experts in obj[key].items():
                        out[int(lk)] = experts
                    return True
            # 递归找 domains
            if "domains" in obj and isinstance(obj["domains"], dict):
                return collect_plist(obj["domains"], out)
        return False

    if not collect_plist(d, layers):
        # 兜底：直接找所有 int-keyed dict
        for k, v in d.items():
            if isinstance(v, dict):
                collect_plist(v, layers)
    return layers


def union_layers(table_sets):
    """多表（按层）取并集：{layer: set(experts)}"""
    union = {}
    for table in table_sets:
        for layer, experts in table.items():
            union.setdefault(layer, set()).update(experts)
    return union


def top_k_of(union: dict, top_k: int, freq: dict = None):
    """并集后截断到 top-K：{layer: sorted list}。

    freq: {layer: Counter(expert -> count)} 可选；提供时按频率保留高频 top-K，
    否则按专家 id 排序截断（仅作兜底）。V4 场景并集通常 < top_k，一般不会触发。
    """
    out = {}
    for layer, experts in union.items():
        if freq and layer in freq:
            # 并集内按频率降序，取 top-K
            cnt = freq[layer]
            ranked = sorted(experts, key=lambda e: -cnt.get(e, 0))[:top_k]
        else:
            ranked = sorted(experts)[:top_k]
        out[layer] = ranked
    return out


def write_router_map(union: dict, path: str):
    with open(path, "w") as f:
        for layer in sorted(union.keys()):
            experts = sorted(union[layer])
            f.write(f"{layer} " + " ".join(str(e) for e in experts) + "\n")
    print(f"[export_router_map] wrote {len(union)} layers -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="moe_l2/data")
    ap.add_argument("--top-k", type=int, default=100, help="并集后每层保留专家数上限")
    ap.add_argument("--output", default="/tmp/router.map")
    args = ap.parse_args()

    flywheel, topics, agg = load_tables(args.data_dir)
    sources = []
    if flywheel:
        sources.append(("flywheel", flywheel))
    if topics:
        sources.append(("topics", topics))
    if agg:
        sources.append(("agg", agg))
    if not sources:
        print("[export_router_map] ERROR: no router table found in", args.data_dir)
        return 1

    tables = []
    for name, path in sources:
        try:
            tables.append(parse_table(path))
            print(f"[export_router_map] parsed {name}: {path}")
        except Exception as e:
            print(f"[export_router_map] skip {name}: {e}")

    union = union_layers(tables)
    print(f"[export_router_map] union: {len(union)} layers, "
          f"avg experts/layer = {sum(len(v) for v in union.values())/max(1, len(union)):.0f}")

    if args.top_k and args.top_k > 0:
        union = top_k_of(union, args.top_k)

    write_router_map(union, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
