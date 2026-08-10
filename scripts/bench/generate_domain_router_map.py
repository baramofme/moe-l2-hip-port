#!/usr/bin/env python3
"""从 qwen8domains 真实门路由 trace 生成领域路由表 (domain_expert_map.json 兼容格式)。
输出: 每领域每层的 top-K 高频专家列表, 用于"判领域时提前 pin 高频专家走快速通道"。
"""
import re
import json
import os
from collections import Counter, defaultdict

TRACE_DIR = '/opt/data/moe-l2/测试数据备份/qwen8domains'
OUT_PATH = '/opt/data/moe-l2/moe_l2/data/domain_router_map_qwen.json'
TOP_K = 100  # 覆盖率 ~91-97%

LINE_RE = re.compile(r'EXPERT\|L(\d+)\|T\d+:\s*\[([^\]]+)\]')

def parse_file(path):
    layers = defaultdict(Counter)
    with open(path) as f:
        for line in f:
            m = LINE_RE.match(line)
            if not m:
                continue
            layer = int(m.group(1))
            rest = line[m.end():]
            for block in re.finditer(r'\[([^\]]+)\]', rest):
                for x in block.group(1).split(','):
                    x = x.strip()
                    if x:
                        layers[layer][int(x)] += 1
    return layers

def main():
    files = sorted(os.listdir(TRACE_DIR))
    domains = defaultdict(list)
    for f in files:
        if not f.startswith('expert_') or not f.endswith('.log'):
            continue
        dom = f.replace('expert_', '').replace('.log', '').rsplit('_', 1)[0]
        domains[dom].append(os.path.join(TRACE_DIR, f))

    router_map = {
        "description": "Domain router map generated from real gating traces (qwen8domains, 24 logs). "
                       "Per-layer top-{} high-frequency experts per domain. Coverage ~91-97% of activations. "
                       "Usage: detect domain -> pin these experts ahead -> DMA fast path.".format(TOP_K),
        "model": "Qwen3.6-35B-A3B (256 experts, top-8)",
        "top_k": TOP_K,
        "domains": {}
    }

    for dom in sorted(domains):
        merged = defaultdict(Counter)
        for f in domains[dom]:
            layers = parse_file(f)
            for layer, counter in layers.items():
                merged[layer].update(counter)
        per_layer = {}
        for layer in sorted(merged):
            experts = [e for e, _ in merged[layer].most_common(TOP_K)]
            per_layer[str(layer)] = experts
        router_map["domains"][dom] = {"per_layer_domain_preferred": per_layer}

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(router_map, f, indent=2)

    # 汇总信息
    print(f'生成: {OUT_PATH}')
    print(f'领域数: {len(router_map["domains"])}, 每层 top-{TOP_K}')
    for dom, d in router_map["domains"].items():
        layers = d["per_layer_domain_preferred"]
        print(f'  {dom}: {len(layers)} 层, 样例 L0 top-5 = {layers["0"][:5]}')

if __name__ == '__main__':
    main()
