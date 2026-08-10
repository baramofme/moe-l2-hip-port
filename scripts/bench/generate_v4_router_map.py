#!/usr/bin/env python3
"""从 V4 actset trace 生成 V4 路由表 (domain_router_map_v4.json)。
V4: DeepSeek-V4-Flash, 43层, 256专家 top-6。
输入: actset_test/server.log (311k EXPERT 行, 4 组话题: 通用50轮 + math/code/chat 各25轮)
"""
import re
import json
import os
from collections import Counter, defaultdict

TRACE_FILE = '/opt/data/moe-l2/测试数据备份/v4-actset-trace-20260808.log'
OUT_PATH = '/opt/data/moe-l2/moe_l2/data/domain_router_map_v4.json'
TOP_K = 75  # 覆盖率 94%

LINE_RE = re.compile(r'EXPERT\|L(\d+)\|T\d+:\s*\[([^\]]+)\]')

def parse_all(path):
    layers = defaultdict(Counter)
    with open(path, errors='replace') as f:
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
    layers = parse_all(TRACE_FILE)
    n_layers = len(layers)
    total_acts = sum(sum(c.values()) for c in layers.values())

    router_map = {
        "description": "Domain router map generated from real gating traces "
                       "(V4 actset experiment 20260808, 311k EXPERT lines, "
                       "4 topic groups: general 50 rounds + math/code/chat 25 rounds each). "
                       "Per-layer top-{} high-frequency experts. Coverage ~94% of activations.".format(TOP_K),
        "model": "DeepSeek-V4-Flash (UD-IQ2_M, 43 layers, 256 experts top-6)",
        "top_k": TOP_K,
        "n_layers": n_layers,
        "total_activations": total_acts,
        "layers": {}
    }

    for layer in sorted(layers):
        experts = [e for e, _ in layers[layer].most_common(TOP_K)]
        router_map["layers"][str(layer)] = experts

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(router_map, f, indent=2)

    print(f'生成: {OUT_PATH}')
    print(f'模型: V4 | 层数: {n_layers} | 总激活: {total_acts:,} | top-{TOP_K}/层')
    print(f'样例 L0 top-10: {router_map["layers"]["0"][:10]}')
    print(f'样例 L20 top-10: {router_map["layers"]["20"][:10]}')
    print(f'样例 L42 top-10: {router_map["layers"]["42"][:10]}')

if __name__ == '__main__':
    main()
