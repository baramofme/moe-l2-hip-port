#!/usr/bin/env python3
"""生成 V4 分话题(领域)路由表 — domain_router_map_v4_topics.json
每个话题独立统计 top-K 高频专家 (分话题聚集性更强, top-50 即覆盖 89-94%)。
话题划分 (125 请求): general 1-50 | math 51-75 | code 76-100 | chat 101-125
"""
import re
import json
import os
from collections import Counter, defaultdict

TRACE_FILE = '/opt/data/moe-l2/测试数据备份/v4-actset-trace-20260808.log'
OUT_PATH = '/opt/data/moe-l2/moe_l2/data/domain_router_map_v4_topics.json'
TOP_K = 75  # 覆盖率 94-98%

TOPICS = [
    ('general', range(1, 51)),
    ('math',    range(51, 76)),
    ('code',    range(76, 101)),
    ('chat',    range(101, 126)),
]

LINE_RE = re.compile(r'EXPERT\|L(\d+)\|T\d+:\s*\[([^\]]+)\]')

def parse_requests(path):
    requests = []
    cur = None
    with open(path, errors='replace') as f:
        for line in f:
            if 'launch_slot_' in line and 'processing task' in line:
                if cur is not None:
                    requests.append(cur)
                cur = defaultdict(Counter)
                continue
            if cur is None:
                continue
            m = LINE_RE.match(line)
            if not m:
                continue
            layer = int(m.group(1))
            rest = line[m.end():]
            for block in re.finditer(r'\[([^\]]+)\]', rest):
                for x in block.group(1).split(','):
                    x = x.strip()
                    if x:
                        cur[layer][int(x)] += 1
        if cur is not None:
            requests.append(cur)
    return requests

def main():
    requests = parse_requests(TRACE_FILE)
    router_map = {
        "description": "V4 per-topic (per-domain) router map from real gating traces "
                       "(actset 20260808, 125 requests: general 50 + math/code/chat 25 each). "
                       "Per-topic per-layer top-{} experts. Coverage 94-98% of activations "
                       "(topic-internal locality > cross-topic aggregation).".format(TOP_K),
        "model": "DeepSeek-V4-Flash (UD-IQ2_M, 43 layers, 256 experts top-6)",
        "top_k": TOP_K,
        "domains": {}
    }

    for name, rng in TOPICS:
        merged = defaultdict(Counter)
        for idx in rng:
            if idx-1 < len(requests):
                for layer, counter in requests[idx-1].items():
                    merged[layer].update(counter)
        per_layer = {}
        for layer in sorted(merged):
            per_layer[str(layer)] = [e for e, _ in merged[layer].most_common(TOP_K)]
        router_map["domains"][name] = {"per_layer_domain_preferred": per_layer}

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(router_map, f, indent=2)

    print(f'生成: {OUT_PATH}')
    print(f'话题数: {len(router_map["domains"])}, 每话题每层 top-{TOP_K}')
    for dom, d in router_map["domains"].items():
        layers = d["per_layer_domain_preferred"]
        print(f'  {dom}: {len(layers)} 层, L0 top-8 = {layers["0"][:8]}')

if __name__ == '__main__':
    main()
