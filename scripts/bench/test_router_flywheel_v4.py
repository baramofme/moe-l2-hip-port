#!/usr/bin/env python3
"""用真实 V4 trace 验证飞轮：喂 311k 行 EXPERT → 生成真实路由表 JSON"""
import sys, os, json, tempfile
sys.path.insert(0, '/opt/data/moe-l2')

from moe_l2.domain_router_flywheel import DomainRouterFlywheel

TRACE = '/opt/data/moe-l2/测试数据备份/v4-actset-trace-20260808.log'
tmp = tempfile.mkdtemp()
map_path = os.path.join(tmp, 'domain_router_map_flywheel.json')

fw = DomainRouterFlywheel(
    map_path=map_path,
    top_k=75,
    rebuild_every=100000,   # 避免中途频繁重建，最后手动触发
    min_records=500,
)

# 简化：全部归为 general（真实部署中由 on_request 分领域）
fw.set_domain('general')

n = 0
with open(TRACE, errors='replace') as f:
    for line in f:
        if line.startswith('EXPERT|'):
            fw.on_expert_line(line)
            n += 1
            if n % 100000 == 0:
                print(f'  已喂 {n:,} 行...')

print(f'共喂 {n:,} 行 EXPERT')
fw.maybe_rebuild(force=True)
print('stats:', fw.stats())

with open(map_path) as f:
    data = json.load(f)
print('\n== 生成的路由表 ==')
print('domains:', list(data['domains'].keys()))
print('层数:', len(data['domains']['general']['per_layer_domain_preferred']))
print('L0 top-10:', data['domains']['general']['per_layer_domain_preferred']['0'][:10])
print('L20 top-10:', data['domains']['general']['per_layer_domain_preferred']['20'][:10])
print('L42 top-10:', data['domains']['general']['per_layer_domain_preferred']['42'][:10])
