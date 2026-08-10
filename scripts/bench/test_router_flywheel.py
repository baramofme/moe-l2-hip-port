#!/usr/bin/env python3
"""测试 domain_router_flywheel 数据飞轮（不改源码，独立验证）"""
import sys, os, json, tempfile
sys.path.insert(0, '/opt/data/moe-l2')

from moe_l2.domain_router_flywheel import DomainRouterFlywheel

# 用临时目录避免污染
tmp = tempfile.mkdtemp()
map_path = os.path.join(tmp, 'domain_router_map_flywheel.json')
fw = DomainRouterFlywheel(
    map_path=map_path,
    top_k=5,
    rebuild_every=20,     # 20 条就重建（测试用）
    min_records=5,
)

# 模拟: domain=codegen, 喂 25 行 EXPERT (L0-L2, 每行 2 token × 4 专家)
fw.set_domain('codegen')
for i in range(25):
    layer = i % 3
    line = f'EXPERT|L{layer}|T2: [10,20,30,40] [10,20,50,60]'
    fw.on_expert_line(line)

# 模拟: domain=math, 喂 10 行 (专家 100/200 高频)
fw.set_domain('math')
for i in range(10):
    layer = i % 3
    line = f'EXPERT|L{layer}|T2: [100,200,30,40] [100,200,50,60]'
    fw.on_expert_line(line)

print('== stats (rebuild 应已自动触发) ==')
print(fw.stats())

# 验证输出 JSON
with open(map_path) as f:
    data = json.load(f)
print('\n== 输出 JSON ==')
print('domains:', list(data['domains'].keys()))
print('codegen L0:', data['domains']['codegen']['per_layer_domain_preferred']['0'])
print('math L0:', data['domains']['math']['per_layer_domain_preferred']['0'])
print('top_k:', data['top_k'])
print('source:', data['source'])

# 断言
assert 'codegen' in data['domains'] and 'math' in data['domains']
assert data['domains']['codegen']['per_layer_domain_preferred']['0'][:2] == [10, 20], data['domains']['codegen']['per_layer_domain_preferred']['0']
assert data['domains']['math']['per_layer_domain_preferred']['0'][:2] == [100, 200]
print('\n✅ 全部断言通过')
