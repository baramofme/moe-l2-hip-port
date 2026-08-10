#!/usr/bin/env python3
"""聚合并集模拟：V4 分话题路由表（4 话题 × top-K）并集后 → 每层专家数 / 覆盖率 / 内存占用

核心问题：决策 #3"聚合并集"会不会退化回 whole-pin（每层 256 个全 pin）？
方法：对每个 top-K，各话题独立统计 top-K → 按层取并集 → 在完整 trace 上算覆盖率。
"""
import re, json
from collections import Counter, defaultdict

TRACE = '/opt/data/moe-l2/测试数据备份/v4-actset-trace-20260808.log'
LAYERS = 43
EXPERT_BYTES_V4 = 2.7 * 1024 * 1024  # V4 每专家 2.7MB IQ2_M

# ── 1. 解析 trace 并按话题分段 ──
# 125 请求：0-49 general, 50-74 math, 75-99 code, 100-124 chat
# 每请求一个 launch_slot_，EXPERT 行跟在其后。用 launch_slot_ 计数分话题。
exp_re = re.compile(r'EXPERT\|L(\d+)\|T\d+:\s*(.*)')
slot_re = re.compile(r'launch_slot_')
EXPERT_N_PER_TOKEN = LAYERS  # V4 每 token 每层 1 行

# 收集每个 (话题, layer) 的专家频率
topic_freq = {t: defaultdict(Counter) for t in ['general', 'math', 'code', 'chat']}
# trace 总激活（全部话题合并，用于全局覆盖率）
all_acts = []  # (layer, experts)

topic_of = {}
with open(TRACE, errors='replace') as f:
    slot_count = 0
    for line in f:
        if slot_re.search(line):
            idx = slot_count
            if idx < 50: t = 'general'
            elif idx < 75: t = 'math'
            elif idx < 100: t = 'code'
            else: t = 'chat'
            topic_of[slot_count] = t
            slot_count += 1
            continue
        m = exp_re.match(line.strip())
        if not m:
            continue
        layer = int(m.group(1))
        body = m.group(2)
        experts = []
        for seg in re.findall(r'\[([0-9, ]+)\]', body):
            for x in seg.split(','):
                x = x.strip()
                if x:
                    experts.append(int(x))
        if not experts:
            continue
        all_acts.append((layer, tuple(sorted(set(experts)))))
        # 归属话题：用最近一次 launch_slot_ 后的行（简化：按顺序，topic 由 slot_count 推断）
        t = topic_of.get(slot_count - 1, 'general') if slot_count > 0 else 'general'
        for e in set(experts):
            topic_freq[t][layer][e] += 1

print(f'解析完成: {len(all_acts)} token-层样本, 话题分段: { {k: sum(1 for v in topic_of.values() if v==k) for k in topic_of.values()} }')

# ── 2. 对每个 top-K：分话题表 → 并集 → 覆盖率 ──
print(f'\n{"top-K":<7}{"单话题/层":<10}{"并集/层":<10}{"并集内存":<10}{"覆盖率":<10}{"是否退化":<12}')
print('-' * 62)

results = []
for top_k in [30, 50, 75, 100, 150, 200]:
    # 各话题 top-K 表
    topic_tables = {}
    for t in topic_freq:
        ttab = {}
        for layer in range(LAYERS):
            ttab[layer] = set(e for e, _ in topic_freq[t][layer].most_common(top_k))
        topic_tables[t] = ttab

    # 按层并集
    union = {}
    union_size = 0
    for layer in range(LAYERS):
        s = set()
        for t in topic_tables:
            s |= topic_tables[t][layer]
        union[layer] = s
        union_size += len(s)

    # 覆盖率：全部激活中命中并集的比例
    total_act = 0
    total_hit = 0
    for layer, exps in all_acts:
        total_act += len(exps)
        total_hit += sum(1 for e in exps if e in union[layer])
    coverage = total_hit / total_act if total_act else 0

    mem_gb = union_size * EXPERT_BYTES_V4 / 1024**3
    per_layer_avg = union_size / LAYERS
    degraded = '是(≈全量)' if per_layer_avg > 200 else ('接近' if per_layer_avg > 150 else '否')
    print(f'{top_k:<7}{top_k:<10}{per_layer_avg:<10.0f}{mem_gb:<10.1f}{coverage*100:<9.1f}%{degraded:<12}')
    results.append({'top_k': top_k, 'union_per_layer': round(per_layer_avg, 1), 'union_gb': round(mem_gb, 2), 'coverage': round(coverage*100, 1), 'degraded': degraded})

# 对比：单话题 top-75（现有表基准）
print('\n=== 对比基准 ===')
print('单话题 top-75: 每层 ~61 个, 内存 ~6.9GB, 该话题内覆盖率 90.6%')
print('whole-pin:     每层 256 个, 内存 82GB(全量), 覆盖率 100%')

# 存结果
out = {
    'model': 'DeepSeek-V4-Flash IQ2_M', 'method': '分话题 top-K 按层并集',
    'results': results,
    'note': '并集/层 >150 = 接近退化回 whole-pin；覆盖率按全 trace 所有话题激活计算',
}
with open('/opt/data/moe-l2/scripts/bench/union_sim_result.json', 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('\n结果已存: scripts/bench/union_sim_result.json')
