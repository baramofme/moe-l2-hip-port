#!/usr/bin/env python3
"""选择性 pin 模拟：V4 trace → 不同 top-K 路由表 → 每 token 冷专家数 / 覆盖率 / fault 开销 / 预估速度

模拟"选择性 pin"方案：只 pin 路由表 top-K 专家，冷专家走 on-demand fault 兜底。
对比 whole-pin（82GB / 10.16 t/s）和动态 pin LRU（17-24GB / 4-5 t/s）。
"""
import sys, re, json
from collections import Counter, defaultdict

TRACE = '/opt/data/moe-l2/测试数据备份/v4-actset-trace-20260808.log'

# 实测基准
WHOLE_PIN_TPS = 10.16          # whole-pin 实测（稳定轮）
WHOLE_PIN_MS_PER_TOKEN = 1000 / WHOLE_PIN_TPS   # 98.4 ms
FAULT_MS = 2.1                 # 冷专家首次注册 = 锁页 0.43 + 缺页读盘 ~1.7
EXPERT_BYTES_V4 = 2.7 * 1024 * 1024  # V4 每专家 2.7MB（IQ2_M）
LAYERS = 43

# ── 1. 解析 trace：每 token 每层激活专家 ──
# EXPERT|L0|T2: [254,222,245,200,53,35] [239,202,122,23,115,57]
exp_re = re.compile(r'EXPERT\|L(\d+)\|T\d+:\s*(.*)')
# 每条 EXPERT 行 = 1 个 token（T 是 batch 内 token id），方括号里是 1 组激活专家
# 实际 V4 格式：每行 T 固定，一组 [..] 即该 token 该层 top-6+1
tokens = []  # list of (layer, experts_tuple) for each token-layer
with open(TRACE, errors='replace') as f:
    for line in f:
        m = exp_re.match(line.strip())
        if not m:
            continue
        layer = int(m.group(1))
        body = m.group(2)
        # 取所有 [..] 组的并集（通常 1 组；保险起见合并）
        experts = []
        for seg in re.findall(r'\[([0-9, ]+)\]', body):
            for x in seg.split(','):
                x = x.strip()
                if x:
                    experts.append(int(x))
        if experts:
            tokens.append((layer, tuple(sorted(set(experts)))))

print(f'解析完成: {len(tokens)} 个 token-层样本')
# 按 token 聚合：找出连续 token id 边界（T 变化 = 新 token）
# 简化：每层各 token 独立统计，用 (layer, expert) 频率代替

# ── 2. 全局专家频率（聚合表）──
layer_freq = defaultdict(Counter)   # layer -> expert -> count
for layer, exps in tokens:
    for e in exps:
        layer_freq[layer][e] += 1

# ── 3. 对每个 top-K 模拟 ──
print(f'\n{"top-K":<8}{"每层pin数":<10}{"pin内存":<10}{"覆盖率":<10}{"冷专家/token":<14}{"fault开销":<10}{"预估速度":<10}')
print('-' * 80)

results = []
for top_k in [30, 50, 75, 100, 150, 200]:
    # 生成 top-K 路由表
    router = {}   # layer -> set of top-K experts
    pinned_count = 0
    for layer in range(LAYERS):
        top = [e for e, _ in layer_freq[layer].most_common(top_k)]
        router[layer] = set(top)
        pinned_count += len(top)

    # 计算覆盖率 + 每 token 冷专家数
    total_act = 0
    total_hit = 0
    # 按 token 分组：V4 每 token 每层 1 条 EXPERT 行 → 每 LAYERS 行 = 1 token
    token_cold = []   # 每个 token 的冷专家总数
    token_total = []
    cur_cold, cur_total = 0, 0
    for idx, (layer, exps) in enumerate(tokens):
        cold = sum(1 for e in exps if e not in router[layer])
        cur_cold += cold
        cur_total += len(exps)
        total_act += len(exps)
        total_hit += len(exps) - cold
        if (idx + 1) % LAYERS == 0:   # 一个 token 结束
            token_cold.append(cur_cold)
            token_total.append(cur_total)
            cur_cold, cur_total = 0, 0

    n_tokens = len(token_cold)
    cold_per_token_avg = sum(token_cold) / n_tokens if n_tokens else 0
    coverage = total_hit / total_act if total_act else 0

    # fault 开销：每 token 冷专家数 × 2.1ms
    fault_ms = cold_per_token_avg * FAULT_MS
    est_tps = 1000 / (WHOLE_PIN_MS_PER_TOKEN + fault_ms) if (WHOLE_PIN_MS_PER_TOKEN + fault_ms) > 0 else 0

    pin_mem_gb = pinned_count * EXPERT_BYTES_V4 / 1024**3
    print(f'{top_k:<8}{pinned_count//LAYERS:<10}{pin_mem_gb:<10.1f}{coverage*100:<9.1f}%{cold_per_token_avg:<14.1f}{fault_ms:<10.1f}ms{est_tps:<10.2f}')
    results.append((top_k, pinned_count//LAYERS, pin_mem_gb, coverage, cold_per_token_avg, fault_ms, est_tps))

print('\n=== 对比基准 ===')
print(f'whole-pin:  82GB  RSS, 10.16 t/s')
print(f'动态pinLRU: 17-24GB, 4-5 t/s')

# ── 4. 存结果 ──
out = {
    'model': 'DeepSeek-V4-Flash IQ2_M',
    'base': {'whole_pin_tps': WHOLE_PIN_TPS, 'fault_ms': FAULT_MS, 'expert_mb': EXPERT_BYTES_V4/1024**2},
    'n_token_layer_samples': len(tokens),
    'results': [
        {'top_k': k, 'pinned_per_layer': pl, 'pin_gb': round(m,2), 'coverage': round(c*100,1),
         'cold_per_token': round(cpt,2), 'fault_ms_per_token': round(fm,1), 'est_tps': round(t,2)}
        for k, pl, m, c, cpt, fm, t in results
    ],
}
with open('/opt/data/moe-l2/scripts/bench/selective_pin_sim_result.json', 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('\n结果已存: scripts/bench/selective_pin_sim_result.json')
