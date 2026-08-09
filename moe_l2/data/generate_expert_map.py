#!/usr/bin/env python3
"""Generate domain→expert mapping from LLAMA_EXPERT_LOG=1 data.

Reads 8 domains × 3 stages of expert_data.log files and produces
a JSON mapping: for each domain × layer, the top-K most frequently
activated experts.

Usage: python generate_expert_map.py [--data-dir ...] [--output ...]
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

# 8 domains matching Phase 1 experiments
DOMAINS = [
    "codegen", "debug", "math", "logic",
    "general_qa", "chinese_tech", "creative_write", "translate",
]
STAGES = ["short", "followup", "longtail"]

# Number of top experts to keep per layer per domain
TOP_K = 32  # sufficient for L2 preload (Qwen3.6: 256 experts, top-16 per step)
# Known backbone experts (appear in all 8 domains' top-15)
BACKBONE_EXPERTS = [41, 72, 89, 95, 112, 127, 191, 217, 221, 231]


def parse_expert_line(line: str):
    """Parse 'EXPERT|L0|T2: [64,161,...] [130,112,...]' → (layer, [experts])

    Qwen3.6 logs: first 4 steps have double-gate (8+8), rest have single-gate (8).
    Both formats are valid — handle 1 or 2 bracket pairs.
    """
    m = re.match(r"EXPERT\|L(\d+)\|T(\d+):\s+\[([^\]]+)\](?:\s+\[([^\]]+)\])?", line.strip())
    if not m:
        return None
    layer = int(m.group(1))
    gate1 = [int(x) for x in m.group(3).split(",")]
    if m.group(4) is not None:
        gate2 = [int(x) for x in m.group(4).split(",")]
        return layer, gate1 + gate2
    return layer, gate1


def parse_domain(data_dir: str, domain: str) -> dict:
    """Parse all 3 stages for one domain → per-layer expert counter."""
    layer_counter = defaultdict(Counter)  # layer → expert → count
    total_lines = 0
    
    for stage in STAGES:
        fp = os.path.join(data_dir, domain, stage, "expert_data.log")
        if not os.path.exists(fp):
            print(f"  [warn] {fp} not found, skipping")
            continue
        n_lines = 0
        for line in open(fp, "r"):
            parsed = parse_expert_line(line)
            if parsed is None:
                continue
            layer, experts = parsed
            layer_counter[layer].update(experts)
            n_lines += 1
        total_lines += n_lines
        print(f"  {domain}/{stage}: {n_lines} lines → {sum(len(layer_counter[layer]) for layer in sorted(layer_counter))} unique expert×layer slots")
    
    return {
        "layer_counter": layer_counter,
        "total_lines": total_lines,
        "total_slots": sum(len(lc) for lc in layer_counter.values()),
    }


def build_mapping(data_dir: str, top_k: int = 32) -> dict:
    """Build the complete domain→expert mapping."""
    # Step 1: Parse all domains
    domain_data = {}
    for d in DOMAINS:
        print(f"\nProcessing {d}...")
        domain_data[d] = parse_domain(data_dir, d)
    
    # Step 2: Build per-domain × per-layer top-K lists
    domain_layers = {}
    for d in DOMAINS:
        lc = domain_data[d]["layer_counter"]
        per_layer_top = {}
        for layer in sorted(lc.keys()):
            top_experts = [e for e, _ in lc[layer].most_common(top_k)]
            per_layer_top[str(layer)] = top_experts
        domain_layers[d] = per_layer_top
    
    # Step 3: Compute backbone experts (all domains' top-15 intersection)
    # Already known from Phase 1 analysis, but recompute to verify
    domain_top15 = {}
    for d in DOMAINS:
        combined = Counter()
        for lc in domain_data[d]["layer_counter"].values():
            combined += lc
        domain_top15[d] = {e for e, _ in combined.most_common(15)}
    
    backbone = sorted(set.intersection(*domain_top15.values()))
    print(f"\nBackbone experts (top-15 intersection across {len(DOMAINS)} domains): {backbone}")
    
    # Step 4: Compute domain-specific experts for each layer
    # For each domain × layer, identify experts that appear MORE often
    # for this domain than the cross-domain median
    # (This is a per-layer "domain preference" score)
    domain_preferred = {}
    for d in DOMAINS:
        lc = domain_data[d]["layer_counter"]
        preferred = {}
        for layer in sorted(lc.keys()):
            # Get this domain's frequency for each expert
            this_freq = lc[layer]
            # Get cross-domain median frequency for each expert
            cross_freqs = defaultdict(list)
            for other_d in DOMAINS:
                if other_d == d:
                    continue
                other_lc = domain_data[other_d]["layer_counter"]
                if layer in other_lc:
                    for e, cnt in other_lc[layer].items():
                        cross_freqs[e].append(cnt)
            
            # Find experts that appear notably more in this domain
            preferred_experts = []
            for e, cnt in this_freq.most_common():
                if e in cross_freqs:
                    others = cross_freqs[e]
                    avg_other = sum(others) / len(others)
                    if cnt > avg_other * 1.5:  # 50% more than cross-domain average
                        preferred_experts.append(e)
                # If expert only appears in this domain for this layer, definitely preferred
                elif cnt > 0:
                    preferred_experts.append(e)
            
            preferred[str(layer)] = preferred_experts[:top_k]
        domain_preferred[d] = preferred
    
    # Step 5: Build metadata
    # Find layers with highest/lowest domain specificity
    layer_specificity = {}
    for layer in range(40):
        # Compute average cross-domain Jaccard at this layer
        all_experts_l = []
        for d in DOMAINS:
            lc = domain_data[d]["layer_counter"]
            if layer in lc:
                all_experts_l.append(set(lc[layer].keys()))
        
        if len(all_experts_l) >= 2:
            jaccards = []
            for i, s1 in enumerate(all_experts_l):
                for s2 in all_experts_l[i+1:]:
                    if s1 and s2:
                        jaccards.append(len(s1 & s2) / len(s1 | s2))
            avg_j = sum(jaccards) / len(jaccards) if jaccards else 1.0
        else:
            avg_j = 1.0
        
        # Non-shared ratio
        common = set.intersection(*all_experts_l) if all_experts_l else set()
        union = set.union(*all_experts_l) if all_experts_l else set()
        specialist_ratio = (len(union) - len(common)) / max(len(union), 1) if union else 0
        
        layer_specificity[str(layer)] = {
            "cross_domain_jaccard": round(avg_j, 4),
            "specialist_ratio": round(specialist_ratio, 4),
            "n_unique_experts": len(union),
            "n_backbone": len(backbone),
        }
    
    # Step 6: Assemble final mapping
    mapping = {
        "model": "qwen3.6-35b-a3b",
        "num_layers": 40,
        "num_experts": 256,
        "experts_per_token": 16,
        "backbone_experts": backbone,
        "top_k_per_layer": top_k,
        "generated_from": f"{len(DOMAINS)} domains × {len(STAGES)} stages × expert_data.log",
        "layer_specificity": layer_specificity,
        "domains": {},
    }
    
    for d in DOMAINS:
        mapping["domains"][d] = {
            "per_layer_top": domain_layers[d],
            "per_layer_domain_preferred": domain_preferred[d],
            "total_lines": domain_data[d]["total_lines"],
            "total_slots": domain_data[d]["total_slots"],
        }
    
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Generate domain→expert mapping")
    parser.add_argument("--data-dir", default="/opt/data/副业操作技巧/可发素材/moe-l2-qwen36-raw",
                        help="Directory containing domain/stage/expert_data.log")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: ../domain_expert_map.json)")
    parser.add_argument("--top-k", type=int, default=32,
                        help="Number of top experts per layer per domain")
    args = parser.parse_args()
    
    if args.output is None:
        # Default: place next to this script
        args.output = os.path.join(os.path.dirname(__file__), "domain_expert_map.json")
    
    print(f"Building domain→expert mapping from: {args.data_dir}")
    print(f"Domains: {DOMAINS}")
    print(f"Top-K: {args.top_k}")
    print("=" * 60)
    
    mapping = build_mapping(args.data_dir, top_k=args.top_k)
    
    with open(args.output, "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    
    size_kb = os.path.getsize(args.output) / 1024
    print(f"\n{'=' * 60}")
    print(f"✓ Mapping saved to: {args.output}")
    print(f"  File size: {size_kb:.1f} KB")
    print(f"  Domains: {len(mapping['domains'])}")
    print(f"  Backbone experts: {len(mapping['backbone_experts'])}")
    print(f"  Layers: {len(mapping['layer_specificity'])}")
    
    # Summary stats
    total_top = 0
    for d in DOMAINS:
        for layer in mapping["domains"][d]["per_layer_top"]:
            total_top += len(mapping["domains"][d]["per_layer_top"][layer])
    print(f"  Total top-K entries: {total_top}")


if __name__ == "__main__":
    main()
