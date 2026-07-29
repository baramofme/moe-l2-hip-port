#!/usr/bin/env python3
"""
LRU GPU expert cache trace simulator — Phase 1.5 of moe-l2.

Parses EXPERT log files from Qwen3.6 runs, simulates per-layer LRU caches
with configurable capacity, pin lists, and allocation strategies.

Usage:
  python lru_trace_sim.py                      # default run (all configs)
  python lru_trace_sim.py --cache-sizes 32 64  # custom sizes
  python lru_trace_sim.py --verbose            # per-layer breakdown
"""

import os
import re
import json
import argparse
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple

LOG_DIR = "/tmp"
LOG_FILES = {
    "math": f"{LOG_DIR}/expert_math.log",
    "code": f"{LOG_DIR}/expert_code.log",
    "general": f"{LOG_DIR}/expert_general.log",
    "cn_math": f"{LOG_DIR}/expert_cn_math.log",
    "cn_tech": f"{LOG_DIR}/expert_cn_tech.log",
}

N_LAYERS = 40
EXPERTS_PER_ACCESS = 8  # top-8 experts selected per token position

LRU_LINE_RE = re.compile(r"EXPERT\|L(\d+)\|T\d+:\s*(\[.*?\])(?:\s*(\[.*?\]))?")


# ── Parsing ────────────────────────────────────────────────────────────────

@dataclass
class AccessEvent:
    """One expert access: a specific (layer, expert_id) pair at a point in the trace."""
    layer: int
    expert_id: int
    domain: str
    position: int      # which token position within this access
    token_step: int    # T1, T2, etc.


def parse_log(filepath: str, domain: str) -> List[AccessEvent]:
    """Parse one EXPERT log file into a list of access events."""
    events = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("EXPERT|"):
                continue
            m = LRU_LINE_RE.match(line)
            if not m:
                continue
            layer = int(m.group(1))
            # Collect all bracket groups (may be 1 or 2)
            brackets = [m.group(2)]
            if m.group(3):
                brackets.append(m.group(3))
            for pos_idx, bracket in enumerate(brackets):
                experts = [int(x.strip()) for x in bracket.strip("[]").split(",")]
                for exp_id in experts:
                    events.append(AccessEvent(
                        layer=layer,
                        expert_id=exp_id,
                        domain=domain,
                        position=pos_idx,
                        token_step=0,  # simplified
                    ))
    return events


def load_all_traces() -> Dict[str, List[AccessEvent]]:
    """Load all domain traces. Returns {domain: [events]}."""
    traces = {}
    for domain, path in LOG_FILES.items():
        if not os.path.exists(path):
            print(f"  ⚠️  {path} not found, skipping domain '{domain}'")
            continue
        traces[domain] = parse_log(path, domain)
        print(f"  ✓ {domain}: {len(traces[domain])} access events ({len(traces[domain])//8} positions)")
    return traces


# ── Per-layer LRU cache ────────────────────────────────────────────────────

class LayerLRUCache:
    """Per-layer LRU cache with pinned experts."""
    
    def __init__(self, capacity: int, pinned: Set[int] = None):
        self.capacity = capacity
        self.pinned = pinned or set()
        self._cache: OrderedDict[int, None] = OrderedDict()
        self.hits = 0
        self.misses = 0
        
    @property
    def used(self) -> int:
        return len(self._cache)
    
    @property
    def available(self) -> int:
        return self.capacity - self.used
    
    def access(self, expert_id: int) -> bool:
        """Returns True if hit, False if miss."""
        if expert_id in self._cache:
            self._cache.move_to_end(expert_id)
            self.hits += 1
            return True
        
        self.misses += 1
        
        # Evict if full
        if self.used >= self.capacity:
            self._evict_one()
        
        self._cache[expert_id] = None
        return False
    
    def _evict_one(self):
        """Evict the LRU non-pinned entry."""
        for eid in list(self._cache.keys()):
            if eid not in self.pinned:
                del self._cache[eid]
                return
        # All entries are pinned — can't evict
        pass
    
    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.get_hit_rate() * 100, 2),
            "used": self.used,
            "capacity": self.capacity,
            "pinned_count": len(self.pinned),
        }


# ── Simulator ────────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    name: str
    per_layer_capacity: int      # same for all layers, or...
    layer_weights: List[int] = None  # ...per-layer capacity if set
    pin_strategy: str = "none"   # "none", "universal", "universal+domain"
    layer_aware: bool = False    # allocate more slots to middle layers


def compute_universal_experts(traces: Dict[str, List[AccessEvent]]) -> Dict[int, Set[int]]:
    """
    For each layer, find experts that appear in ALL domains.
    Returns {layer: {expert_id, ...}}
    """
    layer_domain_experts: Dict[int, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))
    
    for domain, events in traces.items():
        for ev in events:
            layer_domain_experts[ev.layer][domain].add(ev.expert_id)
    
    universal = {}
    for layer, domain_sets in layer_domain_experts.items():
        all_sets = list(domain_sets.values())
        if len(all_sets) >= 2:
            universal[layer] = set.intersection(*all_sets)
        else:
            universal[layer] = set()
    
    return universal


def compute_domain_preferences(traces: Dict[str, List[AccessEvent]]) -> Dict[str, Dict[int, Set[int]]]:
    """
    For each domain, find the top-N most preferred experts per layer.
    Returns {domain: {layer: {expert_id, ...}}}
    Using a simple frequency-based approach: experts with >2x domain frequency vs avg.
    """
    # Count per (domain, layer, expert)
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    total_events = defaultdict(int)
    
    for domain, events in traces.items():
        for ev in events:
            counts[domain][ev.layer][ev.expert_id] += 1
            total_events[domain] += 1
    
    # Average frequency per domain per layer
    domain_prefs = {}
    for domain, layer_counts in counts.items():
        domain_prefs[domain] = {}
        for layer in range(N_LAYERS):
            expert_counts = layer_counts[layer]
            if not expert_counts:
                domain_prefs[domain][layer] = set()
                continue
            avg = sum(expert_counts.values()) / len(expert_counts)
            # Experts with >2x average frequency
            preferred = {eid for eid, cnt in expert_counts.items() if cnt > avg * 1.5}
            domain_prefs[domain][layer] = preferred
    
    return domain_prefs


def get_pinned_set(layer: int, universal: Dict[int, Set[int]],
                   domain_prefs: Dict[str, Dict[int, Set[int]]] = None,
                   strategy: str = "universal") -> Set[int]:
    """Get the set of pinned experts for a given layer based on strategy."""
    if strategy == "none":
        return set()
    
    pinned = set(universal.get(layer, set()))
    
    if strategy == "universal+domain" and domain_prefs:
        for domain_prefs_layer in domain_prefs.values():
            pinned |= domain_prefs_layer.get(layer, set())
    
    return pinned


def get_layer_capacity(base_capacity: int, layer: int, layer_aware: bool) -> int:
    """Calculate capacity for a specific layer, optionally with layer-aware allocation."""
    if not layer_aware:
        return base_capacity
    
    # Layer-aware weighting: more capacity for middle layers (higher domain differentiation)
    # Layer 0-2: 0.7x (general, less benefit from LRU)
    # Layer 3-7: 1.0x
    # Layer 8-20: 1.3x (highest domain differentiation → most benefit)
    # Layer 21-29: 1.0x
    # Layer 30-39: 0.8x
    if layer <= 2:
        weight = 0.7
    elif layer <= 7:
        weight = 1.0
    elif layer <= 20:
        weight = 1.3
    elif layer <= 29:
        weight = 1.0
    else:
        weight = 0.8
    
    return max(8, int(base_capacity * weight))


def run_simulation(traces: Dict[str, List[AccessEvent]],
                   config: SimConfig,
                   universal: Dict[int, Set[int]],
                   domain_prefs: Dict[str, Dict[int, Set[int]]] = None) -> dict:
    """Run LRU simulation for one config. Returns per-layer + aggregate stats."""
    
    # Initialize per-layer caches
    caches = {}
    for layer in range(N_LAYERS):
        cap = get_layer_capacity(config.per_layer_capacity, layer, config.layer_aware)
        pinned = get_pinned_set(layer, universal, domain_prefs, config.pin_strategy)
        caches[layer] = LayerLRUCache(capacity=cap, pinned=pinned)
    
    # Process traces in domain order (simulating topic switching)
    total_hits = 0
    total_misses = 0
    
    for domain, events in traces.items():
        for ev in events:
            hit = caches[ev.layer].access(ev.expert_id)
    
    # Aggregate
    per_layer_stats = {}
    for layer, cache in caches.items():
        per_layer_stats[layer] = cache.get_stats()
        total_hits += cache.hits
        total_misses += cache.misses
    
    total = total_hits + total_misses
    overall_hit_rate = total_hits / total if total > 0 else 0.0
    
    # Calculate bandwidth (misses × 1.01 MB per expert, per-layer sum)
    # Each miss = one expert transfer GPU→CPU (or L2→L0)
    total_bandwidth_mb = total_misses * 1.01
    
    # Per-layer bandwidth
    layer_bandwidth = {}
    for layer, s in per_layer_stats.items():
        layer_bandwidth[layer] = round(s["misses"] * 1.01, 2)
    
    return {
        "config_name": config.name,
        "per_layer_capacity": config.per_layer_capacity,
        "layer_aware": config.layer_aware,
        "pin_strategy": config.pin_strategy,
        "total_hits": total_hits,
        "total_misses": total_misses,
        "total_accesses": total,
        "overall_hit_rate": round(overall_hit_rate * 100, 2),
        "total_bandwidth_mb": round(total_bandwidth_mb, 1),
        "per_layer": per_layer_stats,
        "layer_bandwidth": layer_bandwidth,
        "total_pinned": sum(len(get_pinned_set(l, universal, domain_prefs, config.pin_strategy)) for l in range(N_LAYERS)),
    }


def print_results(results: list, verbose: bool = False):
    """Print simulation results in a readable format."""
    print("\n" + "=" * 78)
    print("  LRU GPU Expert Cache — Trace Simulation Results")
    print("  Model: Qwen3.6-35B-A3B (IQ2_M, 256 experts/layer × 40 layers)")
    print("  Expert size: 1.01 MB | Top-8 active/layer")
    print("=" * 78)
    
    # Configs tested summary
    print(f"\n  Configurations tested: {len(results)}")
    print(f"  {'Name':<30} {'Cap/L':<8} {'Pin':<18} {'L-Aware':<8} {'Hit%':<8} {'BW(MB)':<8} {'Misses':<8}")
    print(f"  {'-'*30} {'-'*8} {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    best_hit_rate = 0
    best_config = None
    
    for r in sorted(results, key=lambda x: x["overall_hit_rate"], reverse=True):
        name = r["config_name"]
        cap = r["per_layer_capacity"]
        pin = r["pin_strategy"]
        la = "✓" if r["layer_aware"] else "✗"
        hr = r["overall_hit_rate"]
        bw = r["total_bandwidth_mb"]
        m = r["total_misses"]
        
        print(f"  {name:<30} {cap:<8} {pin:<18} {la:<8} {hr:<8} {bw:<8} {m:<8}")
        
        if hr > best_hit_rate:
            best_hit_rate = hr
            best_config = r
    
    # Best config breakdown
    if best_config and verbose:
        print(f"\n{'─' * 78}")
        print(f"  Best config: {best_config['config_name']} ({best_config['overall_hit_rate']}% hit rate)")
        print(f"  Total pinned experts: {best_config['total_pinned']} across {N_LAYERS} layers")
        print(f"  Total bandwidth saved: {best_config['total_bandwidth_mb']} MB (vs {best_config['total_misses'] * 1.01:.0f} MB baseline)")
        print(f"\n  Per-layer breakdown (hit rate):")
        print(f"  {'Layer':<8} {'Hits':<8} {'Misses':<8} {'Hit%':<8} {'Cap':<8} {'Pinned':<8} {'BW(MB)':<8}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        
        for layer in range(N_LAYERS):
            s = best_config["per_layer"][layer]
            bw = best_config["layer_bandwidth"][layer]
            print(f"  L{layer:<5} {s['hits']:<8} {s['misses']:<8} {s['hit_rate']:<8} {s['capacity']:<8} {s['pinned_count']:<8} {bw:<8}")
    
    # Summary
    print(f"\n{'─' * 78}")
    no_pin = [r for r in results if r["pin_strategy"] == "none" and not r["layer_aware"]]
    best_pin = [r for r in results if r["pin_strategy"] in ("universal", "universal+domain") and not r["layer_aware"]]
    best_la = [r for r in results if r["layer_aware"]]
    
    if no_pin:
        best_no = max(no_pin, key=lambda x: x["overall_hit_rate"])
        print(f"  Floor (pure LRU, best):      {best_no['overall_hit_rate']}% @ {best_no['per_layer_capacity']} slots/layer")
    if best_pin:
        best_p = max(best_pin, key=lambda x: x["overall_hit_rate"])
        print(f"  With pins, best:              {best_p['overall_hit_rate']}% @ {best_p['per_layer_capacity']} slots/layer")
    if best_la:
        best_l = max(best_la, key=lambda x: x["overall_hit_rate"])
        print(f"  Layer-aware allocation:       {best_l['overall_hit_rate']}% @ {best_l['per_layer_capacity']} base slots/layer")
    
    print(f"\n  85% threshold: {'✅ MET' if best_hit_rate >= 85 else '❌ NOT MET — need more cache'}")
    print(f"  90% threshold: {'✅ MET' if best_hit_rate >= 90 else '❌ NOT MET'}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="LRU GPU expert cache trace simulator")
    parser.add_argument("--cache-sizes", type=int, nargs="+", default=[16, 32, 48, 64, 96, 128],
                        help="Per-layer cache capacities to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-layer breakdown")
    parser.add_argument("--json", "-j", action="store_true", help="Output JSON for programmatic use")
    args = parser.parse_args()
    
    print("Loading traces...")
    traces = load_all_traces()
    if not traces:
        print("❌ No trace data found. Run data collection first.")
        return
    
    n_domains = len(traces)
    total_events = sum(len(events) for events in traces.values())
    print(f"\n  {n_domains} domains, {total_events} total access events "
          f"({total_events // 8} token positions)")
    
    # Compute universal experts and domain preferences
    print("\nComputing expert statistics...")
    universal = compute_universal_experts(traces)
    domain_prefs = compute_domain_preferences(traces)
    
    # Show universal expert stats
    uni_counts = [len(u) for u in universal.values()]
    print(f"  Universal experts/layer: min={min(uni_counts)}, max={max(uni_counts)}, "
          f"avg={sum(uni_counts)/len(uni_counts):.1f}")
    
    # Domain preferences stats
    pref_counts = []
    for domain, layer_prefs in domain_prefs.items():
        for lp in layer_prefs.values():
            pref_counts.append(len(lp))
    if pref_counts:
        print(f"  Domain-preferred experts/layer: avg={sum(pref_counts)/len(pref_counts):.1f}")
    
    # Run simulations
    results = []
    configs = []
    
    for size in args.cache_sizes:
        # Baseline: pure LRU, uniform
        configs.append(SimConfig(
            name=f"LRU-{size}",
            per_layer_capacity=size,
            pin_strategy="none",
            layer_aware=False,
        ))
        # With universal expert pinning
        configs.append(SimConfig(
            name=f"PinU-{size}",
            per_layer_capacity=size,
            pin_strategy="universal",
            layer_aware=False,
        ))
        # With universal + domain preference pinning
        configs.append(SimConfig(
            name=f"PinUD-{size}",
            per_layer_capacity=size,
            pin_strategy="universal+domain",
            layer_aware=False,
        ))
        # Layer-aware + universal pins
        configs.append(SimConfig(
            name=f"LA-PinU-{size}",
            per_layer_capacity=size,
            pin_strategy="universal",
            layer_aware=True,
        ))
    
    print(f"\nRunning {len(configs)} simulations...")
    for cfg in configs:
        r = run_simulation(traces, cfg, universal, domain_prefs)
        results.append(r)
    
    print_results(results, verbose=args.verbose)
    
    # JSON output
    if args.json:
        print("\n\n--- JSON ---")
        print(json.dumps({
            "model": "Qwen3.6-35B-A3B IQ2_M",
            "n_domains": n_domains,
            "total_events": total_events,
            "n_layers": N_LAYERS,
            "results": [{
                "config": r["config_name"],
                "hit_rate": r["overall_hit_rate"],
                "misses": r["total_misses"],
                "bandwidth_mb": r["total_bandwidth_mb"],
                "capacity_per_layer": r["per_layer_capacity"],
                "pin_strategy": r["pin_strategy"],
                "layer_aware": r["layer_aware"],
                "total_pinned": r["total_pinned"],
            } for r in sorted(results, key=lambda x: x["overall_hit_rate"], reverse=True)]
        }, indent=2))
    
    # Save results JSON for reference
    output_path = f"{LOG_DIR}/lru_sim_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": "Qwen3.6-35B-A3B IQ2_M",
            "n_domains": n_domains,
            "total_events": total_events,
            "n_layers": N_LAYERS,
            "universal_experts_per_layer": {str(k): sorted(v) for k, v in sorted(universal.items())},
            "results": [{
                "config": r["config_name"],
                "hit_rate": r["overall_hit_rate"],
                "misses": r["total_misses"],
                "bandwidth_mb": r["total_bandwidth_mb"],
                "capacity_per_layer": r["per_layer_capacity"],
                "pin_strategy": r["pin_strategy"],
                "layer_aware": r["layer_aware"],
                "total_pinned": r["total_pinned"],
                "per_layer_hit_rates": {str(k): v["hit_rate"] for k, v in sorted(r["per_layer"].items())},
            } for r in results]
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
