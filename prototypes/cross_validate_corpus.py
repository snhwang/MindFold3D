"""
Empirical cross-validation of circuit rank vs full cubical β₁ over a corpus of
generator-produced shapes.

Sweeps all archetypes × all difficulty tiers × multiple seeds, records:
- V, E, C, circuit_rank (production metric)
- χ, β₀, β₂, cubical_β₁ (validation metric)
- agreement flag
- thin-structure flag (whether the shape is 1-voxel-wide)

Outputs a JSON summary and per-shape CSV for the paper's Heavy appendix.
"""

import json
import random
import sys
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

from cognitive_mapping import SkeletonSpec
from skeleton_generation import (
    TreeSkeleton, AsymmetricSkeleton, LoopSkeleton
)
from shape_generation import _calculate_cycle_count, _count_components
from topology_extras import (
    euler_characteristic, count_enclosed_cavities, cubical_betti_1
)


GRID = (7, 7, 7)

ARCHETYPES = {
    "tree": TreeSkeleton,
    "chiral": AsymmetricSkeleton,
    "bridge": LoopSkeleton,
}

# Difficulty tier -> skeleton parameters (from SkeletonSpec docstring)
TIERS = {
    "low":    {"num_branches": 2, "num_loops": 0, "num_components": 1, "voxel_count": 8,  "packing": "sparse"},
    "medium": {"num_branches": 3, "num_loops": 1, "num_components": 1, "voxel_count": 12, "packing": "medium"},
    "high":   {"num_branches": 5, "num_loops": 2, "num_components": 2, "voxel_count": 18, "packing": "medium"},
    "expert": {"num_branches": 7, "num_loops": 4, "num_components": 3, "voxel_count": 25, "packing": "dense"},
}


def build_edges(voxels_set):
    """Count face-adjacent voxel pairs (undirected)."""
    edges = 0
    voxels = list(voxels_set)
    voxel_set = set(voxels_set)
    for v in voxels:
        x, y, z = v
        # only look in +direction to avoid double-counting
        for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            if (x+dx, y+dy, z+dz) in voxel_set:
                edges += 1
    return edges


def is_thin(voxels_set):
    """Check if shape is 1-voxel-wide: no voxel has more than 4 face neighbors,
    and no 2x2x2 solid block is present."""
    voxel_set = set(voxels_set)
    # No 2x2x2 solid block: for every voxel v, at least one of the 8 corners of
    # its "closed" 2x2x2 block is missing.
    for v in voxel_set:
        x, y, z = v
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 or dy == 0 or dz == 0:
                        continue
                    # check the 2x2x2 block with corner at (min(x, x+dx), ...)
                    x0 = min(x, x+dx); y0 = min(y, y+dy); z0 = min(z, z+dz)
                    block = {(x0+a, y0+b, z0+c) for a in (0,1) for b in (0,1) for c in (0,1)}
                    if block.issubset(voxel_set):
                        return False
    return True


def cross_validate_shape(voxels_set):
    """Compute both metrics and all supporting quantities for one shape."""
    V = len(voxels_set)
    E = build_edges(voxels_set)
    C = _count_components(voxels_set, grid_size=GRID)
    circuit_rank = _calculate_cycle_count(voxels_set, GRID)
    # Sanity check: circuit_rank should equal E - V + C
    formula_check = E - V + C
    assert circuit_rank == formula_check, (
        f"circuit_rank={circuit_rank} != E-V+C={formula_check}"
    )
    chi = euler_characteristic(voxels_set)
    b0 = _count_components(voxels_set, grid_size=(20, 20, 20))
    b2 = count_enclosed_cavities(voxels_set)
    cub_b1 = cubical_betti_1(voxels_set)
    return {
        "V": V,
        "E": E,
        "C": C,
        "circuit_rank": circuit_rank,
        "chi": chi,
        "beta0": b0,
        "beta2": b2,
        "cubical_b1": cub_b1,
        "agree": circuit_rank == cub_b1,
        "thin": is_thin(voxels_set),
    }


def main():
    random.seed(42)
    results = []
    per_archetype_tier = {}

    N_SEEDS = 20  # 20 seeds per (archetype × tier) = 3 × 4 × 20 = 240 shapes

    for arch_name, ArchClass in ARCHETYPES.items():
        for tier_name, tier_params in TIERS.items():
            key = f"{arch_name}/{tier_name}"
            per_archetype_tier[key] = {
                "n": 0, "agree": 0, "thin": 0,
                "circuit_ranks": [], "cubical_b1s": [],
            }
            for seed in range(N_SEEDS):
                random.seed(seed * 1000 + hash(key) % 1000)
                spec = SkeletonSpec(
                    archetype=arch_name,
                    voxel_count=tier_params["voxel_count"],
                    grid_size=GRID,
                    num_branches=tier_params["num_branches"],
                    num_loops=tier_params["num_loops"],
                    num_components=tier_params["num_components"],
                    direction_spread="moderate",
                    planarity="medium",
                    packing=tier_params["packing"],
                )
                try:
                    gen = ArchClass(GRID)
                    voxels = gen.generate(spec)
                except Exception as e:
                    print(f"  {key} seed={seed}: FAILED — {type(e).__name__}: {e}")
                    continue
                if not voxels:
                    continue
                stats = cross_validate_shape(voxels)
                stats["archetype"] = arch_name
                stats["tier"] = tier_name
                stats["seed"] = seed
                results.append(stats)
                per_archetype_tier[key]["n"] += 1
                if stats["agree"]:
                    per_archetype_tier[key]["agree"] += 1
                if stats["thin"]:
                    per_archetype_tier[key]["thin"] += 1
                per_archetype_tier[key]["circuit_ranks"].append(stats["circuit_rank"])
                per_archetype_tier[key]["cubical_b1s"].append(stats["cubical_b1"])

    # Overall summary
    total = len(results)
    agree = sum(1 for r in results if r["agree"])
    thin = sum(1 for r in results if r["thin"])
    disagree = [r for r in results if not r["agree"]]

    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Corpus size: {total} shapes")
    print(f"Circuit rank agrees with cubical β₁: {agree}/{total} ({100*agree/total:.1f}%)")
    print(f"Thin (1-voxel-wide) shapes: {thin}/{total} ({100*thin/total:.1f}%)")
    print(f"Disagreements: {len(disagree)}")

    if disagree:
        print(f"\nDisagreement details:")
        for r in disagree[:10]:
            print(f"  {r['archetype']}/{r['tier']} seed={r['seed']}: "
                  f"V={r['V']} E={r['E']} circuit_rank={r['circuit_rank']} "
                  f"cubical_b1={r['cubical_b1']} thin={r['thin']}")

    print(f"\n{'='*70}")
    print(f"PER-ARCHETYPE × TIER BREAKDOWN")
    print(f"{'='*70}")
    print(f"{'archetype/tier':<20} {'n':<5} {'agree%':<10} {'thin%':<10} "
          f"{'circuit_range':<15} {'cubical_range'}")
    for key, stats in per_archetype_tier.items():
        n = stats["n"]
        if n == 0:
            continue
        agree_pct = 100 * stats["agree"] / n
        thin_pct = 100 * stats["thin"] / n
        cr = stats["circuit_ranks"]
        cb = stats["cubical_b1s"]
        cr_range = f"{min(cr)}-{max(cr)}" if cr else "-"
        cb_range = f"{min(cb)}-{max(cb)}" if cb else "-"
        print(f"{key:<20} {n:<5} {agree_pct:<10.1f} {thin_pct:<10.1f} "
              f"{cr_range:<15} {cb_range}")

    # Save results as JSON
    out_path = Path("/home/user/workspace/cross_validation_results.json")
    summary = {
        "corpus_size": total,
        "circuit_rank_agrees_with_cubical_b1": agree,
        "circuit_rank_agreement_pct": round(100 * agree / total, 2),
        "thin_structure_count": thin,
        "thin_structure_pct": round(100 * thin / total, 2),
        "disagreement_count": len(disagree),
        "per_archetype_tier": per_archetype_tier,
        "n_seeds_per_cell": N_SEEDS,
        "n_archetypes": len(ARCHETYPES),
        "n_tiers": len(TIERS),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Summary written to {out_path}")

    # Also save per-shape CSV
    import csv
    csv_path = Path("/home/user/workspace/cross_validation_per_shape.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "archetype", "tier", "seed", "V", "E", "C",
            "circuit_rank", "chi", "beta0", "beta2", "cubical_b1",
            "agree", "thin"
        ])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"✅ Per-shape data written to {csv_path}")


if __name__ == "__main__":
    main()
