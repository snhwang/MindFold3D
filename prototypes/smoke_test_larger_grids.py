"""
Smoke test: does LoopSkeleton on larger grids produce shapes where
graph circuit rank actually corresponds to continuum β₁ holes?

Sweeps grid sizes {7, 9, 11, 13}³ × target_loops {1, 2, 3, 4} × voxel counts
scaled to grid size. Reports μ(G) vs β₁(X) per shape.

The question we want answered: how big does the grid need to be before
LoopSkeleton produces shapes with μ ≈ β₁ ≈ target_loops (i.e., actual
through-holes rather than 2×2 face-patch cycles)?
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mindfold3d.cognitive_mapping import SkeletonSpec
from mindfold3d.skeleton_generation import LoopSkeleton
from mindfold3d.shape_generation import _calculate_cycle_count, _count_components
from mindfold3d.topology_extras import (
    euler_characteristic, count_enclosed_cavities, cubical_betti_1
)


def build_edges(voxels_set):
    edges = 0
    vs = set(voxels_set)
    for v in vs:
        x, y, z = v
        for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            if (x+dx, y+dy, z+dz) in vs:
                edges += 1
    return edges


def is_thin(voxels_set):
    vs = set(voxels_set)
    for v in vs:
        x, y, z = v
        for dx in (-1, 1):
            for dy in (-1, 1):
                for dz in (-1, 1):
                    x0 = min(x, x+dx); y0 = min(y, y+dy); z0 = min(z, z+dz)
                    block = {(x0+a, y0+b, z0+c) for a in (0,1) for b in (0,1) for c in (0,1)}
                    if block.issubset(vs):
                        return False
    return True


def run_cell(grid_size, num_loops, voxel_count, packing, n_seeds=15):
    """Return per-seed (mu, b1, thin, agree) list plus summary stats."""
    GRID = (grid_size, grid_size, grid_size)
    mus, b1s, thins, agrees = [], [], [], []
    failures = 0
    for seed in range(n_seeds):
        random.seed(seed * 1000 + hash((grid_size, num_loops, voxel_count)) % 1000)
        spec = SkeletonSpec(
            archetype="bridge",
            voxel_count=voxel_count,
            grid_size=GRID,
            num_branches=max(3, num_loops + 2),
            num_loops=num_loops,
            num_components=1,
            direction_spread="moderate",
            planarity="medium",
            packing=packing,
        )
        try:
            gen = LoopSkeleton(GRID)
            voxels = gen.generate(spec)
        except Exception as e:
            failures += 1
            continue
        if not voxels:
            failures += 1
            continue
        mu = _calculate_cycle_count(voxels, GRID)
        b1 = cubical_betti_1(voxels)
        mus.append(mu)
        b1s.append(b1)
        thins.append(is_thin(voxels))
        agrees.append(mu == b1)
    return {
        "n": len(mus), "failures": failures,
        "mus": mus, "b1s": b1s, "thins": thins, "agrees": agrees,
    }


def summarize(cell):
    if cell["n"] == 0:
        return "no shapes generated"
    mean_mu = sum(cell["mus"]) / cell["n"]
    mean_b1 = sum(cell["b1s"]) / cell["n"]
    agree_pct = 100 * sum(cell["agrees"]) / cell["n"]
    thin_pct = 100 * sum(cell["thins"]) / cell["n"]
    mu_min, mu_max = min(cell["mus"]), max(cell["mus"])
    b1_min, b1_max = min(cell["b1s"]), max(cell["b1s"])
    return (f"μ={mean_mu:5.2f} [{mu_min}-{mu_max}]  "
            f"β₁={mean_b1:5.2f} [{b1_min}-{b1_max}]  "
            f"agree={agree_pct:5.1f}%  thin={thin_pct:5.1f}%")


def main():
    print(f"{'Grid':<6}{'Loops':<7}{'Voxels':<8}{'Packing':<9} Result")
    print("-" * 90)

    configurations = []
    # 7³ baseline (matches current cross_validate_corpus.py Bridge settings)
    for loops, voxels in [(1, 12), (2, 18), (4, 25)]:
        configurations.append((7, loops, voxels, "medium"))
        configurations.append((7, loops, voxels, "dense"))
    # 9³ scale
    for loops, voxels in [(1, 20), (2, 30), (3, 40), (4, 50)]:
        configurations.append((9, loops, voxels, "medium"))
        configurations.append((9, loops, voxels, "sparse"))
    # 11³ scale
    for loops, voxels in [(1, 25), (2, 40), (3, 55), (4, 70)]:
        configurations.append((11, loops, voxels, "medium"))
        configurations.append((11, loops, voxels, "sparse"))
    # 13³ (do we really need this big?)
    for loops, voxels in [(2, 50), (4, 90)]:
        configurations.append((13, loops, voxels, "medium"))
        configurations.append((13, loops, voxels, "sparse"))

    results = {}
    for grid, loops, voxels, packing in configurations:
        cell = run_cell(grid, loops, voxels, packing, n_seeds=15)
        results[(grid, loops, voxels, packing)] = cell
        summary = summarize(cell)
        target_hit = ""
        if cell["n"] > 0:
            mean_b1 = sum(cell["b1s"]) / cell["n"]
            if abs(mean_b1 - loops) < 0.3:
                target_hit = " ← β₁ hits target!"
            elif mean_b1 > loops * 0.7:
                target_hit = " ← β₁ close"
        print(f"{grid:<6}{loops:<7}{voxels:<8}{packing:<9} {summary}{target_hit}")

    # Save raw results
    import json
    out = {}
    for key, cell in results.items():
        k = f"{key[0]}³/loops={key[1]}/voxels={key[2]}/{key[3]}"
        out[k] = {"n": cell["n"], "failures": cell["failures"],
                  "mus": cell["mus"], "b1s": cell["b1s"],
                  "thins": [bool(t) for t in cell["thins"]],
                  "agrees": [bool(a) for a in cell["agrees"]]}
    with open("/home/user/workspace/smoke_test_larger_grids.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n✅ Saved to /home/user/workspace/smoke_test_larger_grids.json")


if __name__ == "__main__":
    main()
