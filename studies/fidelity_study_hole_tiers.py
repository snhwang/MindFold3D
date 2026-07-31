"""
N=200 fidelity study for hole-tier generation — PRODUCTION code path.

This study exercises the integrated production generator
(skeleton_generation.HoleMarkedLoopSkeleton) exactly as
generate_shape_skeleton() drives it:

    per attempt:  gen.generate(spec)      # template -> hole-aware fill
                                          # -> accidental-cavity FILL closure
                  gen._validate(spec)     # single component + cubical beta_1 == target

with per-attempt RNG reseeding and a 20-attempt budget, matching the
production loop in generate_shape_skeleton(). The only difference is that
seeds are deterministic here (seed * 100000 + attempt) for reproducibility,
and validation failures are categorized (b1_high / b1_low / disconnected).

Production semantics captured by this study:
  - Accidental cavities are FILLED, not rejected, so emitted voxel count can
    exceed the target; the per-cell voxel-drift statistics quantify this.
  - The emitted-set guarantee is beta_1 == target on every emitted shape
    (validation + regeneration); exhaustion raises in production
    (HoleTierGenerationError) and is recorded as a cell failure here.

For each (beta_1_target, target_voxels) cell:
  - Attempt to generate 200 shapes (regeneration on validation failure)
  - Track first-pass acceptance rate, mean attempts, failure modes
  - Track per-shape mu (circuit rank), beta_1, voxel count (incl. drift)

Outputs hole_tier_fidelity_N200.json / .csv next to this script.
"""

import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mindfold3d.cognitive_mapping import SkeletonSpec, HOLE_TIER_GRID
from mindfold3d.skeleton_generation import HoleMarkedLoopSkeleton, _count_components
from mindfold3d.shape_generation import _calculate_circuit_rank as calc_mu
from mindfold3d.topology_extras import cubical_betti_1

N_ATTEMPTS_MAX = 20   # matches the production budget in generate_shape_skeleton
N_SHAPES_PER_CELL = 200


def _make_spec(num_loops: int, voxel_count: int) -> SkeletonSpec:
    """Production hole-tier spec (mirrors cognitive_mapping.get_hole_tier_spec,
    but with the study's swept voxel count uncapped)."""
    return SkeletonSpec(
        archetype="hole",
        voxel_count=voxel_count,
        grid_size=HOLE_TIER_GRID,
        num_branches=max(3, num_loops + 2),
        num_loops=num_loops,
        num_components=1,
        direction_spread="planar",
        planarity="medium",
        packing="medium",
        target_b1=num_loops,
        skip_optimizer=True,
    )


def generate_with_validation(
    num_loops: int,
    voxel_count: int,
    seed: int,
    max_attempts: int = N_ATTEMPTS_MAX,
) -> Dict[str, Any]:
    """Generate one certified hole-tier shape via the production class.

    Replicates the generate_shape_skeleton() hole-tier loop with
    deterministic seeds and categorized validation failures.
    """
    grid = HOLE_TIER_GRID
    spec = _make_spec(num_loops, voxel_count)
    failure_modes = []

    for attempt in range(1, max_attempts + 1):
        rng = random.Random(seed * 100000 + attempt)
        gen = HoleMarkedLoopSkeleton(grid, rng=rng)
        # Production sequence: template -> hole-aware fill -> cavity-fill closure
        gen.generate(spec)

        # Production _validate, split out so failures can be categorized.
        n_comp = _count_components(gen.voxels, grid)
        if n_comp != 1:
            failure_modes.append("disconnected")
            continue
        b1 = cubical_betti_1(gen.voxels)
        if b1 > num_loops:
            failure_modes.append("b1_high")
            continue
        if b1 < num_loops:
            failure_modes.append("b1_low")
            continue

        return {
            "success": True,
            "attempts": attempt,
            "mu": calc_mu(gen.voxels, grid),
            "b1": b1,
            "n_voxels": len(gen.voxels),
            "voxel_drift": len(gen.voxels) - voxel_count,
            "n_components": n_comp,
            "failure_modes": failure_modes,
        }

    return {
        "success": False,
        "attempts": max_attempts,
        "mu": None,
        "b1": None,
        "n_voxels": None,
        "voxel_drift": None,
        "failure_modes": failure_modes,
    }


def study_cell(num_loops: int, voxel_count: int, n_shapes: int = N_SHAPES_PER_CELL) -> Dict[str, Any]:
    """Run the fidelity study for one (beta_1, voxel_count) cell."""
    t0 = time.time()
    results = [
        generate_with_validation(num_loops, voxel_count, seed=i)
        for i in range(n_shapes)
    ]

    successes = [r for r in results if r["success"]]
    n_success = len(successes)
    n_fail = n_shapes - n_success
    first_pass = sum(1 for r in successes if r["attempts"] == 1)
    mean_attempts = (
        statistics.mean(r["attempts"] for r in successes) if successes else float("nan")
    )

    all_failure_modes = []
    for r in results:
        all_failure_modes.extend(r["failure_modes"])
    fm_counts = Counter(all_failure_modes)

    def summ(xs):
        if not xs:
            return {"mean": None, "min": None, "max": None, "std": None}
        return {
            "mean": statistics.mean(xs),
            "min": min(xs),
            "max": max(xs),
            "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        }

    return {
        "grid_size": HOLE_TIER_GRID[0],
        "target_b1": num_loops,
        "target_voxels": voxel_count,
        "n_shapes_requested": n_shapes,
        "n_success": n_success,
        "n_fail_exceeded_max_attempts": n_fail,
        "first_pass_rate": first_pass / n_shapes,
        "mean_attempts_when_successful": mean_attempts,
        "failure_mode_counts": dict(fm_counts),
        "mu_stats": summ([r["mu"] for r in successes]),
        "b1_stats": summ([r["b1"] for r in successes]),
        "voxel_count_stats": summ([r["n_voxels"] for r in successes]),
        "voxel_drift_stats": summ([r["voxel_drift"] for r in successes]),
        "elapsed_seconds": time.time() - t0,
    }


def main(n_shapes: int = N_SHAPES_PER_CELL):
    tier_configs = [
        # (beta_1_target, target_voxels)
        (1, 8), (1, 15), (1, 20), (1, 25),
        (2, 14), (2, 20), (2, 25), (2, 30),
        (3, 20), (3, 25), (3, 30),
        (4, 24), (4, 30), (4, 35),
    ]

    print("=" * 76)
    print(f"HOLE-TIER FIDELITY STUDY (N={n_shapes} per cell) — production code path")
    print(f"Grid: {HOLE_TIER_GRID} | Max attempts per shape: {N_ATTEMPTS_MAX}")
    print("=" * 76)

    all_results = []
    for num_loops, voxels in tier_configs:
        print(f"\n>>> beta_1={num_loops}, target_voxels={voxels} <<<", flush=True)
        result = study_cell(num_loops, voxels, n_shapes=n_shapes)
        all_results.append(result)
        print(f"    success: {result['n_success']}/{result['n_shapes_requested']}", flush=True)
        print(f"    first-pass acceptance: {100 * result['first_pass_rate']:.1f}%", flush=True)
        print(f"    mean attempts (when successful): {result['mean_attempts_when_successful']:.3f}", flush=True)
        ms, bs, ds = result['mu_stats'], result['b1_stats'], result['voxel_drift_stats']
        print(f"    mu: mean={ms['mean']:.2f}, range=[{ms['min']}, {ms['max']}]", flush=True)
        print(f"    beta_1: mean={bs['mean']:.2f}, range=[{bs['min']}, {bs['max']}]", flush=True)
        print(f"    voxel drift (template overshoot + cavity closure): mean={ds['mean']:.2f}, max={ds['max']}", flush=True)
        print(f"    failure modes: {result['failure_mode_counts']}", flush=True)
        print(f"    elapsed: {result['elapsed_seconds']:.1f}s", flush=True)

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_json = out_dir / "hole_tier_fidelity_N200.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_json}")

    out_csv = out_dir / "hole_tier_fidelity_N200.csv"
    with open(out_csv, "w") as f:
        f.write("target_b1,target_voxels,n_success,first_pass_pct,mean_attempts,"
                "mu_mean,mu_min,mu_max,b1_mean,b1_min,b1_max,"
                "voxel_count_mean,voxel_drift_mean,voxel_drift_max,elapsed_s\n")
        for r in all_results:
            mus, b1s = r['mu_stats'], r['b1_stats']
            vs, ds = r['voxel_count_stats'], r['voxel_drift_stats']
            f.write(f"{r['target_b1']},{r['target_voxels']},{r['n_success']},"
                    f"{100 * r['first_pass_rate']:.2f},"
                    f"{r['mean_attempts_when_successful']:.3f},"
                    f"{mus['mean']:.2f},{mus['min']},{mus['max']},"
                    f"{b1s['mean']:.2f},{b1s['min']},{b1s['max']},"
                    f"{vs['mean']:.2f},{ds['mean']:.2f},{ds['max']},"
                    f"{r['elapsed_seconds']:.2f}\n")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_SHAPES_PER_CELL
    main(n)
