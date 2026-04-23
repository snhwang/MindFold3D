"""
Generation Fidelity Benchmarking Tool — MindFold 3D

Copyright (c) 2024-2026 Scott N. Hwang, Parviz Safadel. All rights reserved.

Systematically generates shapes across all difficulty combinations, measures
their actual geometric features, and reports fidelity statistics: what fraction
of shapes land in the intended difficulty tier for each feature.

This is an offline characterization tool for publication and patent enablement.
It does NOT gate individual trials — it characterizes the generator's coverage
of the difficulty space.

Usage:
    python benchmark_fidelity.py               # defaults: 50 shapes per cell
    python benchmark_fidelity.py --n 100       # 100 shapes per cell
    python benchmark_fidelity.py --n 20 --json results.json
    python benchmark_fidelity.py --dims spatial_form structural_complexity
"""

import argparse
import datetime
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from itertools import product
from typing import Dict, List, Tuple

from cognitive_mapping import (
    SHAPE_DIMENSIONS,
    get_skeleton_spec,
    reverse_map_cognitive_profile,
)
from skeleton_generation import generate_shape_skeleton
from shape_features import ShapeFeatureSet


LEVELS = ["low", "medium", "high", "expert"]

# Features belonging to each dimension (ordered as in SHAPE_DIMENSIONS)
DIMENSION_FEATURES: Dict[str, List[str]] = {
    dim_name: list(dim_cfg["features"].keys())
    for dim_name, dim_cfg in SHAPE_DIMENSIONS.items()
}

ALL_DIMS = list(SHAPE_DIMENSIONS.keys())


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

def _generate_and_measure(dim_name: str, level: str, n: int) -> List[Dict]:
    """Generate n shapes targeting (dim_name=level), measure all features."""
    results = []
    shape_difficulties = {d: "medium" for d in ALL_DIMS}
    shape_difficulties[dim_name] = level

    for _ in range(n):
        try:
            spec = get_skeleton_spec(
                shape_difficulties=shape_difficulties,
                task_difficulties={},
            )
            result = generate_shape_skeleton(spec)
            features = result.get("features", {})

            # Also run reverse classification
            sfs = ShapeFeatureSet(**{
                k: v for k, v in features.items()
                if k in ShapeFeatureSet.model_fields and v is not None
            })
            profile = reverse_map_cognitive_profile(
                sfs,
                intended_difficulties={dim_name: level},
            )

            classified_level = profile.shape_dimensions.get(dim_name)
            results.append({
                "features": features,
                "classified_level": classified_level.level if classified_level else "unknown",
                "fidelity": profile.overall_fidelity,
            })
        except Exception as e:
            results.append({"error": str(e)})

    return results


def _feature_stats(results: List[Dict], feature_name: str) -> Dict:
    """Compute mean, std, and tier distribution for one feature across results."""
    values = [
        r["features"].get(feature_name)
        for r in results
        if "features" in r and r["features"].get(feature_name) is not None
    ]
    if not values:
        return {"mean": None, "std": None, "n": 0}

    return {
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": len(values),
    }


def _binomial_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96 if confidence == 0.95 else 2.576  # z for 95% or 99%
    p_hat = k / n
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
    return (round(max(0.0, center - spread), 4), round(min(1.0, center + spread), 4))


def _fidelity_rate(results: List[Dict], dim_name: str, intended_level: str) -> Dict:
    """Compute dimension-level fidelity: fraction of shapes that classified correctly."""
    classified = [r["classified_level"] for r in results if "classified_level" in r]
    if not classified:
        return {"fidelity_rate": None, "n": 0, "distribution": {}, "ci_95": None}

    distribution: Dict[str, int] = defaultdict(int)
    for level in classified:
        distribution[level] += 1

    n = len(classified)
    correct = distribution.get(intended_level, 0)
    ci = _binomial_ci(correct, n, 0.95)
    return {
        "fidelity_rate": round(correct / n, 4),
        "n": n,
        "distribution": {k: round(v / n, 4) for k, v in sorted(distribution.items())},
        "ci_95": list(ci),
        "n_correct": correct,
    }


def run_benchmark(
    dims: List[str],
    n_per_cell: int,
) -> Dict:
    """Run full benchmark across specified dimensions and all levels."""
    all_results = {}

    for dim_name in dims:
        if dim_name not in SHAPE_DIMENSIONS:
            print(f"  WARNING: Unknown dimension '{dim_name}', skipping.", file=sys.stderr)
            continue

        all_results[dim_name] = {}
        features_in_dim = DIMENSION_FEATURES[dim_name]

        for level in LEVELS:
            print(f"  Generating {n_per_cell} shapes: {dim_name}={level}...", end=" ", flush=True)
            results = _generate_and_measure(dim_name, level, n_per_cell)
            successes = [r for r in results if "error" not in r]
            errors = len(results) - len(successes)
            print(f"{len(successes)} ok{f', {errors} errors' if errors else ''}")

            cell = {
                "intended_level": level,
                "n_generated": len(successes),
                "n_errors": errors,
                "dimension_fidelity": _fidelity_rate(successes, dim_name, level),
                "features": {},
            }

            for feat in features_in_dim:
                cell["features"][feat] = _feature_stats(successes, feat)

            all_results[dim_name][level] = cell

    return all_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _tier_ranges(dim_name: str, feature_name: str) -> str:
    feat_cfg = SHAPE_DIMENSIONS[dim_name]["features"][feature_name]
    parts = []
    for lvl in LEVELS:
        r = feat_cfg.get(lvl)
        if r is None:
            continue
        if isinstance(r, tuple):
            parts.append(f"{lvl}:[{r[0]},{r[1]}]")
        else:
            parts.append(f"{lvl}:{r}")
    return "  ".join(parts)


def print_report(results: Dict, n_per_cell: int, run_metadata: Dict = None) -> None:
    print()
    print("=" * 78)
    print("  MINDFOLD 3D — GENERATION FIDELITY CHARACTERIZATION")
    print(f"  N = {n_per_cell} shapes per difficulty cell")
    if run_metadata:
        print(f"  Date: {run_metadata.get('date', 'N/A')}")
        print(f"  Total shapes: {run_metadata.get('total_shapes', 'N/A')}")
        print(f"  Total time: {run_metadata.get('total_time_seconds', 'N/A')}s")
    print("=" * 78)

    for dim_name, dim_results in results.items():
        dim_cfg = SHAPE_DIMENSIONS[dim_name]
        print(f"\n{'-' * 78}")
        print(f"  DIMENSION: {dim_name.upper()}")
        print(f"  {dim_cfg['description']}")
        print(f"{'-' * 78}")

        # Dimension-level fidelity summary table
        print(f"\n  Dimension-level fidelity (fraction of shapes classifying to intended tier):\n")
        print(f"  {'Intended':<12} {'Fidelity':>10} {'95% CI':>16}  {'Distribution'}")
        print(f"  {'-'*12} {'-'*10} {'-'*16}  {'-'*40}")
        for level in LEVELS:
            cell = dim_results.get(level, {})
            fid = cell.get("dimension_fidelity", {})
            rate = fid.get("fidelity_rate")
            ci = fid.get("ci_95")
            dist = fid.get("distribution", {})
            rate_str = f"{rate:.1%}" if rate is not None else "N/A"
            ci_str = f"[{ci[0]:.1%}, {ci[1]:.1%}]" if ci else "N/A"
            dist_str = "  ".join(f"{k}:{v:.1%}" for k, v in dist.items())
            print(f"  {level:<12} {rate_str:>10} {ci_str:>16}  {dist_str}")
        print()

        # Per-feature measurement summary
        features_in_dim = DIMENSION_FEATURES[dim_name]
        for feat in features_in_dim:
            print(f"  Feature: {feat}")
            print(f"  Defined ranges: {_tier_ranges(dim_name, feat)}")
            print(f"  {'Intended':<12} {'Mean':>10} {'±Std':>10} {'Min':>10} {'Max':>10} {'N':>6}")
            print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
            for level in LEVELS:
                cell = dim_results.get(level, {})
                fstats = cell.get("features", {}).get(feat, {})
                mean = fstats.get("mean")
                std = fstats.get("std")
                mn = fstats.get("min")
                mx = fstats.get("max")
                n = fstats.get("n", 0)
                if mean is not None:
                    print(f"  {level:<12} {mean:>10.4f} {std:>10.4f} {mn:>10.4f} {mx:>10.4f} {n:>6}")
                else:
                    print(f"  {level:<12} {'N/A':>10}")
            print()

    # Overall summary
    print(f"{'-' * 78}")
    print(f"  OVERALL SUMMARY")
    print(f"{'-' * 78}")
    all_rates = []
    for dim_name, dim_results in results.items():
        rates = []
        for level in LEVELS:
            cell = dim_results.get(level, {})
            r = cell.get("dimension_fidelity", {}).get("fidelity_rate")
            if r is not None:
                rates.append(r)
                all_rates.append(r)
        if rates:
            print(f"  {dim_name:<28}  mean fidelity: {statistics.mean(rates):.1%}  "
                  f"range: {min(rates):.1%}–{max(rates):.1%}")
    if all_rates:
        print(f"\n  {'Overall (all dimensions)':<28}  mean fidelity: {statistics.mean(all_rates):.1%}  "
              f"range: {min(all_rates):.1%}–{max(all_rates):.1%}")
    print()


def print_patent_table(results: Dict) -> None:
    """Print a compact table suitable for inclusion in patent specification or paper."""
    print()
    print("  TABLE: Generation Fidelity by Dimension and Difficulty Level")
    print("  (Fraction of N generated shapes classifying to intended tier)")
    print()
    header = f"  {'Dimension':<24} {'Feature(s)':<30} {'Low':>7} {'Med':>7} {'High':>7} {'Expert':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for dim_name, dim_results in results.items():
        features_in_dim = DIMENSION_FEATURES[dim_name]
        feat_str = ", ".join(features_in_dim)
        if len(feat_str) > 28:
            feat_str = feat_str[:25] + "..."

        rates = []
        for level in LEVELS:
            cell = dim_results.get(level, {})
            r = cell.get("dimension_fidelity", {}).get("fidelity_rate")
            rates.append(f"{r:.0%}" if r is not None else "N/A")

        print(f"  {dim_name:<24} {feat_str:<30} {rates[0]:>7} {rates[1]:>7} {rates[2]:>7} {rates[3]:>7}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MindFold 3D generation fidelity benchmarking tool"
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Number of shapes to generate per difficulty cell (default: 50)"
    )
    parser.add_argument(
        "--dims", nargs="+", default=ALL_DIMS,
        help=f"Dimensions to benchmark (default: all). Choices: {ALL_DIMS}"
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Path to write full results as JSON (optional)"
    )
    parser.add_argument(
        "--patent-table", action="store_true",
        help="Print a compact table formatted for patent/paper inclusion"
    )
    args = parser.parse_args()

    total_shapes = args.n * len(args.dims) * len(LEVELS)
    print(f"\nMindFold 3D — Generation Fidelity Benchmark")
    print(f"Dimensions: {args.dims}")
    print(f"Shapes per cell: {args.n}")
    print(f"Total shapes: {total_shapes}\n")

    start_time = time.time()
    results = run_benchmark(dims=args.dims, n_per_cell=args.n)
    elapsed = round(time.time() - start_time, 1)

    run_metadata = {
        "date": datetime.datetime.now().isoformat(),
        "n_per_cell": args.n,
        "dimensions": args.dims,
        "levels": LEVELS,
        "total_shapes": total_shapes,
        "total_time_seconds": elapsed,
    }

    # Save JSON first to preserve results even if report printing fails
    if args.json:
        output = {
            "metadata": run_metadata,
            "results": results,
        }
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Full results written to: {args.json}\n")

    print_report(results, args.n, run_metadata)

    if args.patent_table:
        print_patent_table(results)


if __name__ == "__main__":
    main()
