"""
Generation Fidelity Study for MindFold 3D.

Validates that the procedural generator produces shapes matching target
feature specifications across the full difficulty parameter space.

Protocol (from paper Section 8.2):
  1. For each Layer 1 dimension, set difficulty to Low/Medium/High while
     holding others at Medium. 3 dims x 3 levels = 9 + 1 baseline = 10.
  2. Add extremes: all-Low, all-High, and selected cross-combinations.
  3. Generate N shapes per condition using generate_shape_advanced().
  4. Compute target fidelity, feature correlations, achievable space,
     and generation time.

Usage:
    python generation_fidelity_study.py [--shapes-per-condition 100] [--output-dir results]
"""

import argparse
import json
import os
import sys
import time
import statistics
from collections import defaultdict
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mindfold3d.cognitive_mapping import (
    SHAPE_DIMENSIONS,
    get_difficulty_spec,
    get_scored_feature_names,
)
from mindfold3d.shape_features import ShapeFeatureSet
from mindfold3d.shape_generation import generate_shape_advanced


# ── Scored features the generator actively optimizes ──────────────────────
SCORED_FEATURES = get_scored_feature_names()

# ── Voxel counts to test (paper uses 8-12 range) ─────────────────────────
VOXEL_COUNTS = [8, 10, 12]


# ═════════════════════════════════════════════════════════════════════════════
# Condition definitions
# ═════════════════════════════════════════════════════════════════════════════

def build_conditions() -> List[Dict[str, Any]]:
    """Build the full set of experimental conditions."""
    dims = list(SHAPE_DIMENSIONS.keys())
    levels = ["low", "medium", "high"]
    conditions = []

    # ── Baseline: all medium ──
    conditions.append({
        "name": "baseline_all_medium",
        "shape_difficulties": {d: "medium" for d in dims},
    })

    # ── One-at-a-time: vary each dimension across low/medium/high ──
    for dim in dims:
        for level in levels:
            diffs = {d: "medium" for d in dims}
            diffs[dim] = level
            conditions.append({
                "name": f"{dim}_{level}",
                "shape_difficulties": diffs,
            })

    # ── Extremes ──
    conditions.append({
        "name": "all_low",
        "shape_difficulties": {d: "low" for d in dims},
    })
    conditions.append({
        "name": "all_high",
        "shape_difficulties": {d: "high" for d in dims},
    })

    # ── Cross-combinations (selected pairs at opposing extremes) ──
    cross_pairs = [
        ("spatial_form", "high", "structural_complexity", "low"),
        ("spatial_form", "low", "structural_complexity", "high"),
        ("spatial_form", "high", "spatial_density", "low"),
        ("spatial_form", "low", "spatial_density", "high"),
        ("structural_complexity", "high", "spatial_density", "low"),
        ("structural_complexity", "low", "spatial_density", "high"),
    ]
    for d1, l1, d2, l2 in cross_pairs:
        diffs = {d: "medium" for d in dims}
        diffs[d1] = l1
        diffs[d2] = l2
        conditions.append({
            "name": f"{d1}_{l1}_x_{d2}_{l2}",
            "shape_difficulties": diffs,
        })

    return conditions


# ═════════════════════════════════════════════════════════════════════════════
# Target range extraction
# ═════════════════════════════════════════════════════════════════════════════

def get_target_ranges(shape_difficulties: Dict[str, str]) -> Dict[str, Tuple[float, float]]:
    """
    For a given difficulty setting, return the (min, max) target range
    for each scored feature. Returns None for features without a range
    at that difficulty level.
    """
    ranges = {}
    defaults = {dim: "medium" for dim in SHAPE_DIMENSIONS}
    defaults.update(shape_difficulties)

    for dim_name, dim_config in SHAPE_DIMENSIONS.items():
        difficulty = defaults[dim_name]
        for feat_name, feat_config in dim_config["features"].items():
            spec = feat_config.get(difficulty)
            if spec is None:
                continue
            if isinstance(spec, tuple) and len(spec) == 2:
                ranges[feat_name] = (float(spec[0]), float(spec[1]))
            else:
                # Scalar target — treat as exact with small tolerance
                val = float(spec)
                ranges[feat_name] = (val, val)
    return ranges


# ═════════════════════════════════════════════════════════════════════════════
# Single condition runner
# ═════════════════════════════════════════════════════════════════════════════

def run_condition(
    condition: Dict[str, Any],
    n_shapes: int,
    voxel_count: int,
) -> Dict[str, Any]:
    """Generate n_shapes for one condition and collect feature data."""
    task_diffs = {
        "mental_rotation": "low",
        "mirror_discrimination": "low",
        "working_memory": "low",
    }

    spec = get_difficulty_spec(
        condition["shape_difficulties"],
        task_diffs,
        target_voxel_count=voxel_count,
    )
    target_sfs = spec.shape_features
    target_sfs.number_of_components = 1  # always single-component for study

    target_ranges = get_target_ranges(condition["shape_difficulties"])
    target_midpoints = {}
    for feat, (lo, hi) in target_ranges.items():
        target_midpoints[feat] = (lo + hi) / 2.0

    results = []
    gen_times = []

    for i in range(n_shapes):
        t0 = time.perf_counter()
        shape_data = generate_shape_advanced(
            target_sfs,
            max_iterations=75 * target_sfs.voxel_count,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        gen_times.append(elapsed_ms)

        feats = shape_data["features"]
        results.append(feats)

        # Progress indicator (one dot per 10 shapes)
        if (i + 1) % 10 == 0:
            print(".", end="", flush=True)

    print(f" ({len(results)} shapes, avg {statistics.mean(gen_times):.0f}ms)")

    return {
        "condition": condition["name"],
        "voxel_count": voxel_count,
        "shape_difficulties": condition["shape_difficulties"],
        "target_ranges": {k: list(v) for k, v in target_ranges.items()},
        "target_midpoints": target_midpoints,
        "n_generated": len(results),
        "generation_times_ms": {
            "mean": round(statistics.mean(gen_times), 1),
            "median": round(statistics.median(gen_times), 1),
            "stdev": round(statistics.stdev(gen_times), 1) if len(gen_times) > 1 else 0,
            "max": round(max(gen_times), 1),
            "min": round(min(gen_times), 1),
        },
        "shape_features": results,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Analysis functions
# ═════════════════════════════════════════════════════════════════════════════

def analyze_target_fidelity(condition_result: Dict) -> Dict[str, Any]:
    """
    For each scored feature, compute:
      - mean achieved value
      - std of achieved values
      - mean absolute deviation from target midpoint
      - % of shapes within target range
    """
    features_data = condition_result["shape_features"]
    target_ranges = {k: tuple(v) for k, v in condition_result["target_ranges"].items()}
    target_midpoints = condition_result["target_midpoints"]

    fidelity = {}
    for feat in SCORED_FEATURES:
        if feat not in target_ranges:
            continue

        lo, hi = target_ranges[feat]
        mid = target_midpoints[feat]
        values = []
        for shape in features_data:
            val = shape.get(feat)
            if val is not None:
                values.append(float(val))

        if not values:
            fidelity[feat] = {"error": "no_data"}
            continue

        achieved_mean = statistics.mean(values)
        achieved_std = statistics.stdev(values) if len(values) > 1 else 0.0

        # Mean absolute deviation from target midpoint
        mad = statistics.mean([abs(v - mid) for v in values])

        # % within target range (with tolerance for scalar targets)
        tolerance = 0.5 if lo == hi else 0.0
        in_range = sum(1 for v in values if (lo - tolerance) <= v <= (hi + tolerance))
        pct_in_range = in_range / len(values) * 100

        fidelity[feat] = {
            "target_range": [lo, hi],
            "target_midpoint": round(mid, 4),
            "achieved_mean": round(achieved_mean, 4),
            "achieved_std": round(achieved_std, 4),
            "mean_absolute_deviation": round(mad, 4),
            "pct_in_range": round(pct_in_range, 1),
            "n_samples": len(values),
        }

    return fidelity


def compute_feature_correlation_matrix(all_shapes: List[Dict]) -> Dict[str, Any]:
    """
    Compute Pearson correlation matrix between all scored features
    across all generated shapes.
    """
    # Collect feature vectors
    feature_vectors = {f: [] for f in SCORED_FEATURES}
    for shape in all_shapes:
        for feat in SCORED_FEATURES:
            val = shape.get(feat)
            feature_vectors[feat].append(float(val) if val is not None else np.nan)

    # Build matrix, skipping features with all-NaN
    valid_features = []
    vectors = []
    for feat in SCORED_FEATURES:
        arr = np.array(feature_vectors[feat], dtype=float)
        if np.all(np.isnan(arr)):
            continue
        # Replace NaN with column mean for correlation
        col_mean = np.nanmean(arr)
        arr = np.where(np.isnan(arr), col_mean, arr)
        valid_features.append(feat)
        vectors.append(arr)

    if len(vectors) < 2:
        return {"error": "insufficient_features"}

    matrix = np.corrcoef(vectors)

    # Identify strongly correlated and independent pairs
    strongly_correlated = []
    independent = []
    for i in range(len(valid_features)):
        for j in range(i + 1, len(valid_features)):
            r = matrix[i][j]
            pair = (valid_features[i], valid_features[j])
            if abs(r) > 0.5:
                strongly_correlated.append({"pair": list(pair), "r": round(float(r), 3)})
            elif abs(r) < 0.3:
                independent.append({"pair": list(pair), "r": round(float(r), 3)})

    # Sort by |r| descending
    strongly_correlated.sort(key=lambda x: abs(x["r"]), reverse=True)

    return {
        "features": valid_features,
        "matrix": [[round(float(x), 3) for x in row] for row in matrix],
        "strongly_correlated_pairs": strongly_correlated,
        "independent_pairs": independent,
        "n_shapes": len(all_shapes),
    }


def compute_achievable_space(all_shapes: List[Dict]) -> Dict[str, Any]:
    """
    Report the min/max/mean/std of each scored feature across all shapes
    to characterize the achievable feature space.
    """
    space = {}
    for feat in SCORED_FEATURES:
        values = [float(s[feat]) for s in all_shapes if s.get(feat) is not None]
        if not values:
            continue
        space[feat] = {
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "mean": round(statistics.mean(values), 4),
            "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    return space


def compute_emergent_distributions(all_shapes: List[Dict]) -> Dict[str, Any]:
    """
    Report distributions of non-scored (emergent) features across all shapes.
    """
    emergent_features = [
        "surface_area", "bounding_box_ratio", "dominant_axis",
        "largest_component_ratio",
    ]
    distributions = {}
    for feat in emergent_features:
        values = [s[feat] for s in all_shapes if s.get(feat) is not None]
        if not values:
            continue
        if isinstance(values[0], (int, float)):
            numeric = [float(v) for v in values]
            distributions[feat] = {
                "min": round(min(numeric), 4),
                "max": round(max(numeric), 4),
                "mean": round(statistics.mean(numeric), 4),
                "std": round(statistics.stdev(numeric), 4) if len(numeric) > 1 else 0.0,
                "n": len(numeric),
            }
        elif isinstance(values[0], str):
            # Categorical — count occurrences
            counts = defaultdict(int)
            for v in values:
                counts[v] += 1
            distributions[feat] = {
                "counts": dict(counts),
                "n": len(values),
            }
    return distributions


# ═════════════════════════════════════════════════════════════════════════════
# Summary report (plain text)
# ═════════════════════════════════════════════════════════════════════════════

def generate_text_report(analysis: Dict) -> str:
    """Produce a human-readable summary of the study results."""
    lines = []
    lines.append("=" * 72)
    lines.append("GENERATION FIDELITY STUDY — RESULTS SUMMARY")
    lines.append("MindFold 3D")
    lines.append("=" * 72)
    lines.append("")

    meta = analysis["metadata"]
    lines.append(f"Shapes per condition:  {meta['shapes_per_condition']}")
    lines.append(f"Voxel counts tested:   {meta['voxel_counts']}")
    lines.append(f"Total conditions:      {meta['total_conditions']}")
    lines.append(f"Total shapes:          {meta['total_shapes']}")
    lines.append(f"Total time:            {meta['total_time_seconds']:.1f}s")
    lines.append("")

    # ── Per-condition fidelity ──
    lines.append("-" * 72)
    lines.append("TARGET FIDELITY BY CONDITION")
    lines.append("-" * 72)
    lines.append("")

    for cond_key, fidelity in sorted(analysis["per_condition_fidelity"].items()):
        lines.append(f"  {cond_key}:")
        for feat, stats in fidelity.items():
            if "error" in stats:
                continue
            rng = stats["target_range"]
            lines.append(
                f"    {feat:25s}  target=[{rng[0]:.2f}, {rng[1]:.2f}]  "
                f"achieved={stats['achieved_mean']:.3f} +/- {stats['achieved_std']:.3f}  "
                f"MAD={stats['mean_absolute_deviation']:.3f}  "
                f"in-range={stats['pct_in_range']:.0f}%"
            )
        lines.append("")

    # ── Correlation highlights ──
    lines.append("-" * 72)
    lines.append("FEATURE CORRELATIONS (pooled across all shapes)")
    lines.append("-" * 72)
    lines.append("")

    corr = analysis["correlation_matrix"]
    if "error" not in corr:
        lines.append("  Strongly correlated (|r| > 0.5):")
        if corr["strongly_correlated_pairs"]:
            for p in corr["strongly_correlated_pairs"]:
                lines.append(f"    {p['pair'][0]:25s} <-> {p['pair'][1]:25s}  r={p['r']:+.3f}")
        else:
            lines.append("    (none)")
        lines.append("")

        lines.append("  Independent (|r| < 0.3):")
        if corr["independent_pairs"]:
            for p in corr["independent_pairs"]:
                lines.append(f"    {p['pair'][0]:25s} <-> {p['pair'][1]:25s}  r={p['r']:+.3f}")
        else:
            lines.append("    (none)")
        lines.append("")

    # ── Achievable space ──
    lines.append("-" * 72)
    lines.append("ACHIEVABLE FEATURE SPACE")
    lines.append("-" * 72)
    lines.append("")

    for feat, stats in analysis["achievable_space"].items():
        lines.append(
            f"  {feat:25s}  range=[{stats['min']:.3f}, {stats['max']:.3f}]  "
            f"mean={stats['mean']:.3f}  std={stats['std']:.3f}"
        )
    lines.append("")

    # ── Generation time ──
    lines.append("-" * 72)
    lines.append("GENERATION TIME (ms per shape)")
    lines.append("-" * 72)
    lines.append("")

    gen_summary = analysis["generation_time_summary"]
    lines.append(f"  Overall mean:   {gen_summary['overall_mean_ms']:.1f} ms")
    lines.append(f"  Overall median: {gen_summary['overall_median_ms']:.1f} ms")
    lines.append(f"  Overall max:    {gen_summary['overall_max_ms']:.1f} ms")
    lines.append("")
    lines.append("  By condition:")
    for cond, t in sorted(gen_summary["by_condition"].items()):
        lines.append(f"    {cond:45s}  mean={t['mean']:.0f}ms  max={t['max']:.0f}ms")
    lines.append("")

    # ── Emergent distributions ──
    lines.append("-" * 72)
    lines.append("EMERGENT FEATURE DISTRIBUTIONS")
    lines.append("-" * 72)
    lines.append("")

    for feat, stats in analysis["emergent_distributions"].items():
        if "counts" in stats:
            lines.append(f"  {feat}: {stats['counts']}")
        else:
            lines.append(
                f"  {feat:25s}  range=[{stats['min']:.3f}, {stats['max']:.3f}]  "
                f"mean={stats['mean']:.3f}  std={stats['std']:.3f}"
            )
    lines.append("")
    lines.append("=" * 72)

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MindFold 3D Generation Fidelity Study")
    parser.add_argument(
        "--shapes-per-condition", type=int, default=100,
        help="Number of shapes to generate per condition (default: 100)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--voxel-counts", type=str, default="8,10,12",
        help="Comma-separated voxel counts to test (default: 8,10,12)",
    )
    args = parser.parse_args()

    n_shapes = args.shapes_per_condition
    output_dir = args.output_dir
    voxel_counts = [int(x.strip()) for x in args.voxel_counts.split(",")]

    os.makedirs(output_dir, exist_ok=True)

    conditions = build_conditions()
    total_runs = len(conditions) * len(voxel_counts)

    print(f"Generation Fidelity Study")
    print(f"  Conditions:          {len(conditions)}")
    print(f"  Voxel counts:        {voxel_counts}")
    print(f"  Shapes per condition: {n_shapes}")
    print(f"  Total shapes:        {total_runs * n_shapes}")
    print(f"  Output:              {output_dir}/")
    print()

    # ── Run all conditions ──
    all_condition_results = []
    all_shapes_pooled = []
    all_gen_times = []
    t_study_start = time.perf_counter()

    run_idx = 0
    for vc in voxel_counts:
        for cond in conditions:
            run_idx += 1
            label = f"vc={vc} | {cond['name']}"
            print(f"[{run_idx:3d}/{total_runs}] {label:55s}", end=" ", flush=True)

            result = run_condition(cond, n_shapes, vc)
            all_condition_results.append(result)
            all_shapes_pooled.extend(result["shape_features"])
            all_gen_times.append(result["generation_times_ms"])

    study_time = time.perf_counter() - t_study_start

    print(f"\nAll generation complete. Total time: {study_time:.1f}s")
    print(f"Analyzing results...")

    # ── Compute per-condition fidelity ──
    per_condition_fidelity = {}
    gen_time_by_condition = {}

    for result in all_condition_results:
        key = f"vc{result['voxel_count']}_{result['condition']}"
        fidelity = analyze_target_fidelity(result)
        per_condition_fidelity[key] = fidelity
        gen_time_by_condition[key] = result["generation_times_ms"]

    # ── Pooled analyses ──
    correlation = compute_feature_correlation_matrix(all_shapes_pooled)
    achievable = compute_achievable_space(all_shapes_pooled)
    emergent = compute_emergent_distributions(all_shapes_pooled)

    # ── Generation time summary ──
    all_means = [t["mean"] for t in all_gen_times]
    all_maxes = [t["max"] for t in all_gen_times]
    gen_time_summary = {
        "overall_mean_ms": round(statistics.mean(all_means), 1),
        "overall_median_ms": round(statistics.median(all_means), 1),
        "overall_max_ms": round(max(all_maxes), 1),
        "by_condition": gen_time_by_condition,
    }

    # ── Assemble final analysis ──
    analysis = {
        "metadata": {
            "shapes_per_condition": n_shapes,
            "voxel_counts": voxel_counts,
            "total_conditions": total_runs,
            "total_shapes": len(all_shapes_pooled),
            "total_time_seconds": round(study_time, 1),
            "scored_features": SCORED_FEATURES,
        },
        "per_condition_fidelity": per_condition_fidelity,
        "correlation_matrix": correlation,
        "achievable_space": achievable,
        "emergent_distributions": emergent,
        "generation_time_summary": gen_time_summary,
    }

    # ── Save raw data (all shapes per condition) ──
    raw_path = os.path.join(output_dir, "raw_generation_data.json")
    raw_export = []
    for r in all_condition_results:
        raw_export.append({
            "condition": r["condition"],
            "voxel_count": r["voxel_count"],
            "shape_difficulties": r["shape_difficulties"],
            "target_ranges": r["target_ranges"],
            "n_generated": r["n_generated"],
            "generation_times_ms": r["generation_times_ms"],
            "shape_features": r["shape_features"],
        })
    with open(raw_path, "w") as f:
        json.dump(raw_export, f, indent=2, default=str)
    print(f"  Raw data:    {raw_path}")

    # ── Save analysis ──
    analysis_path = os.path.join(output_dir, "fidelity_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"  Analysis:    {analysis_path}")

    # ── Save text report ──
    report = generate_text_report(analysis)
    report_path = os.path.join(output_dir, "fidelity_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report:      {report_path}")

    # ── Print summary to console ──
    print()
    print(report)


if __name__ == "__main__":
    main()
