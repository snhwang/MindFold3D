"""
CompactSkeleton Prototype & Fidelity Comparison — MindFold 3D

Standalone test script that implements a CompactSkeleton and benchmarks its
fidelity against the current TreeSkeleton for low Spatial Form shapes.

Does NOT modify any existing files. Imports the existing pipeline and
monkey-patches the skeleton selection only within this script.

Usage:
    python test_compact_grammar.py                # default: 50 shapes per cell
    python test_compact_grammar.py --n 170        # match paper benchmark size
    python test_compact_grammar.py --n 100 --json compact_results.json
"""

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any

from cognitive_mapping import (
    SHAPE_DIMENSIONS,
    SkeletonSpec,
    get_skeleton_spec,
    reverse_map_cognitive_profile,
)
from skeleton_generation import (
    SkeletonRule,
    generate_shape_skeleton,
    _optimize_geometry,
    _compute_symmetry_score_for_gate,
    _get_neighbors,
    _count_components,
    _calculate_cycle_count,
)
from shape_generation import analyze_shape_features
from shape_features import ShapeFeatureSet


# =============================================================================
# CompactSkeleton — BFS flood-fill from center, maximize neighbor density
# =============================================================================

class CompactSkeleton(SkeletonRule):
    """Axis-balanced symmetric growth from grid center for low Spatial Form.

    Targets low anisotropy + high symmetry by:
        1. Seeding a symmetric cross (one voxel per axis direction from center)
           to establish balanced extent from the start.
        2. Growing with strong axis-balance penalties and symmetry rewards.
        3. Using mirror-pair placement: when adding a voxel, also add its
           reflection if budget allows, guaranteeing bilateral symmetry.
        4. Enough randomness (top 4-5 candidates) to produce varied shapes.

    The key insight vs. the previous attempt: a single-voxel seed on a 7x7x7
    grid always grows into the same 2x2x3 block (AI=0.67). The symmetric
    cross seed + mirror-pair placement breaks this degeneracy.
    """

    def _build(self, spec: SkeletonSpec) -> None:
        """Seed a symmetric cross: center + one step in each axis direction."""
        cx = self.grid_size[0] // 2
        cy = self.grid_size[1] // 2
        cz = self.grid_size[2] // 2
        self.add_voxel((cx, cy, cz))

        # Add one voxel in each positive axis direction to establish
        # balanced extent. This gives us a 3-voxel cross as the seed
        # (budget permitting), which has AI ≈ 0 by construction.
        for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            v = (cx + dx, cy + dy, cz + dz)
            if self.in_bounds(v) and len(self.voxels) < spec.voxel_count:
                self.add_voxel(v)

    def _fill_remaining(self, spec: SkeletonSpec) -> None:
        """Axis-balanced growth with mirror-pair placement for isotropy."""
        target = spec.voxel_count

        # Choose a random primary mirror axis for this shape.
        # Mirror-pair placement along this axis guarantees bilateral symmetry.
        mirror_axis = random.randint(0, 2)

        for _ in range(target * 15):
            if len(self.voxels) >= target:
                break

            # Gather all empty face-adjacent candidates
            candidates = set()
            for v in self.voxels:
                for n in _get_neighbors(v, self.grid_size, include_diagonal=False):
                    if n not in self.voxels and self.in_bounds(n):
                        candidates.add(n)

            if not candidates:
                break

            # Precompute bounding box extents
            bb_mins = [min(v[ax] for v in self.voxels) for ax in range(3)]
            bb_maxs = [max(v[ax] for v in self.voxels) for ax in range(3)]
            extents = [bb_maxs[ax] - bb_mins[ax] for ax in range(3)]

            # Axis midpoints for mirror computation
            axis_mids = [bb_mins[ax] + bb_maxs[ax] for ax in range(3)]

            # Center of mass
            n_vox = len(self.voxels)
            com = [
                sum(v[ax] for v in self.voxels) / n_vox
                for ax in range(3)
            ]

            scored = []
            for c in candidates:
                score = 0.0

                # 1. Neighbor count — moderate weight (not dominant like before)
                neighbor_count = sum(
                    1 for n in _get_neighbors(c, self.grid_size, include_diagonal=False)
                    if n in self.voxels
                )
                score += neighbor_count * 15

                # 2. Axis balance — DOMINANT scoring factor.
                #    Penalize growth along the already-longest axis,
                #    reward growth along the shortest axis.
                new_extents = [
                    max(bb_maxs[ax], c[ax]) - min(bb_mins[ax], c[ax])
                    for ax in range(3)
                ]
                # Penalty for increasing the max extent
                max_ext = max(new_extents)
                min_ext = min(new_extents)
                score -= (max_ext - min_ext) * 25

                # Bonus if this candidate grows the shortest axis
                shortest_ax = extents.index(min(extents))
                if c[shortest_ax] < bb_mins[shortest_ax] or c[shortest_ax] > bb_maxs[shortest_ax]:
                    score += 20

                # 3. Mirror-pair bonus — does this candidate have a mirror partner
                #    that's also a valid candidate or already placed?
                mirror_pos = list(c)
                mirror_pos[mirror_axis] = axis_mids[mirror_axis] - c[mirror_axis]
                mirror_pos = tuple(mirror_pos)
                mirror_exists = mirror_pos in self.voxels
                mirror_available = (
                    mirror_pos in candidates
                    and self.in_bounds(mirror_pos)
                )
                if mirror_exists:
                    score += 20  # already symmetric
                elif mirror_available:
                    score += 30  # can place both for guaranteed symmetry

                # 4. Distance from COM — mild penalty to keep growth centered
                dist = sum((c[ax] - com[ax]) ** 2 for ax in range(3)) ** 0.5
                score -= dist * 3

                scored.append((c, score))

            scored.sort(key=lambda x: x[1], reverse=True)

            # Pick from top candidates with enough randomness for variety
            top_n = max(1, min(5, len(scored)))
            choice = random.choice(scored[:top_n])[0]
            self.add_voxel(choice)

            # Mirror-pair placement: if budget allows, also place the mirror
            if len(self.voxels) < target:
                mirror_pos = list(choice)
                mirror_pos[mirror_axis] = axis_mids[mirror_axis] - choice[mirror_axis]
                mirror_pos = tuple(mirror_pos)
                if mirror_pos not in self.voxels and self.in_bounds(mirror_pos):
                    # Check the mirror is adjacent to existing shape
                    adj = any(
                        n in self.voxels
                        for n in _get_neighbors(mirror_pos, self.grid_size, include_diagonal=False)
                    )
                    if adj:
                        self.add_voxel(mirror_pos)

    def _validate(self, spec: SkeletonSpec) -> bool:
        """Compact shapes should always be single-component."""
        return (
            _count_components(self.voxels, self.grid_size) == 1
            and len(self.voxels) >= spec.voxel_count
        )


# =============================================================================
# Generation function using CompactSkeleton
# =============================================================================

def _optimize_relaxed_branching(
    voxels: Set[Tuple[int, int, int]],
    grid_size: Tuple[int, int, int],
    shape_difficulties: Dict[str, str],
    max_iterations: int = 300,
) -> Set[Tuple[int, int, int]]:
    """Geometric optimizer variant that allows branching factor to change.

    For compact/isotropic shapes, the strict branch_tol=0 constraint in the
    main optimizer prevents most swaps because dense clusters have high
    branching and almost every swap changes branching. Relaxing this
    constraint gives the optimizer freedom to reshape the cluster for
    lower anisotropy while preserving only connectivity and cycle count.
    """
    from cognitive_mapping import SHAPE_DIMENSIONS

    GEO_FEATURES = {
        "compactness_score", "planarity_score", "anisotropy_index",
        "shape_form_index", "symmetry_score",
    }
    _LEVELS = ["low", "medium", "high", "expert"]

    def _exclusive_range(feat_config, level):
        range_val = feat_config.get(level)
        if range_val is None:
            return None
        if isinstance(range_val, tuple):
            lo, hi = range_val
        else:
            lo = hi = float(range_val)
        idx = _LEVELS.index(level)
        if idx < len(_LEVELS) - 1:
            next_range = feat_config.get(_LEVELS[idx + 1])
            if isinstance(next_range, tuple) and next_range[0] < hi:
                hi = next_range[0]
        if idx > 0:
            prev_range = feat_config.get(_LEVELS[idx - 1])
            if isinstance(prev_range, tuple) and prev_range[1] > lo:
                lo = prev_range[1]
        if lo > hi:
            return range_val if isinstance(range_val, tuple) else (lo, lo)
        return (lo, hi)

    dim_targets: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for dim_name, dim_config in SHAPE_DIMENSIONS.items():
        level = shape_difficulties.get(dim_name, "medium")
        dt: Dict[str, Tuple[float, float]] = {}
        for feat_name, feat_config in dim_config["features"].items():
            if feat_name not in GEO_FEATURES:
                continue
            excl = _exclusive_range(feat_config, level)
            if excl is not None:
                dt[feat_name] = excl
        if dt:
            dim_targets[dim_name] = dt

    def _compute_symmetry_score(vs):
        if len(vs) < 2:
            return 1.0
        vlist = list(vs)
        best = 0.0
        for axis in range(3):
            coords_ax = [v[axis] for v in vlist]
            mid = min(coords_ax) + max(coords_ax)
            reflected = frozenset(
                tuple(mid - v[axis] if i == axis else v[i] for i in range(3))
                for v in vlist
            )
            overlap = len(vs & reflected) / len(vs)
            if overlap > best:
                best = overlap
        return best

    def compute_geo(vs):
        from skeleton_generation import (
            _calculate_compactness_score, _calculate_planarity_score,
            _get_pca_eigenvalues, _calculate_anisotropy_index,
            _calculate_shape_form_index,
        )
        feats = {}
        feats["compactness_score"] = _calculate_compactness_score(vs, grid_size)
        feats["planarity_score"] = _calculate_planarity_score(vs, grid_size)
        eigvals = _get_pca_eigenvalues(vs)
        feats["anisotropy_index"] = _calculate_anisotropy_index(eigvals)
        feats["shape_form_index"] = _calculate_shape_form_index(eigvals)
        feats["symmetry_score"] = _compute_symmetry_score(vs)
        return feats

    def geo_score(feats):
        total = 0.0
        for dim_name, dt in dim_targets.items():
            dim_dev = 0.0
            for feat_name, (lo, hi) in dt.items():
                val = feats.get(feat_name, 0.0)
                if val < lo:
                    dim_dev += (lo - val) ** 2
                elif val > hi:
                    dim_dev += (val - hi) ** 2
            if dt:
                dim_dev /= len(dt)
            total += dim_dev
        return total

    topo_cycles = _calculate_cycle_count(voxels, grid_size)
    topo_components = _count_components(voxels, grid_size)
    # KEY DIFFERENCE: no branching constraint
    cycle_tol = 0

    def _topo_filter(trial):
        if _count_components(trial, grid_size) != topo_components:
            return False
        if abs(_calculate_cycle_count(trial, grid_size) - topo_cycles) > cycle_tol:
            return False
        # Branching is FREE to change — this is the relaxation
        return True

    current = set(voxels)
    cur_feats = compute_geo(current)
    cur_score = geo_score(cur_feats)

    if cur_score < 1e-6:
        return current

    def _removable_voxels(vs):
        result = []
        for v in vs:
            temp = vs - {v}
            if temp and _count_components(temp, grid_size) == topo_components:
                result.append(v)
        return result

    def _candidate_positions(vs):
        cands = set()
        for v in vs:
            for n in _get_neighbors(v, grid_size, include_diagonal=False):
                if n not in vs:
                    cands.add(n)
        return cands

    def _neighbor_count_in(v, vs):
        return sum(1 for n in _get_neighbors(v, grid_size, include_diagonal=False)
                   if n in vs)

    stale_count = 0
    for _ in range(max_iterations):
        if cur_score < 1e-6 or stale_count > 40:
            break

        removable = _removable_voxels(current)
        if not removable:
            break

        removable.sort(key=lambda v: _neighbor_count_in(v, current))

        improved = False
        for v_rm in removable[:15]:
            after_rm = current - {v_rm}
            cands = _candidate_positions(after_rm)
            cands.discard(v_rm)
            if not cands:
                continue

            cand_list = list(cands)
            cand_list.sort(
                key=lambda c: _neighbor_count_in(c, after_rm), reverse=True
            )

            best_add = None
            best_sc = cur_score
            best_feats = None

            for v_add in cand_list[:25]:
                trial = after_rm | {v_add}
                if not _topo_filter(trial):
                    continue
                tf = compute_geo(trial)
                ts = geo_score(tf)
                if ts < best_sc:
                    best_sc = ts
                    best_add = v_add
                    best_feats = tf

            if best_add is not None:
                current = after_rm | {best_add}
                cur_score = best_sc
                cur_feats = best_feats
                improved = True
                stale_count = 0
                break

        if not improved:
            stale_count += 1

    return current


def generate_with_compact(
    spec: SkeletonSpec,
    max_attempts: int = 5,
    relax_branching: bool = False,
) -> Dict[str, Any]:
    """Generate a shape using CompactSkeleton + optimizer.

    Args:
        relax_branching: If True, use the relaxed optimizer that allows
            branching factor to change freely (better for compact shapes).
    """
    grid_size = spec.grid_size

    best_voxels = None
    for attempt in range(max_attempts):
        skeleton = CompactSkeleton(grid_size)
        skeleton.generate(spec)
        if skeleton._validate(spec):
            best_voxels = skeleton.voxels
            break
        if best_voxels is None:
            best_voxels = skeleton.voxels

    voxels_set = best_voxels if best_voxels else set()

    if relax_branching:
        voxels_set = _optimize_relaxed_branching(
            voxels_set, grid_size, spec.shape_difficulties,
        )
    else:
        voxels_set = _optimize_geometry(
            voxels_set, grid_size, spec.shape_difficulties,
            preserve_chirality=False,
        )

    analyzed_sfs = analyze_shape_features(voxels_set, grid_size)

    return {
        "voxels": list(voxels_set),
        "grid_size": list(grid_size),
        "features": analyzed_sfs.to_dict(),
        "generation_mode": "skeleton",
        "archetype": "compact",
    }


# =============================================================================
# Benchmark logic (derived from benchmark_fidelity.py)
# =============================================================================

LEVELS = ["low", "medium", "high", "expert"]
ALL_DIMS = list(SHAPE_DIMENSIONS.keys())

DIMENSION_FEATURES: Dict[str, List[str]] = {
    dim_name: list(dim_cfg["features"].keys())
    for dim_name, dim_cfg in SHAPE_DIMENSIONS.items()
}


def _generate_one(spec: SkeletonSpec, variant: str) -> Dict[str, Any]:
    """Generate a single shape using the specified variant."""
    if variant == "current":
        return generate_shape_skeleton(spec)
    elif variant == "compact":
        return generate_with_compact(spec, relax_branching=False)
    elif variant == "compact_relaxed":
        return generate_with_compact(spec, relax_branching=True)
    elif variant == "current_relaxed":
        # Current TreeSkeleton but with relaxed branching optimizer
        from skeleton_generation import TreeSkeleton
        grid_size = spec.grid_size
        best_voxels = None
        for attempt in range(5):
            skeleton = TreeSkeleton(grid_size)
            skeleton.generate(spec)
            if skeleton._validate(spec):
                best_voxels = skeleton.voxels
                break
            if best_voxels is None:
                best_voxels = skeleton.voxels
        voxels_set = best_voxels if best_voxels else set()
        voxels_set = _optimize_relaxed_branching(
            voxels_set, grid_size, spec.shape_difficulties,
        )
        analyzed_sfs = analyze_shape_features(voxels_set, grid_size)
        return {
            "voxels": list(voxels_set),
            "grid_size": list(grid_size),
            "features": analyzed_sfs.to_dict(),
            "generation_mode": "skeleton",
            "archetype": "tree",
        }
    else:
        raise ValueError(f"Unknown variant: {variant}")


def _generate_and_measure(
    dim_name: str,
    level: str,
    n: int,
    variant: str,
) -> List[Dict]:
    """Generate n shapes using the specified variant."""
    results = []
    shape_difficulties = {d: "medium" for d in ALL_DIMS}
    shape_difficulties[dim_name] = level

    for _ in range(n):
        try:
            spec = get_skeleton_spec(
                shape_difficulties=shape_difficulties,
                task_difficulties={},
            )

            # Only use non-current variants for low spatial_form
            if variant != "current" and not (dim_name == "spatial_form" and level == "low"):
                result = generate_shape_skeleton(spec)
            else:
                result = _generate_one(spec, variant)

            features = result.get("features", {})

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


def _fidelity_rate(results: List[Dict], intended_level: str) -> Dict:
    classified = [r["classified_level"] for r in results if "classified_level" in r]
    if not classified:
        return {"fidelity_rate": None, "n": 0, "distribution": {}}
    distribution: Dict[str, int] = defaultdict(int)
    for level in classified:
        distribution[level] += 1
    n = len(classified)
    correct = distribution.get(intended_level, 0)
    return {
        "fidelity_rate": round(correct / n, 3),
        "n": n,
        "distribution": {k: round(v / n, 3) for k, v in sorted(distribution.items())},
    }


VARIANTS = [
    ("current",          "Current (Tree + strict opt)"),
    ("current_relaxed",  "Tree + relaxed branching opt"),
    ("compact",          "Compact + strict opt"),
    ("compact_relaxed",  "Compact + relaxed branching opt"),
]


def run_comparison(n_per_cell: int) -> Dict:
    """Run head-to-head comparison across all variants for spatial_form."""
    results = {}

    for variant, desc in VARIANTS:
        results[variant] = {}
        for level in LEVELS:
            is_target = (level == "low")
            tag = f"[{variant[:15]:<15s}]" if is_target else f"[{'—':^17s}]"
            active = f" ({desc})" if is_target else ""
            print(f"  {tag} spatial_form={level}: generating {n_per_cell} shapes{active}...",
                  end=" ", flush=True)

            data = _generate_and_measure("spatial_form", level, n_per_cell, variant)
            successes = [r for r in data if "error" not in r]
            errors = len(data) - len(successes)
            print(f"{len(successes)} ok{f', {errors} errors' if errors else ''}")

            cell = {
                "intended_level": level,
                "n_generated": len(successes),
                "n_errors": errors,
                "dimension_fidelity": _fidelity_rate(successes, level),
                "features": {},
            }
            for feat in DIMENSION_FEATURES["spatial_form"]:
                cell["features"][feat] = _feature_stats(successes, feat)

            results[variant][level] = cell

    return results


def _tier_range_str(feature_name: str) -> str:
    feat_cfg = SHAPE_DIMENSIONS["spatial_form"]["features"][feature_name]
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


def print_comparison(results: Dict, n_per_cell: int) -> None:
    variant_keys = [v for v, _ in VARIANTS]
    variant_labels = {v: desc for v, desc in VARIANTS}

    print()
    print("=" * 90)
    print("  SPATIAL FORM FIDELITY — VARIANT COMPARISON")
    print(f"  N = {n_per_cell} shapes per cell")
    print(f"  Variants tested (applied to LOW tier only; other tiers use current pipeline):")
    for v, desc in VARIANTS:
        print(f"    {v:<20s} {desc}")
    print("=" * 90)

    # Fidelity table — focus on LOW tier (the only one that differs)
    print(f"\n  LOW TIER FIDELITY (the 80% gap target):\n")
    print(f"  {'Variant':<22s} {'Fidelity':>10}  {'Distribution'}")
    print(f"  {'─'*22} {'─'*10}  {'─'*45}")

    cur_rate = None
    for variant in variant_keys:
        fid = results[variant]["low"]["dimension_fidelity"]
        rate = fid.get("fidelity_rate", 0) or 0
        dist = "  ".join(f"{k}:{v:.0%}" for k, v in fid.get("distribution", {}).items())
        if cur_rate is None:
            cur_rate = rate
            delta_str = "(baseline)"
        else:
            delta = rate - cur_rate
            delta_str = f"({delta:+.1%})" if delta != 0 else "(same)"
        marker = " **" if variant != "current" and rate > cur_rate else ""
        print(f"  {variant:<22s} {rate:>10.1%}  {dist:<35s} {delta_str}{marker}")

    # Per-feature comparison for LOW tier
    print(f"\n  Per-feature detail for LOW tier:\n")
    for feat in DIMENSION_FEATURES["spatial_form"]:
        print(f"  Feature: {feat}")
        print(f"  Defined range: {_tier_range_str(feat)}")
        print(f"  {'Variant':<22s} {'Mean':>10} {'±Std':>10} {'Min':>10} {'Max':>10}")
        print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
        for variant in variant_keys:
            fstats = results[variant]["low"]["features"].get(feat, {})
            mean = fstats.get("mean")
            std = fstats.get("std")
            mn = fstats.get("min")
            mx = fstats.get("max")
            if mean is not None:
                print(f"  {variant:<22s} {mean:>10.4f} {std:>10.4f} {mn:>10.4f} {mx:>10.4f}")
            else:
                print(f"  {variant:<22s} {'N/A':>10}")
        print()

    # Sanity check: other tiers should be identical across variants
    print(f"  OTHER TIERS (should be identical across variants — sanity check):\n")
    print(f"  {'Variant':<22s} {'Medium':>10} {'High':>10} {'Expert':>10}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10}")
    for variant in variant_keys:
        rates = []
        for level in ["medium", "high", "expert"]:
            r = results[variant][level]["dimension_fidelity"].get("fidelity_rate", 0) or 0
            rates.append(f"{r:.1%}")
        print(f"  {variant:<22s} {rates[0]:>10} {rates[1]:>10} {rates[2]:>10}")
    print()

    # Summary
    print(f"  {'─' * 70}")
    print(f"  SUMMARY — PROJECTED OVERALL SYSTEM FIDELITY")
    print(f"  (using best low-tier result + SC=100% + SS=100%)\n")
    for variant in variant_keys:
        rates = [
            results[variant][l]["dimension_fidelity"].get("fidelity_rate", 0) or 0
            for l in LEVELS
        ]
        sf_mean = statistics.mean(rates)
        overall = (sf_mean + 1.0 + 1.0) / 3
        low_rate = rates[0]
        print(f"  {variant:<22s}  SF low={low_rate:.1%}  SF mean={sf_mean:.1%}  Overall={overall:.1%}")
    print()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CompactSkeleton prototype fidelity comparison"
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Number of shapes per cell (default: 50)"
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Path to write full results as JSON"
    )
    args = parser.parse_args()

    total = args.n * len(LEVELS) * len(VARIANTS)
    print(f"\nCompactSkeleton Fidelity Test")
    print(f"Shapes per cell: {args.n}")
    print(f"Total shapes: {total}\n")

    results = run_comparison(args.n)
    print_comparison(results, args.n)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results written to: {args.json}\n")


if __name__ == "__main__":
    main()
