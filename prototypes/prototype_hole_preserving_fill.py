"""
Prototype: LoopSkeleton with fixed template + hole-preserving fill.

Approach:
1. Use the existing `_build_single_loop` as a fixed 8-voxel ring template.
2. Identify the "protected interior voxel" — the voxel inside the ring that
   must remain empty to preserve β₁ = 1.
3. Fill phase adds voxels around the ring but rejects any candidate that
   fills the protected voxel or blocks the through-hole.

Test on 7³ and 11³ grids, various target voxel counts.
Verify β₁ = 1 across many random seeds.
"""

import random
import sys
from pathlib import Path
from typing import Set, Tuple, List

sys.path.insert(0, str(Path(__file__).parent))

from cognitive_mapping import SkeletonSpec
from skeleton_generation import (
    LoopSkeleton, TreeSkeleton, _get_neighbors, _calculate_cycle_count, _count_components
)
from shape_generation import _calculate_cycle_count as calc_cycle_count
from topology_extras import cubical_betti_1, count_enclosed_cavities


class HolePreservingLoopSkeleton(LoopSkeleton):
    """LoopSkeleton with hole-preserving fill.

    Protected voxels: the interior of each loop skeleton, which must stay
    empty to preserve continuum β₁ = num_loops.
    """

    def generate(self, spec: SkeletonSpec) -> Set[Tuple[int, int, int]]:
        self.voxels.clear()
        self._protected = set()  # voxels that must NEVER be filled
        self._build(spec)
        self._identify_protected_voxels(spec)
        if spec.direction_spread != "planar":
            self._add_perpendicular_seeds_safe(spec)
        self._fill_remaining_hole_preserving(spec)
        return self.voxels

    def _identify_protected_voxels(self, spec: SkeletonSpec) -> None:
        """Find the interior voxels of each ring in the skeleton.

        For a rectangular ring in plane (ax1, ax2), the interior voxels are
        those inside the rectangle bounds along (ax1, ax2), at the same ax3
        coordinates as the ring, and not part of the skeleton.

        We identify these by finding voxels that would close the hole:
        for each empty voxel, check if it's "enclosed" by skeleton voxels
        on all sides within the ring's plane.
        """
        if not self.voxels:
            return

        # Simple approach: for each voxel NOT in the skeleton, check whether
        # placing it there would decrease β₁ (i.e., fill the hole).
        # Use cubical_betti_1 as the ground truth.
        current_b1 = cubical_betti_1(self.voxels)
        if current_b1 == 0:
            # No holes to preserve
            return

        # Get bounding box of skeleton
        xs = [v[0] for v in self.voxels]
        ys = [v[1] for v in self.voxels]
        zs = [v[2] for v in self.voxels]
        # Expand slightly to catch just-outside protected voxels
        x_range = range(max(0, min(xs)), min(self.grid_size[0], max(xs) + 1))
        y_range = range(max(0, min(ys)), min(self.grid_size[1], max(ys) + 1))
        z_range = range(max(0, min(zs)), min(self.grid_size[2], max(zs) + 1))

        for x in x_range:
            for y in y_range:
                for z in z_range:
                    v = (x, y, z)
                    if v in self.voxels:
                        continue
                    # Would filling this voxel decrease β₁?
                    test_set = self.voxels | {v}
                    if cubical_betti_1(test_set) < current_b1:
                        self._protected.add(v)

    def _add_perpendicular_seeds_safe(self, spec: SkeletonSpec) -> None:
        """Add perpendicular seed voxels, avoiding protected voxels."""
        # Call the original method but filter its output
        # Actually — safer to just skip this for the prototype and see
        # if we can get β₁=1 without 3D seeds first.
        # Real implementation would need to make _add_perpendicular_seeds
        # protected-voxel-aware.
        pass  # Skip perpendicular seeds for prototype simplicity

    def _fill_remaining_hole_preserving(self, spec: SkeletonSpec) -> None:
        """Add voxels while preserving β₁ = num_loops.

        Rejects any candidate voxel that is protected (would fill a hole).
        """
        target = spec.voxel_count
        max_iters = max(50, (target - len(self.voxels)) * 8)

        preferred_axis = self._get_preferred_axis(spec)

        for _ in range(max_iters):
            if len(self.voxels) >= target:
                break

            # Gather candidates adjacent to current shape
            candidates = set()
            for v in self.voxels:
                for n in _get_neighbors(v, self.grid_size, include_diagonal=False):
                    if n not in self.voxels and self.in_bounds(n):
                        candidates.add(n)

            if not candidates:
                break

            # HOLE-PRESERVING FILTER: reject candidates that are protected
            valid_by_protection = [c for c in candidates if c not in self._protected]

            if not valid_by_protection:
                # Every candidate is protected — we're surrounded by holes we
                # must preserve. Stop growing.
                break

            # Pick using existing logic (from parent's _pick_fill_candidate)
            choice = self._pick_fill_candidate(valid_by_protection, spec, preferred_axis)
            self.add_voxel(choice)

    def _validate(self, spec: SkeletonSpec) -> bool:
        """Validate: connectivity + β₁ preserved."""
        if _count_components(self.voxels, self.grid_size) != 1:
            return False
        b1 = cubical_betti_1(self.voxels)
        target_loops = max(0, spec.num_loops)
        return b1 == target_loops


def test_grid(grid_size, num_loops, voxel_count, packing="medium", n_seeds=20):
    """Run the prototype on a given (grid, loops, voxels) configuration."""
    GRID = (grid_size, grid_size, grid_size)
    results = []
    for seed in range(n_seeds):
        random.seed(seed * 1000 + hash((grid_size, num_loops, voxel_count)) % 1000)
        spec = SkeletonSpec(
            archetype="bridge",
            voxel_count=voxel_count,
            grid_size=GRID,
            num_branches=max(3, num_loops + 2),
            num_loops=num_loops,
            num_components=1,
            direction_spread="planar",  # keep 2D for prototype simplicity
            planarity="medium",
            packing=packing,
        )
        try:
            gen = HolePreservingLoopSkeleton(GRID)
            voxels = gen.generate(spec)
        except Exception as e:
            print(f"  seed={seed}: FAILED — {type(e).__name__}: {e}")
            continue
        if not voxels:
            continue
        mu = calc_cycle_count(voxels, GRID)
        b1 = cubical_betti_1(voxels)
        results.append({
            "seed": seed,
            "n_voxels": len(voxels),
            "mu": mu,
            "b1": b1,
            "target_b1": num_loops,
            "b1_hit": b1 == num_loops,
            "mu_matches_b1": mu == b1,
        })
    return results


def summarize(results, label):
    if not results:
        print(f"{label}: no results")
        return
    n = len(results)
    hits = sum(1 for r in results if r["b1_hit"])
    mu_b1_agree = sum(1 for r in results if r["mu_matches_b1"])
    mean_mu = sum(r["mu"] for r in results) / n
    mean_b1 = sum(r["b1"] for r in results) / n
    mean_vox = sum(r["n_voxels"] for r in results) / n
    mu_range = (min(r["mu"] for r in results), max(r["mu"] for r in results))
    b1_range = (min(r["b1"] for r in results), max(r["b1"] for r in results))
    print(f"{label}")
    print(f"  n={n}  target_β₁={results[0]['target_b1']}")
    print(f"  β₁ hits target: {hits}/{n} ({100*hits/n:.1f}%)")
    print(f"  μ==β₁: {mu_b1_agree}/{n} ({100*mu_b1_agree/n:.1f}%)")
    print(f"  Mean μ = {mean_mu:.2f} [{mu_range[0]}-{mu_range[1]}]")
    print(f"  Mean β₁ = {mean_b1:.2f} [{b1_range[0]}-{b1_range[1]}]")
    print(f"  Mean voxel count = {mean_vox:.1f}")


def main():
    print("=" * 70)
    print("HOLE-PRESERVING LOOPSKELETON PROTOTYPE")
    print("Fixed ring template + β₁-preserving fill")
    print("=" * 70)

    # Test 1: 7³ grid, one loop, various voxel counts
    print("\n--- 7³ grid, single loop (target β₁=1) ---")
    for voxels in [8, 12, 16, 20]:
        results = test_grid(7, 1, voxels, "medium", n_seeds=20)
        summarize(results, f"7³, target_voxels={voxels}, packing=medium")

    # Test 2: 11³ grid, one loop, various voxel counts
    print("\n--- 11³ grid, single loop (target β₁=1) ---")
    for voxels in [8, 15, 25, 40]:
        results = test_grid(11, 1, voxels, "medium", n_seeds=20)
        summarize(results, f"11³, target_voxels={voxels}, packing=medium")

    # Test 3: 7³ grid, two loops (double_loop template)
    print("\n--- 7³ grid, double loop (target β₁=2) ---")
    for voxels in [12, 16, 20]:
        results = test_grid(7, 2, voxels, "medium", n_seeds=20)
        summarize(results, f"7³, target_voxels={voxels}, packing=medium")

    # Test 4: 11³ grid, two loops
    print("\n--- 11³ grid, double loop (target β₁=2) ---")
    for voxels in [14, 20, 30]:
        results = test_grid(11, 2, voxels, "medium", n_seeds=20)
        summarize(results, f"11³, target_voxels={voxels}, packing=medium")

    # Test 5: 11³ grid, multi loop (β₁=3, 4)
    print("\n--- 11³ grid, multi loop ---")
    for num_loops in [3, 4]:
        for voxels in [20, 30]:
            results = test_grid(11, num_loops, voxels, "medium", n_seeds=15)
            summarize(results, f"11³, target_loops={num_loops}, target_voxels={voxels}")


if __name__ == "__main__":
    main()
