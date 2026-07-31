"""
Prototype v2: Hole-marked loop templates.

Design:
- Templates declare TWO sets of voxels:
    * skeleton (filled — part of the shape)
    * hole (reserved — cannot be filled, don't count toward voxel count)
- Fill phase treats hole voxels as forbidden. No β₁ computation during fill.
- β₁ is preserved by construction: the hole column through each ring is
  inviolable, so tube capping is topologically impossible.

Features added over v1 prototype:
- Axis-aligned rotation (ring can lie in xy, xz, or yz plane)
- Variable ring sizing (3x3, 4x4, 5x5, ... up to grid capacity)
- Random translation within grid bounds
- Templates for β₁ = 1, 2, 3, 4
"""

import random
import sys
from pathlib import Path
from typing import Set, Tuple, List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

from mindfold3d.cognitive_mapping import SkeletonSpec
from mindfold3d.skeleton_generation import (
    SkeletonRule, TreeSkeleton, _get_neighbors, _count_components
)
from mindfold3d.shape_generation import _calculate_cycle_count as calc_cycle_count
from mindfold3d.topology_extras import cubical_betti_1


Voxel = Tuple[int, int, int]


# ==============================================================
# Ring template builders
# ==============================================================

def build_rectangle_perimeter(
    center: Voxel,
    plane_axes: Tuple[int, int],
    size_a: int,
    size_b: int,
) -> Set[Voxel]:
    """Build a rectangular perimeter of size (size_a x size_b) in plane_axes,
    centered at `center`.

    Returns the set of perimeter voxels (open rectangle, no interior).
    """
    ax1, ax2 = plane_axes
    voxels: Set[Voxel] = set()

    # Top and bottom edges
    for i in range(size_a):
        for j in (0, size_b - 1):
            v = list(center)
            v[ax1] = center[ax1] - size_a // 2 + i
            v[ax2] = center[ax2] - size_b // 2 + j
            voxels.add(tuple(v))
    # Left and right edges
    for j in range(1, size_b - 1):
        for i in (0, size_a - 1):
            v = list(center)
            v[ax1] = center[ax1] - size_a // 2 + i
            v[ax2] = center[ax2] - size_b // 2 + j
            voxels.add(tuple(v))
    return voxels


def build_rectangle_interior_hole(
    center: Voxel,
    plane_axes: Tuple[int, int],
    size_a: int,
    size_b: int,
    grid_size: Tuple[int, int, int],
    protect_full_column: bool = True,
) -> Set[Voxel]:
    """Return the hole set for a rectangle perimeter.

    - Interior voxels at the ring's plane (empty voxels inside the perimeter).
    - Optionally, extend the interior through the full perpendicular column
      so the tube cannot be capped from above/below.
    """
    ax1, ax2 = plane_axes
    ax3 = 3 - ax1 - ax2  # perpendicular axis
    hole: Set[Voxel] = set()

    for i in range(1, size_a - 1):
        for j in range(1, size_b - 1):
            v = list(center)
            v[ax1] = center[ax1] - size_a // 2 + i
            v[ax2] = center[ax2] - size_b // 2 + j
            hole.add(tuple(v))

    if protect_full_column:
        # Extend the entire interior column through the perpendicular axis
        column_seeds = list(hole)
        for seed in column_seeds:
            for k in range(grid_size[ax3]):
                v = list(seed)
                v[ax3] = k
                hole.add(tuple(v))

    return hole


def choose_random_plane(rng: random.Random) -> Tuple[int, int]:
    """Return a random plane axis pair (ax1, ax2) with ax1 < ax2.
    Choices: (0,1)=xy, (0,2)=xz, (1,2)=yz.
    """
    return rng.choice([(0, 1), (0, 2), (1, 2)])


def choose_ring_size(
    grid_size: Tuple[int, int, int],
    plane_axes: Tuple[int, int],
    rng: random.Random,
    min_size: int = 3,
) -> Tuple[int, int]:
    """Choose a random rectangle (size_a, size_b) that fits in the grid plane.

    Leaves a margin of 1 voxel on each side so the ring is not flush with the
    grid boundary (allows fill room + rotation flexibility).
    """
    ax1, ax2 = plane_axes
    max_a = grid_size[ax1] - 2  # margin of 1 each side
    max_b = grid_size[ax2] - 2
    max_a = max(min_size, max_a)
    max_b = max(min_size, max_b)

    size_a = rng.randint(min_size, max_a)
    size_b = rng.randint(min_size, max_b)
    return size_a, size_b


def choose_ring_center(
    grid_size: Tuple[int, int, int],
    plane_axes: Tuple[int, int],
    size_a: int,
    size_b: int,
    rng: random.Random,
) -> Voxel:
    """Choose a random center for a ring that keeps it inside the grid."""
    ax1, ax2 = plane_axes
    ax3 = 3 - ax1 - ax2

    # In-plane: center must allow size_a and size_b to fit
    c1_lo = size_a // 2
    c1_hi = grid_size[ax1] - 1 - size_a // 2
    c2_lo = size_b // 2
    c2_hi = grid_size[ax2] - 1 - size_b // 2
    # Perpendicular: any value in grid
    c3_lo = 0
    c3_hi = grid_size[ax3] - 1

    center = [0, 0, 0]
    center[ax1] = rng.randint(c1_lo, max(c1_lo, c1_hi))
    center[ax2] = rng.randint(c2_lo, max(c2_lo, c2_hi))
    center[ax3] = rng.randint(c3_lo, c3_hi)
    return tuple(center)


# ==============================================================
# HoleMarkedLoopSkeleton — full prototype
# ==============================================================

class HoleMarkedLoopSkeleton(SkeletonRule):
    """Loop skeleton with hole-marked interior voxels.

    - Fixed rectangular ring templates for β₁ ∈ {1, 2, 3, 4}
    - Each template declares filled voxels (skeleton) AND hole voxels (reserved)
    - Fill phase respects hole voxels; β₁ is preserved by construction
    - Supports axis-aligned rotation and variable ring sizing
    """

    def __init__(self, grid_size, rng: Optional[random.Random] = None):
        super().__init__(grid_size)
        self.hole_voxels: Set[Voxel] = set()
        self.rng = rng or random.Random()

    def generate(self, spec: SkeletonSpec) -> Set[Voxel]:
        self.voxels.clear()
        self.hole_voxels.clear()

        target_loops = max(0, spec.num_loops)

        if target_loops == 0:
            # No loops requested — degenerate to tree
            tree = TreeSkeleton(self.grid_size)
            tree._build(spec)
            self.voxels = tree.voxels
            tree._fill_remaining(spec) if hasattr(tree, "_fill_remaining") else None
            return self.voxels

        if target_loops == 1:
            self._build_single_ring(spec)
        elif target_loops == 2:
            self._build_double_ring(spec)
        elif target_loops == 3:
            self._build_triple_ring(spec)
        else:  # target_loops >= 4
            self._build_quad_ring(spec, num_rings=min(target_loops, 6))

        self._fill_hole_aware(spec)
        return self.voxels

    # ---------------------------------------------------------
    # Template constructors
    # ---------------------------------------------------------

    def _build_single_ring(self, spec: SkeletonSpec) -> None:
        """β₁=1: one rectangular ring, random plane, random size."""
        plane = choose_random_plane(self.rng)
        # Packing controls ring size preference
        if spec.packing == "dense":
            min_s, max_bump = 3, 0
        else:
            min_s, max_bump = 3, 2

        size_a, size_b = choose_ring_size(self.grid_size, plane, self.rng, min_size=min_s)
        # Prefer smaller rings when dense
        if spec.packing == "dense":
            size_a = min(size_a, min_s + 1)
            size_b = min(size_b, min_s + 1)

        center = choose_ring_center(self.grid_size, plane, size_a, size_b, self.rng)

        skeleton = build_rectangle_perimeter(center, plane, size_a, size_b)
        hole = build_rectangle_interior_hole(center, plane, size_a, size_b, self.grid_size)

        for v in skeleton:
            if self.in_bounds(v):
                self.add_voxel(v)
        self.hole_voxels.update(v for v in hole if self.in_bounds(v))

    def _build_double_ring(self, spec: SkeletonSpec) -> None:
        """β₁=2: two rings, either linked or side-by-side.

        Strategy: place two rectangular rings in the same plane, offset
        along ax1. They share one wall (figure-eight topology).
        """
        plane = choose_random_plane(self.rng)
        ax1, ax2 = plane
        ax3 = 3 - ax1 - ax2

        # Ring size (square for simplicity)
        min_s = 3
        max_s = min(4, self.grid_size[ax1] // 2 - 1, self.grid_size[ax2] - 2)
        max_s = max(min_s, max_s)
        size = self.rng.randint(min_s, max_s)

        # Two rings side by side sharing an edge
        # Ring 1 centered at (c1 - size//2, c2, c3)
        # Ring 2 centered at (c1 + size//2, c2, c3)
        c1_lo = size
        c1_hi = self.grid_size[ax1] - 1 - size
        c2_lo = size // 2
        c2_hi = self.grid_size[ax2] - 1 - size // 2
        c3_lo = 0
        c3_hi = self.grid_size[ax3] - 1

        c1 = self.rng.randint(c1_lo, max(c1_lo, c1_hi))
        c2 = self.rng.randint(c2_lo, max(c2_lo, c2_hi))
        c3 = self.rng.randint(c3_lo, c3_hi)

        offset = size - 1  # shared wall
        center1 = [0, 0, 0]
        center1[ax1] = c1 - offset // 2
        center1[ax2] = c2
        center1[ax3] = c3

        center2 = [0, 0, 0]
        center2[ax1] = c1 + (offset + 1) // 2
        center2[ax2] = c2
        center2[ax3] = c3

        for center in (tuple(center1), tuple(center2)):
            skel = build_rectangle_perimeter(center, plane, size, size)
            hole = build_rectangle_interior_hole(center, plane, size, size, self.grid_size)
            for v in skel:
                if self.in_bounds(v):
                    self.add_voxel(v)
            self.hole_voxels.update(v for v in hole if self.in_bounds(v))

    def _build_triple_ring(self, spec: SkeletonSpec) -> None:
        """β₁=3: three rings in a chain, each sharing a wall with the next."""
        plane = choose_random_plane(self.rng)
        ax1, ax2 = plane
        ax3 = 3 - ax1 - ax2

        min_s = 3
        max_s = min(4, self.grid_size[ax1] // 3, self.grid_size[ax2] - 2)
        max_s = max(min_s, max_s)
        size = self.rng.randint(min_s, max_s)

        c2 = self.rng.randint(size // 2, self.grid_size[ax2] - 1 - size // 2)
        c3 = self.rng.randint(0, self.grid_size[ax3] - 1)

        # Chain of 3 rings along ax1
        start = 1  # small margin
        num_rings = 3
        step = size - 1  # shared wall

        for i in range(num_rings):
            center = [0, 0, 0]
            center[ax1] = start + size // 2 + i * step
            center[ax2] = c2
            center[ax3] = c3

            if center[ax1] + size // 2 >= self.grid_size[ax1]:
                break

            skel = build_rectangle_perimeter(tuple(center), plane, size, size)
            hole = build_rectangle_interior_hole(tuple(center), plane, size, size, self.grid_size)
            for v in skel:
                if self.in_bounds(v):
                    self.add_voxel(v)
            self.hole_voxels.update(v for v in hole if self.in_bounds(v))

    def _build_quad_ring(self, spec: SkeletonSpec, num_rings: int = 4) -> None:
        """β₁ ≥ 4: N rings in a chain."""
        plane = choose_random_plane(self.rng)
        ax1, ax2 = plane
        ax3 = 3 - ax1 - ax2

        min_s = 3
        # More rings need more space
        available = self.grid_size[ax1] - 2
        max_ring_size = max(min_s, (available - (num_rings - 1)) // num_rings + 1)
        max_s = min(max_ring_size, 4)
        max_s = max(min_s, max_s)
        size = self.rng.randint(min_s, max_s)

        c2 = self.rng.randint(size // 2, self.grid_size[ax2] - 1 - size // 2)
        c3 = self.rng.randint(0, self.grid_size[ax3] - 1)

        start = 1
        step = size - 1

        rings_placed = 0
        for i in range(num_rings):
            center = [0, 0, 0]
            center[ax1] = start + size // 2 + i * step
            center[ax2] = c2
            center[ax3] = c3

            if center[ax1] + size // 2 >= self.grid_size[ax1]:
                break

            skel = build_rectangle_perimeter(tuple(center), plane, size, size)
            hole = build_rectangle_interior_hole(tuple(center), plane, size, size, self.grid_size)
            for v in skel:
                if self.in_bounds(v):
                    self.add_voxel(v)
            self.hole_voxels.update(v for v in hole if self.in_bounds(v))
            rings_placed += 1

    # ---------------------------------------------------------
    # Hole-aware fill
    # ---------------------------------------------------------

    def _fill_hole_aware(self, spec: SkeletonSpec) -> None:
        """Add voxels adjacent to shape, skipping any voxel in self.hole_voxels.

        No β₁ computation. No cycle-count filtering. Just: grow outward,
        respect the hole set.
        """
        target = spec.voxel_count
        max_iters = max(50, (target - len(self.voxels)) * 8)

        for _ in range(max_iters):
            if len(self.voxels) >= target:
                break

            candidates = set()
            for v in self.voxels:
                for n in _get_neighbors(v, self.grid_size, include_diagonal=False):
                    if (n not in self.voxels
                            and n not in self.hole_voxels
                            and self.in_bounds(n)):
                        candidates.add(n)

            if not candidates:
                break

            # Random choice (simple version — could add branching heuristics later)
            choice = self.rng.choice(sorted(candidates))
            self.add_voxel(choice)

    def _validate(self, spec: SkeletonSpec) -> bool:
        target_loops = max(0, spec.num_loops)
        if _count_components(self.voxels, self.grid_size) != 1:
            return False
        b1 = cubical_betti_1(self.voxels)
        return b1 == target_loops


# ==============================================================
# Test harness
# ==============================================================

def test_config(grid_size, num_loops, voxel_count, packing="medium", n_seeds=20):
    GRID = (grid_size, grid_size, grid_size)
    results = []
    for seed in range(n_seeds):
        rng = random.Random(seed * 1000 + num_loops * 7 + voxel_count * 13)
        spec = SkeletonSpec(
            archetype="bridge",
            voxel_count=voxel_count,
            grid_size=GRID,
            num_branches=max(3, num_loops + 2),
            num_loops=num_loops,
            num_components=1,
            direction_spread="planar",
            planarity="medium",
            packing=packing,
        )
        try:
            gen = HoleMarkedLoopSkeleton(GRID, rng=rng)
            voxels = gen.generate(spec)
        except Exception as e:
            print(f"  seed={seed}: FAILED — {type(e).__name__}: {e}")
            continue
        if not voxels:
            continue

        mu = calc_cycle_count(voxels, GRID)
        b1 = cubical_betti_1(voxels)
        n_comp = _count_components(voxels, GRID)

        results.append({
            "seed": seed,
            "n_voxels": len(voxels),
            "n_hole": len(gen.hole_voxels),
            "mu": mu,
            "b1": b1,
            "n_components": n_comp,
            "target_b1": num_loops,
            "b1_hit": b1 == num_loops,
            "connected": n_comp == 1,
        })
    return results


def summarize(results, label):
    if not results:
        print(f"{label}: no results")
        return
    n = len(results)
    hits = sum(1 for r in results if r["b1_hit"])
    connected = sum(1 for r in results if r["connected"])
    both = sum(1 for r in results if r["b1_hit"] and r["connected"])
    mean_mu = sum(r["mu"] for r in results) / n
    mean_b1 = sum(r["b1"] for r in results) / n
    mean_vox = sum(r["n_voxels"] for r in results) / n
    mean_hole = sum(r["n_hole"] for r in results) / n
    mu_range = (min(r["mu"] for r in results), max(r["mu"] for r in results))
    b1_range = (min(r["b1"] for r in results), max(r["b1"] for r in results))

    print(f"{label}")
    print(f"  n={n}  target_β₁={results[0]['target_b1']}")
    print(f"  β₁ hits target: {hits}/{n} ({100*hits/n:.1f}%)")
    print(f"  connected: {connected}/{n}")
    print(f"  BOTH β₁ and connected: {both}/{n} ({100*both/n:.1f}%)")
    print(f"  Mean μ = {mean_mu:.2f} [{mu_range[0]}-{mu_range[1]}]")
    print(f"  Mean β₁ = {mean_b1:.2f} [{b1_range[0]}-{b1_range[1]}]")
    print(f"  Mean voxel count = {mean_vox:.1f}")
    print(f"  Mean hole voxels reserved = {mean_hole:.1f}")


def main():
    print("=" * 72)
    print("HOLE-MARKED LOOP TEMPLATE PROTOTYPE v2")
    print("Fixed templates + hole voxels + rotation + variable sizing")
    print("=" * 72)

    print("\n--- 7³ grid, single ring (β₁=1) ---")
    for voxels in [8, 12, 16, 20]:
        r = test_config(7, 1, voxels, "medium", n_seeds=20)
        summarize(r, f"7³, target_voxels={voxels}")

    print("\n--- 11³ grid, single ring (β₁=1) ---")
    for voxels in [8, 15, 25, 40, 60]:
        r = test_config(11, 1, voxels, "medium", n_seeds=20)
        summarize(r, f"11³, target_voxels={voxels}")

    print("\n--- 7³ grid, double ring (β₁=2) ---")
    for voxels in [12, 16, 20, 25]:
        r = test_config(7, 2, voxels, "medium", n_seeds=20)
        summarize(r, f"7³, target_voxels={voxels}")

    print("\n--- 11³ grid, double ring (β₁=2) ---")
    for voxels in [14, 20, 30, 50]:
        r = test_config(11, 2, voxels, "medium", n_seeds=20)
        summarize(r, f"11³, target_voxels={voxels}")

    print("\n--- 11³ grid, triple ring (β₁=3) ---")
    for voxels in [20, 30, 45]:
        r = test_config(11, 3, voxels, "medium", n_seeds=15)
        summarize(r, f"11³, target_voxels={voxels}")

    print("\n--- 11³ grid, quad ring (β₁=4) ---")
    for voxels in [24, 35, 50]:
        r = test_config(11, 4, voxels, "medium", n_seeds=15)
        summarize(r, f"11³, target_voxels={voxels}")


if __name__ == "__main__":
    main()
