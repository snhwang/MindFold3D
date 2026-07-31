"""
Prototype v3: Hole-marked loop templates + accidental-hole closure.

Design additions over v2:
- After fill completes, scan for "accidental holes" — empty voxels enclosed
  by shape voxels that aren't part of any reserved hole set.
- Fill accidental holes to bring β₁ back to target_loops.

Since accidental-hole voxels aren't in reserved_holes, they can safely be
filled without breaking any intended topology.
"""

import random
import sys
from collections import deque
from pathlib import Path
from typing import Set, Tuple, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from cognitive_mapping import SkeletonSpec
from skeleton_generation import SkeletonRule, TreeSkeleton, _get_neighbors, _count_components
from shape_generation import _calculate_cycle_count as calc_cycle_count
from topology_extras import cubical_betti_1

# Reuse builders from v2
from prototype_hole_marked_loops import (
    build_rectangle_perimeter,
    build_rectangle_interior_hole,
    choose_random_plane,
    choose_ring_size,
    choose_ring_center,
)


Voxel = Tuple[int, int, int]


class HoleMarkedLoopSkeletonV3(SkeletonRule):
    """Hole-marked loop with accidental-hole closure.

    Fill runs freely, then a post-pass identifies and closes any
    empty region enclosed by shape voxels that is not part of the
    reserved hole set.
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
            tree = TreeSkeleton(self.grid_size)
            tree._build(spec)
            self.voxels = tree.voxels
            return self.voxels

        if target_loops == 1:
            self._build_single_ring(spec)
        elif target_loops == 2:
            self._build_double_ring(spec)
        elif target_loops == 3:
            self._build_triple_ring(spec)
        else:
            self._build_multi_ring(spec, num_rings=min(target_loops, 6))

        self._fill_hole_aware(spec)
        self._close_accidental_holes()
        return self.voxels

    # ---------- Template constructors (same as v2) ----------

    def _build_single_ring(self, spec: SkeletonSpec) -> None:
        plane = choose_random_plane(self.rng)
        min_s = 3
        max_bump = 0 if spec.packing == "dense" else 2
        size_a, size_b = choose_ring_size(self.grid_size, plane, self.rng, min_size=min_s)
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
        plane = choose_random_plane(self.rng)
        ax1, ax2 = plane
        ax3 = 3 - ax1 - ax2
        min_s = 3
        max_s = min(4, self.grid_size[ax1] // 2 - 1, self.grid_size[ax2] - 2)
        max_s = max(min_s, max_s)
        size = self.rng.randint(min_s, max_s)
        c1_lo = size
        c1_hi = self.grid_size[ax1] - 1 - size
        c2_lo = size // 2
        c2_hi = self.grid_size[ax2] - 1 - size // 2
        c1 = self.rng.randint(c1_lo, max(c1_lo, c1_hi))
        c2 = self.rng.randint(c2_lo, max(c2_lo, c2_hi))
        c3 = self.rng.randint(0, self.grid_size[ax3] - 1)
        offset = size - 1
        for shift in (-offset // 2, (offset + 1) // 2):
            center = [0, 0, 0]
            center[ax1] = c1 + shift
            center[ax2] = c2
            center[ax3] = c3
            skel = build_rectangle_perimeter(tuple(center), plane, size, size)
            hole = build_rectangle_interior_hole(tuple(center), plane, size, size, self.grid_size)
            for v in skel:
                if self.in_bounds(v):
                    self.add_voxel(v)
            self.hole_voxels.update(v for v in hole if self.in_bounds(v))

    def _build_triple_ring(self, spec: SkeletonSpec) -> None:
        self._build_ring_chain(spec, num_rings=3)

    def _build_multi_ring(self, spec: SkeletonSpec, num_rings: int) -> None:
        self._build_ring_chain(spec, num_rings=num_rings)

    def _build_ring_chain(self, spec: SkeletonSpec, num_rings: int) -> None:
        plane = choose_random_plane(self.rng)
        ax1, ax2 = plane
        ax3 = 3 - ax1 - ax2
        min_s = 3

        # Compute the largest ring size that lets all num_rings fit along ax1.
        # A chain of N size-S rings occupies (N * (S - 1) + 1) voxels along ax1.
        # We need this to fit within grid_size[ax1] with a small margin.
        margin = 2  # 1 on each side
        avail = self.grid_size[ax1] - margin
        max_s_for_chain = (avail - 1) // num_rings + 1
        # Also cap by grid size along ax2 (rings are square)
        max_s_grid = self.grid_size[ax2] - margin
        max_s = min(max_s_for_chain, max_s_grid, 4)  # cap ring size at 4
        if max_s < min_s:
            # Grid too small for requested chain; fall back to fewer rings
            max_s = min_s
            num_rings = max(1, (self.grid_size[ax1] - margin - 1) // (min_s - 1))
        size = self.rng.randint(min_s, max(min_s, max_s))

        step = size - 1
        chain_len = num_rings * step + 1
        start_ax1 = self.rng.randint(1, max(1, self.grid_size[ax1] - chain_len))

        c2 = self.rng.randint(size // 2, self.grid_size[ax2] - 1 - size // 2)
        c3 = self.rng.randint(0, self.grid_size[ax3] - 1)

        for i in range(num_rings):
            center = [0, 0, 0]
            center[ax1] = start_ax1 + size // 2 + i * step
            center[ax2] = c2
            center[ax3] = c3
            if center[ax1] + size // 2 >= self.grid_size[ax1]:
                break  # safety guard, shouldn't fire now
            skel = build_rectangle_perimeter(tuple(center), plane, size, size)
            hole = build_rectangle_interior_hole(tuple(center), plane, size, size, self.grid_size)
            for v in skel:
                if self.in_bounds(v):
                    self.add_voxel(v)
            self.hole_voxels.update(v for v in hole if self.in_bounds(v))

    # ---------- Fill ----------

    def _fill_hole_aware(self, spec: SkeletonSpec) -> None:
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
            choice = self.rng.choice(sorted(candidates))
            self.add_voxel(choice)

    # ---------- Accidental-hole closure ----------

    def _close_accidental_holes(self) -> None:
        """Find empty voxels enclosed by shape (not part of reserved_holes)
        and fill them.

        Method:
        1. Start a BFS from every grid-boundary empty voxel that is NOT
           in reserved_holes and NOT in self.voxels.
        2. Any empty voxel reachable from the boundary is "outside" — leave it.
        3. Empty voxels that are NOT reachable from boundary AND NOT in
           reserved_holes are accidental-hole voxels — fill them.

        Reserved-hole voxels are treated as passable when computing
        reachability (they're not shape, they're not accidental — they're
        exempt from the accidental-hole check).
        """
        gs = self.grid_size

        # Set of empty voxels (not in shape)
        # Reserved holes are counted as "empty" for reachability purposes
        # so that they don't block the BFS from reaching accidental holes
        # that happen to lie beyond a reserved column.

        # Actually, for the primary correctness case, we want:
        # - Reserved holes: leave alone (they are the topology we want)
        # - Empty voxels reachable from boundary: leave alone (they are outside)
        # - Empty voxels NOT reachable from boundary AND NOT reserved: fill them
        #
        # This means BFS should treat reserved-hole voxels as passable, because
        # if an accidental hole is on the other side of a reserved column,
        # it's still an accidental hole (the reserved column is a valid path).

        visited = set()
        queue = deque()

        # Seed BFS from all boundary empty (or reserved) voxels
        for x in range(gs[0]):
            for y in range(gs[1]):
                for z in range(gs[2]):
                    on_boundary = (x == 0 or x == gs[0] - 1
                                   or y == 0 or y == gs[1] - 1
                                   or z == 0 or z == gs[2] - 1)
                    if not on_boundary:
                        continue
                    v = (x, y, z)
                    if v in self.voxels:
                        continue
                    if v in visited:
                        continue
                    visited.add(v)
                    queue.append(v)

        # BFS through empty and reserved-hole voxels (both are non-shape)
        while queue:
            v = queue.popleft()
            for n in _get_neighbors(v, gs, include_diagonal=False):
                if n in visited:
                    continue
                if n in self.voxels:
                    continue
                # n is not shape → passable (either empty or reserved-hole)
                visited.add(n)
                queue.append(n)

        # Any non-shape voxel not visited is enclosed
        # If it's also not in reserved_holes, it's an accidental hole → fill
        accidental = set()
        for x in range(gs[0]):
            for y in range(gs[1]):
                for z in range(gs[2]):
                    v = (x, y, z)
                    if v in self.voxels:
                        continue
                    if v in visited:
                        continue
                    # v is enclosed
                    if v not in self.hole_voxels:
                        accidental.add(v)

        for v in accidental:
            self.add_voxel(v)

    def _validate(self, spec: SkeletonSpec) -> bool:
        target_loops = max(0, spec.num_loops)
        if _count_components(self.voxels, self.grid_size) != 1:
            return False
        b1 = cubical_betti_1(self.voxels)
        return b1 == target_loops


# ==============================================================
# Test harness (matches v2)
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
            gen = HoleMarkedLoopSkeletonV3(GRID, rng=rng)
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
    vox_range = (min(r["n_voxels"] for r in results), max(r["n_voxels"] for r in results))
    mu_range = (min(r["mu"] for r in results), max(r["mu"] for r in results))
    b1_range = (min(r["b1"] for r in results), max(r["b1"] for r in results))
    print(f"{label}")
    print(f"  n={n}  target_β₁={results[0]['target_b1']}")
    print(f"  β₁ hits target: {hits}/{n} ({100*hits/n:.1f}%)")
    print(f"  connected: {connected}/{n}")
    print(f"  BOTH β₁ and connected: {both}/{n} ({100*both/n:.1f}%)")
    print(f"  Mean μ = {mean_mu:.2f} [{mu_range[0]}-{mu_range[1]}]")
    print(f"  Mean β₁ = {mean_b1:.2f} [{b1_range[0]}-{b1_range[1]}]")
    print(f"  Voxel count = {mean_vox:.1f} [{vox_range[0]}-{vox_range[1]}]")


def main():
    print("=" * 72)
    print("HOLE-MARKED TEMPLATES v3: with accidental-hole closure")
    print("=" * 72)

    print("\n--- 7³ grid, β₁=1 ---")
    for voxels in [8, 12, 16, 20, 25]:
        r = test_config(7, 1, voxels, "medium", n_seeds=20)
        summarize(r, f"7³, target_voxels={voxels}")

    print("\n--- 11³ grid, β₁=1 ---")
    for voxels in [8, 15, 25, 40, 60, 80]:
        r = test_config(11, 1, voxels, "medium", n_seeds=20)
        summarize(r, f"11³, target_voxels={voxels}")

    print("\n--- 7³ grid, β₁=2 ---")
    for voxels in [12, 16, 20, 25, 30]:
        r = test_config(7, 2, voxels, "medium", n_seeds=20)
        summarize(r, f"7³, target_voxels={voxels}")

    print("\n--- 11³ grid, β₁=2 ---")
    for voxels in [14, 20, 30, 50, 70]:
        r = test_config(11, 2, voxels, "medium", n_seeds=20)
        summarize(r, f"11³, target_voxels={voxels}")

    print("\n--- 11³ grid, β₁=3 ---")
    for voxels in [20, 30, 45, 60]:
        r = test_config(11, 3, voxels, "medium", n_seeds=15)
        summarize(r, f"11³, target_voxels={voxels}")

    print("\n--- 11³ grid, β₁=4 ---")
    for voxels in [24, 35, 50, 70]:
        r = test_config(11, 4, voxels, "medium", n_seeds=15)
        summarize(r, f"11³, target_voxels={voxels}")


if __name__ == "__main__":
    main()
