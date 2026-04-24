"""
Skeleton-First Cognitive Shape Generation Engine for MindFold 3D.

Copyright (c) 2024-2026 Scott N. Hwang, Parviz Safadel. All rights reserved.
Patent pending. See docs/PATENT_SPECIFICATION.md for claims.

This module implements the Skeleton-First Shape Generation System, a core
novel component of the MindFold 3D invention (Claims 2, 3, 8, 9, 10, 11).

Key inventive classes and methods:
  - TreeSkeleton:       Branching structures for structural decomposition (Claim 3c, 10)
  - AsymmetricSkeleton: Asymmetric structures for mirror discrimination (Claim 3c, 9)
  - LoopSkeleton:       Topological cycle structures for topological reasoning (Claim 3c, 11)
  - SkeletonRule._fill_remaining(): Scored candidate fill preserving structural
    intent via packing, direction spread, branching, and cycle constraints (Claim 8)
  - generate_shape_skeleton(): End-to-end skeleton generation pipeline (Claim 3)

Each skeleton rule takes a SkeletonSpec (cognitive dimensions mapped to structural
parameters) and builds shapes accordingly. No numeric feature optimization —
features are measured post-generation for verification only.
"""

import random
from typing import List, Tuple, Dict, Any, Set

from cognitive_mapping import SkeletonSpec
from shape_generation import (
    analyze_shape_features,
    _get_neighbors,
    _count_components,
    _calculate_cycle_count,
    _calculate_compactness_score,
    _calculate_planarity_score,
    _calculate_branching_factor,
    _get_pca_eigenvalues,
    _calculate_anisotropy_index,
    _calculate_shape_form_index,
    generate_mirror_reflection,
)

ALL_DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

# Plane definitions for pancake/planar direction selection
PLANES = {
    "XY": [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)],
    "XZ": [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)],
    "YZ": [(0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
}


class SkeletonRule:
    """Base class for skeleton-first shape generation rules."""

    def __init__(self, grid_size: Tuple[int, int, int]):
        self.grid_size = grid_size
        self.voxels: Set[Tuple[int, int, int]] = set()

    def in_bounds(self, v: Tuple[int, int, int]) -> bool:
        gx, gy, gz = self.grid_size
        return 0 <= v[0] < gx and 0 <= v[1] < gy and 0 <= v[2] < gz

    def add_voxel(self, v: Tuple[int, int, int]) -> bool:
        if self.in_bounds(v) and v not in self.voxels:
            self.voxels.add(v)
            return True
        return False

    def generate(self, spec: SkeletonSpec) -> Set[Tuple[int, int, int]]:
        """Build structure from cognitive spec, then fill remaining voxels."""
        self.voxels.clear()
        self._build(spec)
        self._fill_remaining(spec)
        return self.voxels

    def _build(self, spec: SkeletonSpec) -> None:
        """Build the structural skeleton. Override in subclasses."""
        raise NotImplementedError

    def _fill_remaining(self, spec: SkeletonSpec) -> None:
        """Add remaining voxels consistent with structural intent.

        Uses packing and direction_spread from the cognitive spec to guide
        where new voxels are placed. Preserves the skeleton's topology:
        - Low branching → fill extends from endpoints (no new junctions)
        - No cycles → reject candidates that would create loops
        """
        target = spec.voxel_count
        max_iters = max(50, (target - len(self.voxels)) * 8)

        # Precompute direction bias for fill
        preferred_axis = self._get_preferred_axis(spec)

        # Structural constraints from cognitive spec
        max_cycles = spec.num_loops  # cap cycles at intended level
        limit_branching = spec.num_branches <= 2  # strict endpoint-only for low SC

        for _ in range(max_iters):
            if len(self.voxels) >= target:
                break

            # Gather unique candidates adjacent to current shape
            candidates = set()
            for v in self.voxels:
                for n in _get_neighbors(v, self.grid_size, include_diagonal=False):
                    if n not in self.voxels and self.in_bounds(n):
                        candidates.add(n)

            if not candidates:
                break

            # Apply structural constraints
            valid = list(candidates)

            # Cap cycles at intended level
            current_cycles = _calculate_cycle_count(self.voxels, self.grid_size)
            if current_cycles >= max_cycles:
                filtered = []
                for c in valid:
                    temp = self.voxels | {c}
                    if _calculate_cycle_count(temp, self.grid_size) <= current_cycles:
                        filtered.append(c)
                if filtered:
                    valid = filtered

            if limit_branching:
                # Only extend from existing endpoints (voxels with degree 1).
                # This prevents the fill from creating new branch points by
                # attaching to the hub or mid-branch voxels.
                endpoints = set()
                for v in self.voxels:
                    degree = sum(
                        1 for n in _get_neighbors(v, self.grid_size, include_diagonal=False)
                        if n in self.voxels
                    )
                    if degree == 1:
                        endpoints.add(v)

                # Candidates that are neighbors of endpoints only
                endpoint_extension_candidates = []
                for c in valid:
                    neighbors_in_shape = [
                        n for n in _get_neighbors(c, self.grid_size, include_diagonal=False)
                        if n in self.voxels
                    ]
                    # Must touch exactly 1 voxel, and that voxel must be an endpoint
                    if len(neighbors_in_shape) == 1 and neighbors_in_shape[0] in endpoints:
                        endpoint_extension_candidates.append(c)

                if endpoint_extension_candidates:
                    valid = endpoint_extension_candidates

            choice = self._pick_fill_candidate(valid, spec, preferred_axis)
            self.add_voxel(choice)

    def _get_preferred_axis(self, spec: SkeletonSpec) -> int:
        """Get a preferred axis index for directional fill (0=x, 1=y, 2=z)."""
        return random.randint(0, 2)

    def _pick_fill_candidate(self, candidates: list, spec: SkeletonSpec,
                             preferred_axis: int) -> Tuple[int, int, int]:
        """Score and pick a fill candidate based on packing and direction spread."""
        # Compute center of mass
        cx = sum(v[0] for v in self.voxels) / len(self.voxels)
        cy = sum(v[1] for v in self.voxels) / len(self.voxels)
        cz = sum(v[2] for v in self.voxels) / len(self.voxels)
        com = (cx, cy, cz)

        # Precompute bounding box for compact spread axis-balance metric.
        if spec.direction_spread == "compact" and self.voxels:
            _bb_mins = [min(v[ax] for v in self.voxels) for ax in range(3)]
            _bb_maxs = [max(v[ax] for v in self.voxels) for ax in range(3)]
        else:
            _bb_mins = _bb_maxs = None

        # Precompute symmetry data for spatial_form targeting.
        # For low SF (want symmetry 0.7-1.0): reward placements that have a mirror.
        # For high/expert SF (want symmetry 0.0-0.15): penalize such placements.
        sf_level = spec.shape_difficulties.get("spatial_form", "medium")
        use_symmetry_bias = sf_level in ("low", "high", "expert") and self.voxels
        if use_symmetry_bias:
            vlist = list(self.voxels)
            # axis_mids[ax] = min_ax + max_ax = integer 2*center_ax
            axis_mids = [
                min(v[ax] for v in vlist) + max(v[ax] for v in vlist)
                for ax in range(3)
            ]

        scored = []
        for c in candidates:
            score = 0.0

            # Neighbor count for packing
            neighbor_count = sum(
                1 for n in _get_neighbors(c, self.grid_size, include_diagonal=False)
                if n in self.voxels
            )

            # Packing bias — heavily weighted to dominate placement
            if spec.packing == "dense":
                # Strongly cluster: prefer positions with 2+ existing neighbors
                # and positions close to center of mass
                score += neighbor_count * 20
                dist_to_com = sum((c[i] - com[i]) ** 2 for i in range(3)) ** 0.5
                score -= dist_to_com * 5  # penalize distant positions
            elif spec.packing == "sparse":
                # Strongly extend: prefer positions with exactly 1 neighbor (tips)
                # and positions far from center of mass
                if neighbor_count == 1:
                    score += 15
                else:
                    score -= neighbor_count * 5
                dist_to_com = sum((c[i] - com[i]) ** 2 for i in range(3)) ** 0.5
                score += dist_to_com * 3  # reward distant positions
            else:
                # Medium: mild compactness preference
                score += neighbor_count * 5

            # Direction spread bias
            if spec.direction_spread == "elongated_3d":
                # Extend along primary axis BUT also reward off-axis positions
                # for 3D spread (anti-planar). High anisotropy + low planarity.
                dist_on_axis = abs(c[preferred_axis] - com[preferred_axis])
                # Also check: does this candidate add depth in a new axis?
                off_axis_dists = [abs(c[i] - com[i]) for i in range(3) if i != preferred_axis]
                max_off = max(off_axis_dists) if off_axis_dists else 0
                score += dist_on_axis * 5  # elongation
                score += max_off * 3       # reward 3D spread
            elif spec.direction_spread == "planar":
                # Stay in one plane — penalize the out-of-plane axis
                off_plane_axis = preferred_axis
                dist_off_plane = abs(c[off_plane_axis] - com[off_plane_axis])
                score -= dist_off_plane * 15  # strongly penalize leaving the plane
            elif spec.direction_spread == "compact":
                # Spherical/isotropic growth: minimize anisotropy by keeping
                # the overall shape bounding box balanced across all three axes.
                score += neighbor_count * 15  # strong clustering
                dist_to_com = sum((c[i] - com[i]) ** 2 for i in range(3)) ** 0.5
                score -= dist_to_com * 10     # strongly prefer near-center
                # Penalize shapes where one axis extent is much larger than others
                # AFTER adding this candidate.  Uses overall bounding box, not the
                # candidate's distance to COM (which shifts as the shape grows).
                if _bb_mins is not None:
                    new_extents = [
                        max(_bb_maxs[ax], c[ax]) - min(_bb_mins[ax], c[ax])
                        for ax in range(3)
                    ]
                    score -= (max(new_extents) - min(new_extents)) * 10

            # Symmetry bias: steer symmetry_score toward spatial_form target.
            # A candidate "has a mirror" if its reflection across any axis-aligned
            # midplane of the current shape is already occupied.
            if use_symmetry_bias:
                has_mirror = False
                for ax in range(3):
                    mirror = list(c)
                    mirror[ax] = axis_mids[ax] - c[ax]
                    if tuple(mirror) in self.voxels:
                        has_mirror = True
                        break
                if sf_level == "low":
                    # Want HIGH symmetry: strongly reward symmetric placements
                    score += 20 if has_mirror else -15
                else:
                    # Want LOW symmetry (high/expert SF): penalize symmetric placements
                    score += -20 if has_mirror else 8

            scored.append((c, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        # For dense packing, be very deterministic (top 1-2 candidates)
        # For sparse, allow more randomness
        if spec.packing == "dense":
            top_n = max(1, min(2, len(scored)))
        elif spec.packing == "sparse":
            top_n = max(1, len(scored) // 3)
        else:
            top_n = max(1, len(scored) // 4)
        return random.choice(scored[:top_n])[0]

    def _validate(self, spec: SkeletonSpec) -> bool:
        """Check structural guarantees. Override in subclasses."""
        return _count_components(self.voxels, self.grid_size) == 1


def _select_directions(spec: SkeletonSpec, count: int) -> List[Tuple[int, int, int]]:
    """Choose directions based on direction_spread from cognitive spec.

    Direction spreads map to cognitive spatial_form difficulty:
      planar:       flat shape in one plane (low SF difficulty → high planarity)
      moderate:     some 3D spread (medium SF difficulty)
      elongated_3d: elongated with off-axis branches (high SF → low planarity, high anisotropy)
    """
    spread = spec.direction_spread

    if spread == "planar":
        # Stay in one plane — high planarity, low anisotropy
        plane_name = random.choice(list(PLANES.keys()))
        plane_dirs = list(PLANES[plane_name])
        random.shuffle(plane_dirs)
        return plane_dirs[:count]

    if spread == "elongated_3d":
        # Primary axis for elongation + perpendicular branches for 3D spread
        # This creates high anisotropy (elongated) AND low planarity (3D)
        primary_axis = random.choice([0, 1, 2])
        primary = [0, 0, 0]
        primary[primary_axis] = random.choice([1, -1])
        primary = tuple(primary)
        # Secondary directions are perpendicular to primary (forces off-plane)
        perp_dirs = [d for d in ALL_DIRS if d[primary_axis] == 0]
        random.shuffle(perp_dirs)
        # Interleave: primary first, then perpendicular for 3D spread
        result = [primary]
        for d in perp_dirs:
            result.append(d)
            if len(result) >= count:
                break
        return result[:count]

    if spread == "compact":
        # All directions equally likely — isotropic spherical skeleton
        dirs = list(ALL_DIRS)
        random.shuffle(dirs)
        return dirs[:count]

    # "moderate" (default): random with some diversity across axes
    dirs = list(ALL_DIRS)
    random.shuffle(dirs)
    return dirs[:count]


class TreeSkeleton(SkeletonRule):
    """
    Generates branching structures for structural decomposition tasks.

    Direct from cognitive params:
      spec.num_branches -> number of branches from hub
      spec.packing -> branch length: sparse=long, dense=short
      spec.direction_spread -> branch direction selection
    """

    def _build(self, spec: SkeletonSpec) -> None:
        center = (self.grid_size[0] // 2, self.grid_size[1] // 2, self.grid_size[2] // 2)
        self.add_voxel(center)

        if spec.direction_spread == "compact":
            # Compact/isotropic: just the center voxel; fill builds a spherical blob
            return

        if spec.packing == "dense":
            # Dense: minimal skeleton (just hub + immediate neighbors as branch stubs)
            # Let the fill create a compact cluster around the hub
            skeleton_budget = min(spec.num_branches + 1, spec.voxel_count)
            directions = _select_directions(spec, spec.num_branches)
            for direction in directions[:spec.num_branches]:
                if len(self.voxels) >= skeleton_budget:
                    break
                next_v = (center[0] + direction[0], center[1] + direction[1], center[2] + direction[2])
                self.add_voxel(next_v)
        else:
            # Medium/sparse: full branching skeleton
            skeleton_budget = max(4, int(spec.voxel_count * 0.6))
            num_branches = spec.num_branches
            directions = _select_directions(spec, num_branches)

            remaining_budget = skeleton_budget - 1
            voxels_per_branch = max(1, remaining_budget // num_branches)
            if spec.packing == "sparse":
                voxels_per_branch = max(2, voxels_per_branch)

            for direction in directions[:num_branches]:
                curr = center
                for _ in range(voxels_per_branch):
                    if len(self.voxels) >= skeleton_budget:
                        break
                    next_v = (curr[0] + direction[0], curr[1] + direction[1], curr[2] + direction[2])
                    if self.add_voxel(next_v):
                        curr = next_v

    def _get_preferred_axis(self, spec: SkeletonSpec) -> int:
        """Use the primary branch direction as preferred axis for fill."""
        directions = _select_directions(spec, 1)
        if directions:
            d = directions[0]
            # Return the axis with the largest absolute component
            return max(range(3), key=lambda i: abs(d[i]))
        return random.randint(0, 2)


class AsymmetricSkeleton(SkeletonRule):
    """
    Generates asymmetric shapes (no reflection symmetry) for mirror
    discrimination and mental rotation tasks.

    Direct from cognitive params:
      spec.direction_spread -> leg length ratios (see _compute_leg_lengths):
        "elongated_3d": one dominant leg (~55% of budget) + two shorter legs
        "planar":       two long legs in-plane, one very short out-of-plane
        other (e.g. "moderate"): roughly equal legs across all three axes
      spec.planarity -> axis choices
        high: 2 legs in same plane
        low/medium: spread across all 3 axes
      spec.packing -> skeleton vs fill ratio
    """

    def _build(self, spec: SkeletonSpec) -> None:
        if spec.voxel_count < 4:
            self._random_walk(spec.voxel_count)
            return

        # For expert SF: use a staircase/helix skeleton.
        # Straight-line legs are palindromic under axis reflection (high symmetry),
        # so the optimizer can't reduce symmetry because those voxels are articulation
        # points.  A staircase avoids this: it extends in a primary axis (high anisotropy)
        # while stepping perpendicular at regular intervals (breaks axis symmetry).
        sf_level = spec.shape_difficulties.get("spatial_form", "medium")
        if sf_level == "expert":
            self._build_staircase(spec)
            return

        skeleton_budget = max(4, int(spec.voxel_count * 0.5))
        center = (self.grid_size[0] // 2, self.grid_size[1] // 2, self.grid_size[2] // 2)
        self.add_voxel(center)

        remaining = skeleton_budget - 1
        lengths = self._compute_leg_lengths(remaining, spec.direction_spread)
        axes = self._choose_axes(spec.planarity)

        for axis_dir, length in zip(axes, lengths):
            curr = center
            for _ in range(length):
                if len(self.voxels) >= skeleton_budget:
                    break
                next_v = (curr[0] + axis_dir[0], curr[1] + axis_dir[1], curr[2] + axis_dir[2])
                if self.add_voxel(next_v):
                    curr = next_v

    def _build_staircase(self, spec: SkeletonSpec) -> None:
        """Build a biased-random-walk skeleton for expert SF.

        70% of steps advance in the primary axis (for high anisotropy), 30% deviate
        in random perpendicular directions (breaks axis-aligned reflection symmetry).
        The random deviations prevent the deterministic mirror pairs created by a
        fixed staircase pattern.
        """
        skeleton_budget = max(4, int(spec.voxel_count * 0.6))
        gx, gy, gz = self.grid_size
        primary_ax = random.randint(0, 2)
        perp_axes = [i for i in range(3) if i != primary_ax]

        # Start near one end so the primary walk has room to extend
        start = [gx // 2, gy // 2, gz // 2]
        start[primary_ax] = 1
        center = tuple(start)
        self.add_voxel(center)

        curr = center
        primary_dir = [0, 0, 0]
        primary_dir[primary_ax] = 1

        while len(self.voxels) < skeleton_budget:
            r = random.random()
            if r < 0.7:
                d = primary_dir
            elif r < 0.85:
                d = [0, 0, 0]
                d[perp_axes[0]] = random.choice([1, -1])
            else:
                d = [0, 0, 0]
                d[perp_axes[1]] = random.choice([1, -1])

            next_v = (curr[0] + d[0], curr[1] + d[1], curr[2] + d[2])
            if self.add_voxel(next_v):
                curr = next_v
            else:
                # Fallback: try primary direction
                pv = (curr[0] + primary_dir[0], curr[1] + primary_dir[1], curr[2] + primary_dir[2])
                if self.add_voxel(pv):
                    curr = pv
                else:
                    break  # stuck — skeleton complete

    def _compute_leg_lengths(self, remaining: int, direction_spread: str) -> List[int]:
        """Compute three distinct leg lengths for chirality."""
        if remaining <= 3:
            lengths = [1, 1, max(1, remaining - 2)]
            if lengths[0] == lengths[1] == lengths[2] and remaining > 1:
                lengths[0] = max(1, lengths[0] - 1)
                lengths[2] += 1
            return sorted(lengths, reverse=True)

        if direction_spread == "elongated_3d":
            # One dominant leg + perpendicular extensions for 3D
            primary = max(2, int(remaining * 0.55))
            rest = remaining - primary
            return sorted([primary, max(1, rest // 2), max(1, rest - rest // 2)], reverse=True)
        elif direction_spread == "planar":
            # Two long legs in-plane, one very short out-of-plane
            each = max(1, (remaining - 1) // 2)
            short = max(1, remaining - each * 2)
            return sorted([each + 1, each, short], reverse=True)
        else:
            # Moderate: some variation, all three axes represented
            base = max(1, remaining // 3)
            return sorted([base + 1, base, max(1, remaining - base * 2 - 1)], reverse=True)

    def _choose_axes(self, planarity: str) -> List[Tuple[int, int, int]]:
        """Choose three axis directions for chiral legs."""
        axis_options = [
            (random.choice([1, -1]), 0, 0),
            (0, random.choice([1, -1]), 0),
            (0, 0, random.choice([1, -1])),
        ]

        if planarity == "high":
            # High planarity: two legs in same plane, one perpendicular
            # Pick a plane, assign two legs there, one out-of-plane
            plane_idx = random.randint(0, 2)
            out_of_plane = axis_options[plane_idx]
            in_plane = [a for i, a in enumerate(axis_options) if i != plane_idx]
            # Put the shortest leg (last) out of plane
            return in_plane + [out_of_plane]
        else:
            # Low/medium planarity: spread across all three axes
            random.shuffle(axis_options)
            return axis_options

    def _validate(self, spec: SkeletonSpec) -> bool:
        if _count_components(self.voxels, self.grid_size) != 1:
            return False
        voxels_list = [list(v) for v in self.voxels]
        mirror = generate_mirror_reflection(voxels_list, self.grid_size)
        return mirror is not None  # Not None means shape IS chiral

    def _random_walk(self, target_voxels: int) -> None:
        center = (self.grid_size[0] // 2, self.grid_size[1] // 2, self.grid_size[2] // 2)
        self.add_voxel(center)
        while len(self.voxels) < target_voxels:
            base = random.choice(list(self.voxels))
            dirs = list(ALL_DIRS)
            random.shuffle(dirs)
            for dx, dy, dz in dirs:
                new_v = (base[0] + dx, base[1] + dy, base[2] + dz)
                if self.add_voxel(new_v):
                    break


class LoopSkeleton(SkeletonRule):
    """
    Forces topological cycles (holes/loops) for topological reasoning tasks.

    Direct from cognitive params:
      spec.num_loops -> number of rectangular loops. Values produced by
        get_skeleton_spec are 0 (low), 1 (medium), 2 (high), 4 (expert);
        the inner helpers cap at 5.
      spec.packing -> loop dimensions: dense=minimal 3x2, sparse=wider 3x3
      spec.direction_spread -> loop orientation and extension direction
    """

    def generate(self, spec: SkeletonSpec) -> Set[Tuple[int, int, int]]:
        """Build loops, optionally add 3D seeds, then fill remaining."""
        self.voxels.clear()
        self._build(spec)
        if spec.direction_spread != "planar":
            self._add_perpendicular_seeds(spec)
        self._fill_remaining(spec)
        return self.voxels

    def _build(self, spec: SkeletonSpec) -> None:
        if spec.voxel_count < 4:
            # Too few voxels for any loop — fall back to tree
            tree = TreeSkeleton(self.grid_size)
            tree._build(spec)
            self.voxels = tree.voxels
            return

        center = (self.grid_size[0] // 2, self.grid_size[1] // 2, self.grid_size[2] // 2)

        # For expert-level bridges (3+ loops) with non-planar spread, use compact
        # 2-wide chain skeletons that leave budget for 3D fill. Standard bridges
        # use the original 3×3 perimeter chains for precise cycle control.
        if spec.direction_spread != "planar" and spec.num_loops >= 3:
            self._build_compact_chain(center, spec)
        elif spec.num_loops >= 3 and spec.voxel_count >= 18:
            self._build_multi_loop(center, spec)
        elif spec.num_loops >= 2 and spec.voxel_count >= 12:
            self._build_double_loop(center, spec)
        else:
            self._build_single_loop(center, spec)

    def _build_compact_chain(self, center: Tuple[int, int, int], spec: SkeletonSpec) -> None:
        """Build a compact 2-wide filled chain that creates cycles efficiently.

        An (N+1)×2 filled rectangle creates N independent cycles using only
        2(N+1) voxels. This is much more efficient than 3×3 perimeters,
        leaving budget for perpendicular fill that creates 3D spread.

        For 4 expert cycles: 5×2 = 10 voxels (vs 23 for perimeter chain).
        """
        plane_axes = self._choose_loop_plane(spec.direction_spread)
        ax1, ax2 = plane_axes

        chain_length = spec.num_loops + 1  # N+1 for N cycles
        start = list(center)
        start[ax1] -= chain_length // 2
        # No ax2 offset needed — chain is only 2 wide

        for i in range(chain_length):
            for j in range(2):
                v = list(center)
                v[ax1] = start[ax1] + i
                v[ax2] = center[ax2] + j
                self.add_voxel(tuple(v))

    def _build_single_loop(self, center: Tuple[int, int, int], spec: SkeletonSpec) -> None:
        """Build a rectangular loop perimeter in a chosen plane."""
        # Loop size from packing
        if spec.packing == "dense":
            loop_w, loop_h = 3, 2
        else:
            loop_w, loop_h = 3, 3

        # Choose loop orientation based on direction_spread
        plane_axes = self._choose_loop_plane(spec.direction_spread)
        ax1, ax2 = plane_axes

        start = list(center)
        start[ax1] -= loop_w // 2
        start[ax2] -= loop_h // 2

        # Draw rectangle perimeter in the chosen plane
        for i in range(loop_w):
            v1, v2 = list(center), list(center)
            v1[ax1] = start[ax1] + i
            v1[ax2] = start[ax2]
            v2[ax1] = start[ax1] + i
            v2[ax2] = start[ax2] + loop_h - 1
            self.add_voxel(tuple(v1))
            self.add_voxel(tuple(v2))
        for j in range(1, loop_h - 1):
            v1, v2 = list(center), list(center)
            v1[ax1] = start[ax1]
            v1[ax2] = start[ax2] + j
            v2[ax1] = start[ax1] + loop_w - 1
            v2[ax2] = start[ax2] + j
            self.add_voxel(tuple(v1))
            self.add_voxel(tuple(v2))

    def _build_double_loop(self, center: Tuple[int, int, int], spec: SkeletonSpec) -> None:
        """Build two adjacent rectangular loops sharing a wall (figure-8)."""
        plane_axes = self._choose_loop_plane(spec.direction_spread)
        ax1, ax2 = plane_axes

        start = list(center)
        start[ax1] -= 2
        start[ax2] -= 1

        # First loop: 3x3 perimeter
        for i in range(3):
            v1, v2 = list(center), list(center)
            v1[ax1] = start[ax1] + i
            v1[ax2] = start[ax2]
            v2[ax1] = start[ax1] + i
            v2[ax2] = start[ax2] + 2
            self.add_voxel(tuple(v1))
            self.add_voxel(tuple(v2))
        v_left, v_right = list(center), list(center)
        v_left[ax1] = start[ax1]
        v_left[ax2] = start[ax2] + 1
        v_right[ax1] = start[ax1] + 2
        v_right[ax2] = start[ax2] + 1
        self.add_voxel(tuple(v_left))
        self.add_voxel(tuple(v_right))

        # Second loop: extend along ax1, sharing the right wall
        for i in range(3, 5):
            v1, v2 = list(center), list(center)
            v1[ax1] = start[ax1] + i
            v1[ax2] = start[ax2]
            v2[ax1] = start[ax1] + i
            v2[ax2] = start[ax2] + 2
            self.add_voxel(tuple(v1))
            self.add_voxel(tuple(v2))
        v_end = list(center)
        v_end[ax1] = start[ax1] + 4
        v_end[ax2] = start[ax2] + 1
        self.add_voxel(tuple(v_end))

    def _build_multi_loop(self, center: Tuple[int, int, int], spec: SkeletonSpec) -> None:
        """Build a chain of adjacent rectangular loops sharing walls.

        Each loop is a 3x3 perimeter (8 voxels), sharing 3 voxels with
        the previous loop. Supports 3-5 loops for expert difficulty.
        Voxel cost: 8 + 5*(num_loops-1) = 8, 13, 18, 23, 28 for 1-5 loops.
        """
        plane_axes = self._choose_loop_plane(spec.direction_spread)
        ax1, ax2 = plane_axes

        num_loops = min(spec.num_loops, 5)  # Cap at 5

        # Start position: offset so chain is roughly centered
        start = list(center)
        chain_length = 2 * num_loops + 1  # Total ax1 extent
        start[ax1] -= chain_length // 2
        start[ax2] -= 1

        for loop_idx in range(num_loops):
            offset = loop_idx * 2  # Each loop advances 2 along ax1
            for i in range(3):
                # Top and bottom edges of this loop
                v_top, v_bot = list(center), list(center)
                v_top[ax1] = start[ax1] + offset + i
                v_top[ax2] = start[ax2]
                v_bot[ax1] = start[ax1] + offset + i
                v_bot[ax2] = start[ax2] + 2
                self.add_voxel(tuple(v_top))
                self.add_voxel(tuple(v_bot))
            # Left wall (first loop) or shared wall is already placed by previous loop
            if loop_idx == 0:
                v_left = list(center)
                v_left[ax1] = start[ax1]
                v_left[ax2] = start[ax2] + 1
                self.add_voxel(tuple(v_left))
            # Right wall of this loop
            v_right = list(center)
            v_right[ax1] = start[ax1] + offset + 2
            v_right[ax2] = start[ax2] + 1
            self.add_voxel(tuple(v_right))

    def _add_perpendicular_seeds(self, spec: SkeletonSpec) -> None:
        """Add voxels perpendicular to the loop plane to break planarity.

        Loop skeletons are built in a single plane, giving planarity=1.0.
        Adding perpendicular seeds creates 3D spread that the fill phase
        will extend further, reducing planarity toward the expert target.
        Seeds extend from perimeter voxels so they don't fill loop holes.
        """
        remaining = spec.voxel_count - len(self.voxels)
        if remaining <= 2:
            return

        # Detect the loop plane: find which axis has zero/minimal spread
        coords = list(self.voxels)
        axis_ranges = [
            max(v[i] for v in coords) - min(v[i] for v in coords)
            for i in range(3)
        ]
        # Perpendicular axis = the one with smallest range (likely 0 for flat loops)
        perp_axis = axis_ranges.index(min(axis_ranges))

        # Budget: use up to 1/3 of remaining budget for seeds
        target_seeds = min(remaining // 3, max(2, len(self.voxels) // 4))

        edge_voxels = list(self.voxels)
        random.shuffle(edge_voxels)
        seeds_added = 0

        for v in edge_voxels:
            if seeds_added >= target_seeds:
                break
            for delta in [1, -1]:
                if seeds_added >= target_seeds:
                    break
                new_v = list(v)
                new_v[perp_axis] += delta
                new_v = tuple(new_v)
                if self.in_bounds(new_v) and new_v not in self.voxels:
                    self.voxels.add(new_v)
                    seeds_added += 1

    def _choose_loop_plane(self, direction_spread: str) -> Tuple[int, int]:
        """Choose which two axes form the loop plane."""
        if direction_spread == "rod":
            # Loop extends along primary axis
            primary = random.choice([0, 1, 2])
            secondary = random.choice([a for a in [0, 1, 2] if a != primary])
            return (primary, secondary)
        elif direction_spread == "pancake":
            # Loop lies flat in a chosen plane
            plane = random.choice([(0, 1), (0, 2), (1, 2)])
            return plane
        else:
            # Isotropic/moderate: random plane
            plane = random.choice([(0, 1), (0, 2), (1, 2)])
            return plane

    def _fill_remaining(self, spec: SkeletonSpec) -> None:
        """Fill that preserves cycle count while respecting packing/direction.

        Maintains cycles within [target_loops, target_loops + 2] to prevent
        the compact chain skeleton from producing excessive cycles during fill.
        """
        target = spec.voxel_count
        target_loops = max(1, spec.num_loops)
        # Compact chains (expert) benefit from generous cycle allowance: fill
        # creating 2×2 blocks is the mechanism that produces 3D spread.
        # Standard perimeter chains need tighter caps for cycle precision.
        max_cycles = target_loops * 3 if spec.num_loops >= 3 else target_loops + 1
        preferred_axis = self._get_preferred_axis(spec)
        max_iters = max(30, (target - len(self.voxels)) * 5)

        for _ in range(max_iters):
            if len(self.voxels) >= target:
                break

            current_cycles = _calculate_cycle_count(self.voxels, self.grid_size)

            candidates = set()
            for v in self.voxels:
                for n in _get_neighbors(v, self.grid_size, include_diagonal=False):
                    if n not in self.voxels and self.in_bounds(n):
                        candidates.add(n)

            if not candidates:
                break

            # Filter: maintain cycles within [target_loops, max_cycles]
            valid = []
            for c in candidates:
                temp = self.voxels | {c}
                new_cycles = _calculate_cycle_count(temp, self.grid_size)
                if new_cycles >= target_loops and new_cycles <= max_cycles:
                    valid.append(c)

            if not valid:
                # Relax: accept any candidate that doesn't decrease below target
                valid = [c for c in candidates
                         if _calculate_cycle_count(self.voxels | {c}, self.grid_size) >= target_loops]
            if not valid:
                valid = list(candidates)

            choice = self._pick_fill_candidate(valid, spec, preferred_axis)

            # For high SC (num_loops < 3): boost candidates adjacent to degree-2 nodes.
            # This steers branches toward corners, raising branching_factor into [4,5].
            if spec.num_loops < 3 and valid:
                current_branching = _calculate_branching_factor(self.voxels, self.grid_size)
                if current_branching < 4:
                    best_score = -1
                    best_c = choice
                    for c in valid:
                        nb_degrees = [
                            sum(1 for nb2 in _get_neighbors(nb, self.grid_size, include_diagonal=False)
                                if nb2 in self.voxels)
                            for nb in _get_neighbors(c, self.grid_size, include_diagonal=False)
                            if nb in self.voxels
                        ]
                        branching_score = sum(1 for d in nb_degrees if d == 2)
                        if branching_score > best_score:
                            best_score = branching_score
                            best_c = c
                    if best_score > 0:
                        choice = best_c

            self.add_voxel(choice)

    def _validate(self, spec: SkeletonSpec) -> bool:
        target_loops = max(1, spec.num_loops)
        if _count_components(self.voxels, self.grid_size) != 1:
            return False
        return _calculate_cycle_count(self.voxels, self.grid_size) >= target_loops


def _compute_symmetry_score_for_gate(vs: Set[Tuple[int, int, int]]) -> float:
    """Bilateral symmetry score used by the post-optimization symmetry gate."""
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


def _optimize_geometry(
    voxels: Set[Tuple[int, int, int]],
    grid_size: Tuple[int, int, int],
    shape_difficulties: Dict[str, str],
    preserve_chirality: bool = False,
    max_iterations: int = 300,
) -> Set[Tuple[int, int, int]]:
    """Post-generation geometric optimizer: local search to hit cognitive targets.

    Preserves topological structure (connectivity, cycle count, branching factor)
    while adjusting geometric features (compactness, planarity, form index,
    anisotropy) toward their cognitive-difficulty target ranges.

    Uses greedy swap search: each iteration removes a non-articulation voxel and
    adds a new adjacent voxel, accepting swaps that reduce the distance-to-target
    score across all geometric features.

    This closes the round-trip: cognitive params → skeleton → optimize → features
    that reverse-map back to the original cognitive params.
    """
    from cognitive_mapping import SHAPE_DIMENSIONS

    # Build target ranges for geometric features only
    GEO_FEATURES = {
        "compactness_score", "planarity_score", "anisotropy_index", "shape_form_index",
        "symmetry_score",
    }
    targets: Dict[str, Tuple[float, float]] = {}
    for dim_name, dim_config in SHAPE_DIMENSIONS.items():
        level = shape_difficulties.get(dim_name, "medium")
        for feat_name, feat_config in dim_config["features"].items():
            if feat_name not in GEO_FEATURES:
                continue
            range_val = feat_config.get(level)
            if range_val is None:
                continue
            if isinstance(range_val, tuple):
                targets[feat_name] = range_val
            else:
                targets[feat_name] = (float(range_val), float(range_val))

    if not targets:
        return voxels

    # Record topology baseline to preserve.  Tolerances are set below:
    # cycle_tol is always 0 (exact cycle count preserved); branch_tol is 999
    # (effectively unbounded) for low spatial_form to allow compact reshaping,
    # and 0 (exact branch count preserved) otherwise.
    topo_cycles = _calculate_cycle_count(voxels, grid_size)
    topo_branching = _calculate_branching_factor(voxels, grid_size)
    topo_components = _count_components(voxels, grid_size)
    # Strict topology preservation: no tolerance on cycles or branching,
    # EXCEPT for low spatial_form where compact shapes have high branching
    # and the strict constraint prevents the optimizer from achieving isotropy.
    # Relaxing branching for low SF lets the optimizer freely reshape the
    # cluster while preserving connectivity and cycle count.
    cycle_tol = 0
    sf_level = shape_difficulties.get("spatial_form", "medium")
    branch_tol = 999 if sf_level == "low" else 0

    # Group targets by dimension so we can weight dimensions equally.
    # This prevents spatial_form (3 features) from dominating spatial_density
    # (1 feature) in the optimizer's scoring function.
    #
    # Use EXCLUSIVE target ranges: clip each range so it doesn't overlap with
    # the adjacent tier's range.  This prevents the optimizer from settling in
    # the overlap zone (e.g. anisotropy 0.75-0.9 is both high AND expert, so
    # the reverse-mapper classifies it as expert), which would yield 0% fidelity
    # for the intended tier even when the value is technically "in range".
    _LEVELS = ["low", "medium", "high", "expert"]

    def _exclusive_range(feat_config, level):
        """Return the portion of level's range that doesn't overlap higher tiers."""
        range_val = feat_config.get(level)
        if range_val is None:
            return None
        if isinstance(range_val, tuple):
            lo, hi = range_val
        else:
            lo = hi = float(range_val)
        idx = _LEVELS.index(level)
        # Clip upper bound: avoid overlapping the next higher tier's lower bound
        if idx < len(_LEVELS) - 1:
            next_range = feat_config.get(_LEVELS[idx + 1])
            if isinstance(next_range, tuple) and next_range[0] < hi:
                hi = next_range[0]
        # Clip lower bound: avoid overlapping the next lower tier's upper bound
        if idx > 0:
            prev_range = feat_config.get(_LEVELS[idx - 1])
            if isinstance(prev_range, tuple) and prev_range[1] > lo:
                lo = prev_range[1]
        if lo > hi:
            # Degenerate after clipping — fall back to original range
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
        """Bilateral symmetry: max voxel overlap when reflected across any axis midplane."""
        if len(vs) < 2:
            return 1.0
        vlist = list(vs)
        best = 0.0
        for axis in range(3):
            coords_ax = [v[axis] for v in vlist]
            mid = min(coords_ax) + max(coords_ax)  # integer 2*center
            reflected = frozenset(
                tuple(mid - v[axis] if i == axis else v[i] for i in range(3))
                for v in vlist
            )
            overlap = len(vs & reflected) / len(vs)
            if overlap > best:
                best = overlap
        return best

    def compute_geo(vs):
        feats = {}
        feats["compactness_score"] = _calculate_compactness_score(vs, grid_size)
        feats["planarity_score"] = _calculate_planarity_score(vs, grid_size)
        eigvals = _get_pca_eigenvalues(vs)
        feats["anisotropy_index"] = _calculate_anisotropy_index(eigvals)
        feats["shape_form_index"] = _calculate_shape_form_index(eigvals)
        feats["symmetry_score"] = _compute_symmetry_score(vs)
        return feats

    def geo_score(feats):
        """Dimension-weighted score: each cognitive dimension contributes equally.

        Within a dimension the per-feature deviations are averaged, then all
        dimensions are summed.  This gives spatial_density (1 feature) equal
        pull to spatial_form (3 features).  Lower is better; 0 = all in range.
        """
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
                dim_dev /= len(dt)  # average within dimension
            total += dim_dev
        return total

    def topo_ok(vs):
        """Check that topological invariants are preserved."""
        if _count_components(vs, grid_size) != topo_components:
            return False
        if abs(_calculate_cycle_count(vs, grid_size) - topo_cycles) > cycle_tol:
            return False
        if abs(_calculate_branching_factor(vs, grid_size) - topo_branching) > branch_tol:
            return False
        if preserve_chirality:
            vl = [list(v) for v in vs]
            if generate_mirror_reflection(vl, grid_size) is None:
                return False  # Lost chirality
        return True

    current = set(voxels)
    cur_feats = compute_geo(current)
    cur_score = geo_score(cur_feats)

    if cur_score < 1e-6:
        return current  # Already in range

    def _removable_voxels(vs):
        """Surface voxels whose removal preserves connectivity."""
        result = []
        for v in vs:
            temp = vs - {v}
            if temp and _count_components(temp, grid_size) == topo_components:
                result.append(v)
        return result

    def _candidate_positions(vs):
        """Empty cells adjacent to the shape."""
        cands = set()
        for v in vs:
            for n in _get_neighbors(v, grid_size, include_diagonal=False):
                if n not in vs:
                    cands.add(n)
        return cands

    def _neighbor_count_in(v, vs):
        """Count of face-adjacent occupied neighbors of v."""
        return sum(1 for n in _get_neighbors(v, grid_size, include_diagonal=False)
                   if n in vs)

    def _topo_filter(trial):
        """Check topology preservation."""
        if _count_components(trial, grid_size) != topo_components:
            return False
        if abs(_calculate_cycle_count(trial, grid_size) - topo_cycles) > cycle_tol:
            return False
        if abs(_calculate_branching_factor(trial, grid_size) - topo_branching) > branch_tol:
            return False
        if preserve_chirality:
            vl = [list(v) for v in trial]
            if generate_mirror_reflection(vl, grid_size) is None:
                return False
        return True

    # --- Phase 1: General geometric optimizer ---
    stale_count = 0
    for _ in range(max_iterations):
        if cur_score < 1e-6 or stale_count > 40:
            break

        removable = _removable_voxels(current)
        if not removable:
            break

        # Prioritize removing sparse voxels (fewest neighbors): they contribute
        # least to compactness and are easiest to relocate profitably.
        removable.sort(key=lambda v: _neighbor_count_in(v, current))

        improved = False
        for v_rm in removable[:15]:
            after_rm = current - {v_rm}
            cands = _candidate_positions(after_rm)
            cands.discard(v_rm)
            if not cands:
                continue

            # Score candidates: prefer high-neighbor positions (compact) and
            # positions that help with planarity / form.  Shuffle within
            # groups to maintain diversity.
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

    # --- Phase 2: Targeted compaction ---
    # If compactness is still below target, run a dedicated compaction pass.
    # Unlike a raw compaction, this only accepts moves that ALSO don't worsen
    # the overall geometric score — preventing compaction from undoing the
    # spatial_form gains achieved in Phase 1.
    compact_target = targets.get("compactness_score")
    if compact_target and cur_feats["compactness_score"] < compact_target[0]:
        for _ in range(max_iterations // 2):
            if cur_feats["compactness_score"] >= compact_target[0]:
                break

            removable = _removable_voxels(current)
            if not removable:
                break

            removable.sort(key=lambda v: _neighbor_count_in(v, current))

            improved = False
            for v_rm in removable[:8]:
                after_rm = current - {v_rm}
                cands = _candidate_positions(after_rm)
                cands.discard(v_rm)
                if not cands:
                    continue

                cand_list = sorted(
                    cands,
                    key=lambda c: _neighbor_count_in(c, after_rm),
                    reverse=True,
                )

                for v_add in cand_list[:15]:
                    trial = after_rm | {v_add}
                    if not _topo_filter(trial):
                        continue
                    tf = compute_geo(trial)
                    ts = geo_score(tf)
                    # Accept only if: compactness improved AND overall score
                    # didn't get worse (protects spatial_form gains).
                    if (tf["compactness_score"] > cur_feats["compactness_score"]
                            and ts <= cur_score + 1e-6):
                        current = trial
                        cur_feats = tf
                        cur_score = ts
                        improved = True
                        break

                if improved:
                    break

    return current


def _grow_isolated_blob(
    seed: Tuple[int, int, int],
    target_size: int,
    existing: Set[Tuple[int, int, int]],
    grid_size: Tuple[int, int, int],
) -> Set[Tuple[int, int, int]]:
    """Grow a connected blob from seed without touching existing voxels.

    The forbidden zone is the existing voxels PLUS all their face-adjacent
    neighbors, ensuring the resulting component stays disconnected.
    """
    forbidden: Set[Tuple[int, int, int]] = set(existing)
    for v in existing:
        for n in _get_neighbors(v, grid_size, include_diagonal=False):
            forbidden.add(n)

    # If seed is forbidden, scan nearby for a free cell
    if seed in forbidden:
        found = False
        for radius in range(1, 6):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        alt = (seed[0] + dx, seed[1] + dy, seed[2] + dz)
                        gx, gy, gz = grid_size
                        if 0 <= alt[0] < gx and 0 <= alt[1] < gy and 0 <= alt[2] < gz:
                            if alt not in forbidden:
                                seed = alt
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        else:
            return set()

    blob: Set[Tuple[int, int, int]] = {seed}
    for _ in range(target_size * 15):
        if len(blob) >= target_size:
            break
        candidates = [
            n
            for v in blob
            for n in _get_neighbors(v, grid_size, include_diagonal=False)
            if n not in blob and n not in forbidden
        ]
        if not candidates:
            break
        blob.add(random.choice(candidates))

    return blob


def _build_multi_component(spec: SkeletonSpec) -> Set[Tuple[int, int, int]]:
    """Generate a shape with num_components disconnected voxel clusters.

    The full grid is partitioned into n_comps axis-aligned strips separated by
    a 1-voxel gap.  Each strip is the full gy×gz extent but a fraction of gx.
    Each component runs its skeleton rule in the local strip coordinates, so the
    rule has the full gy×gz room to build cycles and branches.  Local
    voxels are offset by the strip's x-start to produce global coordinates.
    """
    import copy as _copy

    grid = spec.grid_size
    gx, gy, gz = grid
    n_comps = spec.num_components
    total = spec.voxel_count

    # Distribute voxels evenly; remainder goes to first components
    base = total // n_comps
    sizes = [base] * n_comps
    for i in range(total - base * n_comps):
        sizes[i] += 1

    # Partition gx into n_comps strips with 1-voxel gaps between them
    gap = 1
    usable = gx - gap * (n_comps - 1)
    strip_width = max(3, usable // n_comps)
    strips: List[Tuple[int, int]] = []   # (x_start, x_width)
    x = 0
    for i in range(n_comps):
        w = strip_width if i < n_comps - 1 else (gx - x)
        strips.append((x, w))
        x += w + gap

    all_voxels: Set[Tuple[int, int, int]] = set()
    for size, (x_start, x_width) in zip(sizes, strips):
        sub_spec = _copy.copy(spec)
        sub_spec.voxel_count = max(3, size)
        sub_spec.num_components = 1
        sub_grid = (x_width, gy, gz)  # strip-local grid: full gy×gz height/depth

        # Use compact dense fill for each component: this creates filled rectangular
        # blocks that simultaneously produce BOTH branching and cycles per unit volume.
        # Ring structures (loop skeleton) give cycles but zero branching_factor.
        sub_spec.direction_spread = "compact"
        sub_spec.packing = "dense"

        comp_skeleton = TreeSkeleton(sub_grid)
        comp_skeleton.generate(sub_spec)

        # Offset local x → global x; y, z unchanged
        for v in comp_skeleton.voxels:
            gv = (v[0] + x_start, v[1], v[2])
            if 0 <= gv[0] < gx and 0 <= gv[1] < gy and 0 <= gv[2] < gz:
                all_voxels.add(gv)

    return all_voxels


def generate_shape_skeleton(spec: SkeletonSpec, max_attempts: int = 5) -> Dict[str, Any]:
    """
    Generate a shape from a SkeletonSpec using the appropriate skeleton rule.

    Args:
        spec: Direct cognitive-to-skeleton parameters.
        max_attempts: Number of retry attempts for structural validation.

    Returns:
        Dict with voxels, grid_size, features, generation_mode, archetype.
    """
    archetype = spec.archetype.lower()
    grid_size = spec.grid_size

    skeleton_classes = {
        "tree": TreeSkeleton,
        "chiral": AsymmetricSkeleton,
        "bridge": LoopSkeleton,
    }
    cls = skeleton_classes.get(archetype, TreeSkeleton)

    # Multi-component shapes are assembled from isolated blobs, not single skeleton runs
    if spec.num_components > 1:
        best_voxels = _build_multi_component(spec)
    else:
        best_voxels = None
        for attempt in range(max_attempts):
            skeleton = cls(grid_size)
            skeleton.generate(spec)

            if skeleton._validate(spec):
                best_voxels = skeleton.voxels
                break
            if best_voxels is None:
                best_voxels = skeleton.voxels

    voxels_set = best_voxels if best_voxels else set()

    # Post-generation geometric optimization: steer anisotropy_index, symmetry_score,
    # and other geometric features toward target ranges for the spatial_form difficulty.
    # Topology preserved with zero tolerance for cycles and components; branching
    # tolerance is relaxed (branch_tol=999) for low SF to allow compact reshaping.
    voxels_set = _optimize_geometry(
        voxels_set,
        grid_size,
        spec.shape_difficulties,
        preserve_chirality=(archetype == "chiral"),
    )

    # Post-optimization symmetry gate for expert spatial_form.
    # The optimizer can often get symmetry to 0 (odd-span shapes) but sometimes
    # stalls at 0.18 (even-span self-mirrors).  Retry if symmetry > target.
    sf_level = spec.shape_difficulties.get("spatial_form", "medium")
    if sf_level == "expert" and spec.num_components == 1:
        sym_target_hi = 0.15
        for _sym_attempt in range(12):
            sym_val = _compute_symmetry_score_for_gate(voxels_set)
            if sym_val <= sym_target_hi:
                break
            # Regenerate and re-optimize with a different seed
            skeleton2 = cls(grid_size)
            skeleton2.generate(spec)
            cand = skeleton2.voxels if skeleton2.voxels else voxels_set
            cand = _optimize_geometry(
                cand,
                grid_size,
                spec.shape_difficulties,
                preserve_chirality=(archetype == "chiral"),
            )
            if _compute_symmetry_score_for_gate(cand) < sym_val:
                voxels_set = cand

    analyzed_sfs = analyze_shape_features(voxels_set, grid_size)

    return {
        "voxels": list(voxels_set),
        "grid_size": list(grid_size),
        "features": analyzed_sfs.to_dict(),
        "generation_mode": "skeleton",
        "archetype": archetype,
    }
