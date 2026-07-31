"""
Topological invariants for voxel objects — supplementary implementations.

Copyright (c) 2026 Scott N. Hwang, Parviz Safadel. All rights reserved.
Patent pending.

This module computes the first Betti number of a voxel object treated as a
full cubical complex (with 2-cells and 3-cells added, not just the
1-skeleton).  It is used for:

  1. The production `betti_1` feature: shape_generation._calculate_betti_1
     delegates here, so every emitted shape's feature vector carries the true
     cubical β₁ (volumetric hole count).
  2. Hole-tier validation: HoleMarkedLoopSkeleton._validate certifies
     cubical β₁ = target on every emitted hole-tier shape.
  3. The paper's methodological supplement comparing cubical β₁ against the
     graph circuit rank μ = |E| − |V| + C on generator output. The two
     invariants are DISTINCT and diverge in both directions on real output:
     μ > β₁ on shapes with 2×2 face patches (graph cycles with no volumetric
     hole), and β₁ > μ where edge-adjacent voxels enclose a tunnel invisible
     to the face-adjacency graph.

The generator's real-time fill/scoring loops use circuit rank
(shape_generation._calculate_circuit_rank) because it is O(|V|); this module
runs once per emitted shape in feature analysis and at hole-tier validation.

Mathematical background
-----------------------
Each occupied voxel contributes a closed unit 3-cube to a cubical complex.
The Euler characteristic of this complex is

    χ = V − E + F − C₃

where V, E, F, C₃ are the counts of *distinct* 0-, 1-, 2-, and 3-cells
(corners, edges, faces, cubes).  In three dimensions the Euler–Poincaré
identity gives

    χ = β₀ − β₁ + β₂   ⇔   β₁ = β₀ + β₂ − χ

where β₀ is the number of connected components, β₁ the number of
independent 1-cycles (through-holes / handles), and β₂ the number of
enclosed cavities.  β₂ is computed as the number of complement components
that do not touch a padded bounding box (bounded holes in the complement).

References
----------
Chen, L. (2005). Digital topology: Introduction and survey. In *Handbook of
    Digital and Discrete Geometry*.
Kong, T. Y., & Rosenfeld, A. (1989). Digital topology: Introduction and
    survey. *Computer Vision, Graphics, and Image Processing*, 48(3), 357–393.
Boissonnat, J.-D., Chazal, F., & Yvinec, M. (2018). *Geometric and Topological
    Inference*. Cambridge University Press.
"""

from typing import List, Set, Tuple

# We import _count_components from shape_generation to keep component-counting
# semantics identical between the production circuit-rank metric and this
# validation metric.
from shape_generation import _count_components


def _cube_corners(v: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """The eight integer corners of the unit cube at voxel `v`."""
    x, y, z = v
    return [
        (x + dx, y + dy, z + dz)
        for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)
    ]


def _cube_edges(
    v: Tuple[int, int, int],
) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """The twelve edges of the unit cube at voxel `v`, each as a sorted pair of corners."""
    corners = _cube_corners(v)
    edges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []
    n = len(corners)
    for i in range(n):
        ci = corners[i]
        for j in range(i + 1, n):
            cj = corners[j]
            # A cube edge: corners differ in exactly one coordinate.
            if sum(1 for k in range(3) if ci[k] != cj[k]) == 1:
                edges.append((ci, cj) if ci < cj else (cj, ci))
    return edges


def _cube_faces(
    v: Tuple[int, int, int],
) -> List[Tuple[Tuple[int, int, int], ...]]:
    """The six faces of the unit cube at voxel `v`, each as a sorted 4-tuple of corners."""
    x, y, z = v
    # Each face is characterized by fixing one axis to either its low or high value.
    faces = []
    for axis in range(3):
        for offset in (0, 1):
            corners = []
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        d = (dx, dy, dz)
                        if d[axis] == offset:
                            corners.append((x + dx, y + dy, z + dz))
            faces.append(tuple(sorted(corners)))
    return faces


def euler_characteristic(voxels_set: Set[Tuple[int, int, int]]) -> int:
    """Euler characteristic χ = V − E + F − C₃ of the cubical complex."""
    V: Set[Tuple[int, int, int]] = set()
    E: Set[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = set()
    F: Set[Tuple[Tuple[int, int, int], ...]] = set()
    for v in voxels_set:
        V.update(_cube_corners(v))
        E.update(_cube_edges(v))
        F.update(_cube_faces(v))
    C3 = len(voxels_set)
    return len(V) - len(E) + len(F) - C3


def count_enclosed_cavities(voxels_set: Set[Tuple[int, int, int]]) -> int:
    """β₂ — number of enclosed cavities (finite complement components).

    Flood-fill the complement inside a padded bounding box under 6-connectivity.
    Cavities are complement components that do not touch the padding boundary.
    """
    if not voxels_set:
        return 0
    min_x = min(v[0] for v in voxels_set) - 1
    max_x = max(v[0] for v in voxels_set) + 1
    min_y = min(v[1] for v in voxels_set) - 1
    max_y = max(v[1] for v in voxels_set) + 1
    min_z = min(v[2] for v in voxels_set) - 1
    max_z = max(v[2] for v in voxels_set) + 1

    # Flood-fill from all boundary points to identify the "outside" complement
    # component; anything the flood does not reach is an enclosed cavity.
    visited: Set[Tuple[int, int, int]] = set()
    stack: List[Tuple[int, int, int]] = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            for z in (min_z, max_z):
                p = (x, y, z)
                if p not in voxels_set and p not in visited:
                    stack.append(p)
                    visited.add(p)
            for z in range(min_z, max_z + 1):
                for y_edge in (min_y, max_y):
                    p = (x, y_edge, z)
                    if p not in voxels_set and p not in visited:
                        stack.append(p)
                        visited.add(p)
                for x_edge in (min_x, max_x):
                    p = (x_edge, y, z)
                    if p not in voxels_set and p not in visited:
                        stack.append(p)
                        visited.add(p)

    while stack:
        cx, cy, cz = stack.pop()
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            if not (min_x <= nx <= max_x and min_y <= ny <= max_y and min_z <= nz <= max_z):
                continue
            p = (nx, ny, nz)
            if p in voxels_set or p in visited:
                continue
            visited.add(p)
            stack.append(p)

    # Count remaining complement points and group them into components.
    unvisited: Set[Tuple[int, int, int]] = set()
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            for z in range(min_z, max_z + 1):
                p = (x, y, z)
                if p not in voxels_set and p not in visited:
                    unvisited.add(p)

    cavities = 0
    while unvisited:
        seed = next(iter(unvisited))
        comp_stack = [seed]
        unvisited.discard(seed)
        while comp_stack:
            cx, cy, cz = comp_stack.pop()
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                p = (cx + dx, cy + dy, cz + dz)
                if p in unvisited:
                    unvisited.discard(p)
                    comp_stack.append(p)
        cavities += 1
    return cavities


def cubical_betti_1(voxels_set: Set[Tuple[int, int, int]]) -> int:
    """First Betti number β₁ of the voxel cubical complex, via Euler characteristic.

    Computes β₁ = β₀ + β₂ − χ, where β₀ is the number of 6-connected components,
    β₂ is the number of enclosed cavities, and χ is the Euler characteristic
    of the cubical complex.

    This differs from ``shape_generation._calculate_circuit_rank`` (which
    returns the circuit rank of the voxel 6-adjacency graph) in that it
    accounts for 2-cells (voxel faces): a solid 2×2×2 cube has graph circuit
    rank 5 but cubical β₁ = 0, because the graph cycles are all bounded by
    faces of the 2-cube complex.

    Under 6-connectivity — the connectivity the MindFold 3D generator uses —
    the two invariants agree on every shape the generator can produce, since
    the generator never creates diagonal-only adjacencies.  This function is
    provided for validation and for the paper's methodological supplement.
    """
    if not voxels_set:
        return 0
    # Effective grid needs to accommodate the padded bounding box used by
    # cavity detection so _count_components can safely be called on it.
    max_x = max(v[0] for v in voxels_set) + 2
    max_y = max(v[1] for v in voxels_set) + 2
    max_z = max(v[2] for v in voxels_set) + 2
    eff_grid = (max_x, max_y, max_z)
    chi = euler_characteristic(voxels_set)
    b0 = _count_components(voxels_set, grid_size=eff_grid)
    b2 = count_enclosed_cavities(voxels_set)
    return max(0, b0 + b2 - chi)


__all__ = [
    "euler_characteristic",
    "count_enclosed_cavities",
    "cubical_betti_1",
]
