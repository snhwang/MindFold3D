"""
Unit tests for the two distinct cyclic-structure metrics:

  - circuit rank μ = |E| − |V| + C of the voxel 6-adjacency graph
    (shape_generation._calculate_circuit_rank) — the graph invariant the
    generator's fill/scoring loops control, recorded as `circuit_rank`;
  - cubical first Betti number β₁ (topology_extras.cubical_betti_1) — the
    volumetric hole count, recorded as `betti_1` and enforced at hole-tier
    validation.

The two invariants coincide on genuinely 1-voxel-wide thin shapes (verified
by the cross-validation tests below) but DIVERGE on real generator output in
both directions: 2×2 face patches give μ > β₁, and edge-adjacency tunnels
give β₁ > μ (see the documented-divergence tests, one of which uses an
actual production TreeSkeleton emission). They must never be conflated or
reported under one name.

Run with:  python -m pytest test_cycle_count.py -v
"""

from typing import Set, Tuple

from mindfold3d.shape_generation import _calculate_circuit_rank as _calculate_cycle_count
from mindfold3d.topology_extras import cubical_betti_1

Voxel = Tuple[int, int, int]
GRID = (16, 16, 16)


# ---------------------------------------------------------------------------
# Shape constructors (all shapes are 1-voxel-wide unless otherwise noted)
# ---------------------------------------------------------------------------

def _straight_chain(n: int) -> Set[Voxel]:
    """n voxels in a straight line — a tree, no cycles."""
    return {(i, 0, 0) for i in range(n)}


def _y_branch() -> Set[Voxel]:
    """A tree with one branch point — still no cycles."""
    return {(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0), (1, 2, 0)}


def _thin_ring_planar(side: int) -> Set[Voxel]:
    """A 1-voxel-wide square ring of outer side `side` in the z=0 plane.

    For side=3: 8 voxels enclosing a 1×1 hole. For side=4: 12 voxels
    enclosing a 2×2 hole. Both have β₁ = 1.
    """
    ring: Set[Voxel] = set()
    for i in range(side):
        ring.add((i, 0, 0))
        ring.add((i, side - 1, 0))
        ring.add((0, i, 0))
        ring.add((side - 1, i, 0))
    return ring


def _figure_eight_planar() -> Set[Voxel]:
    """Two 3×3 square rings sharing a single edge — β₁ = 2."""
    # Left ring: x in [0,2]
    left = _thin_ring_planar(3)
    # Right ring: shift so it shares the right edge of the left ring
    right = {(x + 2, y, 0) for (x, y, _) in left}
    return left | right


def _two_separate_rings() -> Set[Voxel]:
    """Two disjoint square rings — β₁ = 2 (one per component)."""
    r1 = _thin_ring_planar(3)
    r2 = {(x + 6, y, 0) for (x, y, _) in _thin_ring_planar(3)}
    return r1 | r2


# ---------------------------------------------------------------------------
# Tests: production metric returns the expected values
# ---------------------------------------------------------------------------

def test_empty_and_tiny():
    assert _calculate_cycle_count(set(), GRID) == 0
    assert _calculate_cycle_count({(0, 0, 0)}, GRID) == 0
    assert _calculate_cycle_count({(0, 0, 0), (1, 0, 0)}, GRID) == 0
    assert _calculate_cycle_count({(0, 0, 0), (1, 0, 0), (2, 0, 0)}, GRID) == 0


def test_straight_chain_no_cycles():
    for n in range(4, 12):
        assert _calculate_cycle_count(_straight_chain(n), GRID) == 0, f"chain n={n}"


def test_tree_with_branch_no_cycles():
    assert _calculate_cycle_count(_y_branch(), GRID) == 0


def test_thin_planar_ring_beta1_eq_1():
    for side in (3, 4, 5, 6):
        ring = _thin_ring_planar(side)
        assert _calculate_cycle_count(ring, GRID) == 1, f"ring side={side}"


def test_figure_eight_beta1_eq_2():
    fe = _figure_eight_planar()
    assert _calculate_cycle_count(fe, GRID) == 2


def test_two_separate_rings_beta1_eq_2():
    tr = _two_separate_rings()
    assert _calculate_cycle_count(tr, GRID) == 2


def test_ring_plus_tail_beta1_eq_1():
    """Attaching a chain (tail) to a ring does not change β₁."""
    ring = _thin_ring_planar(4)
    tail = {(3, i, 0) for i in range(4, 8)}
    assert _calculate_cycle_count(ring | tail, GRID) == 1


# ---------------------------------------------------------------------------
# Cross-validation: circuit rank agrees with full cubical β₁ on the shapes
# the generator can produce (1-voxel-wide skeletons).
# ---------------------------------------------------------------------------

def test_cross_validation_thin_shapes():
    shapes = {
        "chain-5": _straight_chain(5),
        "y-branch": _y_branch(),
        "ring-3": _thin_ring_planar(3),
        "ring-4": _thin_ring_planar(4),
        "ring-6": _thin_ring_planar(6),
        "figure-8": _figure_eight_planar(),
        "two-rings": _two_separate_rings(),
    }
    for name, voxels in shapes.items():
        circuit = _calculate_cycle_count(voxels, GRID)
        cubical = cubical_betti_1(voxels)
        assert circuit == cubical, (
            f"{name}: circuit_rank={circuit} != cubical_β₁={cubical} — "
            f"the two invariants should agree on 1-voxel-wide shapes"
        )


# ---------------------------------------------------------------------------
# Documented divergence cases. NOTE: divergence is NOT out-of-distribution —
# production shapes with medium/dense fill routinely contain 2×2 face patches
# (μ > β₁), and edge-adjacency tunnels giving β₁ > μ occur in generator
# output as well (see test_edge_adjacency_tunnel below). These tests pin
# down the expected behavior of both metrics so they are never conflated.
# ---------------------------------------------------------------------------

def test_solid_2x2x2_cube_documented_divergence():
    """Solid cubes are not producible by the generator, but if fed to
    _calculate_cycle_count they return the graph circuit rank (5), while
    full cubical β₁ is 0. This is documented behavior."""
    cube = {(i, j, k) for i in range(2) for j in range(2) for k in range(2)}
    assert _calculate_cycle_count(cube, GRID) == 5
    assert cubical_betti_1(cube) == 0


def test_hollow_shell_documented_divergence():
    """A 3×3×3 hollow shell has 1 cavity (β₂=1) and no through-holes (β₁=0),
    but its 6-adjacency graph has many cycles (circuit rank = 23)."""
    shell: Set[Voxel] = set()
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i in (0, 2) or j in (0, 2) or k in (0, 2):
                    shell.add((i, j, k))
    assert _calculate_cycle_count(shell, GRID) == 23
    assert cubical_betti_1(shell) == 0


def test_edge_adjacency_tunnel_beta1_exceeds_mu():
    """β₁ > μ is possible: edge-adjacent voxels can enclose a tunnel that the
    face-adjacency graph cannot see. This exact voxel set was emitted by the
    production TreeSkeleton pipeline (archetype=tree, 15 voxels, 7³ grid) and
    verified independently by GF(2) boundary-matrix homology: the adjacency
    graph is a tree (μ = 0) but the cubical complex has one tunnel (β₁ = 1).
    Refutes the claim that μ ≥ β₁ always holds on generator output."""
    shape: Set[Voxel] = {
        (0, 3, 2), (0, 3, 3), (1, 3, 3), (2, 3, 3), (3, 2, 3),
        (3, 3, 1), (3, 3, 2), (3, 3, 3), (3, 4, 1), (3, 4, 3),
        (3, 4, 4), (3, 5, 2), (3, 5, 3), (4, 3, 3), (4, 5, 2),
    }
    assert _calculate_cycle_count(shape, GRID) == 0
    assert cubical_betti_1(shape) == 1


# ---------------------------------------------------------------------------
# Optional gudhi cross-check
# ---------------------------------------------------------------------------

def test_gudhi_cross_validation_if_available():
    """If gudhi is installed, verify cubical_betti_1 agrees with it on thin
    shapes.  gudhi is NOT a runtime dependency."""
    try:
        import gudhi  # type: ignore
    except ImportError:
        return  # skip silently

    def gudhi_b1(voxels: Set[Voxel]) -> int:
        if not voxels:
            return 0
        min_x = min(v[0] for v in voxels)
        min_y = min(v[1] for v in voxels)
        min_z = min(v[2] for v in voxels)
        max_x = max(v[0] for v in voxels) + 1
        max_y = max(v[1] for v in voxels) + 1
        max_z = max(v[2] for v in voxels) + 1
        dx = max_x - min_x
        dy = max_y - min_y
        dz = max_z - min_z
        # Build a top-dimensional cubical complex: 0 inside, 1 outside
        top = [
            [
                [
                    0.0 if (x + min_x, y + min_y, z + min_z) in voxels else 1.0
                    for z in range(dz)
                ]
                for y in range(dy)
            ]
            for x in range(dx)
        ]
        cc = gudhi.CubicalComplex(top_dimensional_cells=top)
        cc.compute_persistence()
        # Count features born at 0
        pers = cc.persistence()
        b1 = sum(1 for dim, (birth, death) in pers if dim == 1 and birth < 0.5)
        return b1

    shapes = [_thin_ring_planar(3), _thin_ring_planar(4), _figure_eight_planar()]
    for s in shapes:
        assert cubical_betti_1(s) == gudhi_b1(s)


if __name__ == "__main__":
    # Manual runner (avoids requiring pytest)
    import inspect
    ns = dict(globals())
    passed = failed = 0
    for name, fn in ns.items():
        if name.startswith("test_") and callable(fn) and inspect.getfullargspec(fn).args == []:
            try:
                fn()
                print(f"PASS  {name}")
                passed += 1
            except AssertionError as e:
                print(f"FAIL  {name}: {e}")
                failed += 1
            except Exception as e:
                print(f"ERROR {name}: {type(e).__name__}: {e}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    exit(1 if failed else 0)
