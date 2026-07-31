"""Investigate why circuit rank diverges from cubical β₁ on 'thin' shapes."""
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from cognitive_mapping import SkeletonSpec
from skeleton_generation import TreeSkeleton
from shape_generation import _calculate_cycle_count, _count_components
from topology_extras import euler_characteristic, count_enclosed_cavities, cubical_betti_1


def analyze(voxels, label):
    print(f"\n=== {label} ===")
    voxel_set = set(voxels)
    V = len(voxel_set)
    # Edges: face-adjacent pairs
    edges = 0
    for v in voxel_set:
        x, y, z = v
        for dx, dy, dz in ((1,0,0), (0,1,0), (0,0,1)):
            if (x+dx, y+dy, z+dz) in voxel_set:
                edges += 1
    E = edges
    C = _count_components(voxel_set, grid_size=(30, 30, 30))
    circuit = _calculate_cycle_count(voxel_set, (30, 30, 30))
    chi = euler_characteristic(voxel_set)
    b0 = _count_components(voxel_set, grid_size=(30, 30, 30))
    b2 = count_enclosed_cavities(voxel_set)
    cub_b1 = cubical_betti_1(voxel_set)
    print(f"  V={V}, E={E}, C={C}")
    print(f"  circuit_rank (E-V+C) = {circuit}")
    print(f"  cubical: χ={chi}, β₀={b0}, β₂={b2}, β₁={cub_b1}")
    print(f"  voxels: {sorted(voxel_set)}")

    # Check "thin" more carefully: does any voxel have a 2×2 face neighborhood in the same plane?
    # For circuit rank to equal shape β₁, we need no small graph cycles bounded by voxel faces.
    # A voxel v has such a small cycle if there's a 2x2 face of voxels (4 voxels in a square in some coordinate plane).
    face_2x2_count = 0
    for v in voxel_set:
        x, y, z = v
        # Check each of 12 possible 2x2 squares containing v (v is one corner)
        for axis in range(3):  # which axis is the "normal" of the 2x2 square
            other_axes = [i for i in range(3) if i != axis]
            for sa in (-1, 0, 1):
                if sa == 0: continue
                for sb in (-1, 0, 1):
                    if sb == 0: continue
                    if sa != -1 and sb != -1: continue  # avoid duplicate
                    a, b = other_axes
                    p1 = list(v)
                    p2 = list(v); p2[a] += sa
                    p3 = list(v); p3[b] += sb
                    p4 = list(v); p4[a] += sa; p4[b] += sb
                    if all(tuple(p) in voxel_set for p in (p1, p2, p3, p4)):
                        face_2x2_count += 1
    print(f"  2×2 face patches (potential small graph cycles): {face_2x2_count // 4}")


# Case 1: manually construct a tree/medium-like shape that generator produces
# Let me first generate one to see its structure
random.seed(0)
gen = TreeSkeleton((7, 7, 7))
spec = SkeletonSpec(
    archetype="tree", voxel_count=12, grid_size=(7,7,7),
    num_branches=3, num_loops=1, num_components=1,
    direction_spread="moderate", planarity="medium", packing="medium",
)
voxels = gen.generate(spec)
analyze(voxels, "tree/medium seed=0")

# Simpler diagnostic: what's the smallest 2x2 face patch that makes circuit rank > shape β₁?
print("\n\n=== SIMPLE DIAGNOSTIC ===")
# Four voxels forming a 2x2 face patch in the xy plane
square = {(0,0,0), (1,0,0), (0,1,0), (1,1,0)}
analyze(square, "2×2 face patch (4 voxels in a square, planar)")

# A 3-voxel L-shape
L = {(0,0,0), (1,0,0), (1,1,0)}
analyze(L, "L-shape (3 voxels)")

# A thin ring (side=3) - the classic case
ring = set()
for i in range(3):
    ring.add((i, 0, 0))
    ring.add((i, 2, 0))
    ring.add((0, i, 0))
    ring.add((2, i, 0))
analyze(ring, "Thin ring side=3 (8 voxels)")
