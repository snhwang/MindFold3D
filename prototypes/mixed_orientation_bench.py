"""Feasibility benchmark for mixed-orientation hole templates.

Mirrors the Study 2 protocol: per-shape deterministic seeding, up to 20
attempts per shape, cubical-beta1 + connectivity validation. Reports
first-pass acceptance, mean attempts, how often placement succeeded
(vs fallback to the coplanar chain), plane diversity among mixed shapes,
and emitted voxel counts.
"""
import sys
import random
from collections import Counter

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from mindfold3d.cognitive_mapping import SkeletonSpec
from mindfold3d.skeleton_generation import HoleMarkedLoopSkeleton
from mindfold3d.topology_extras import cubical_betti_1

GRID = (11, 11, 11)
N = 100
BASE_SEED = 2026
CELLS = [(2, 17), (2, 20), (2, 25), (2, 30), (3, 25), (3, 30), (4, 30), (4, 35)]

print(f"{'b1':>3} {'vox':>4} {'first':>6} {'att':>5} {'mixed%':>7} "
      f"{'multi-plane%':>13} {'vox mean':>9} {'b1 exact':>9}")

for b1, vox in CELLS:
    spec = SkeletonSpec(
        archetype="loop",
        voxel_count=vox,
        grid_size=GRID,
        num_branches=4,
        num_loops=b1,
        target_b1=b1,
        skip_optimizer=True,
        mixed_orientation=True,
    )
    first_pass = 0
    attempts_total = 0
    mixed_count = 0
    multi_plane = 0
    vox_sum = 0
    b1_exact = 0
    failures = 0
    for i in range(N):
        success = False
        for attempt in range(1, 21):
            rng = random.Random(f"mixedbench|{BASE_SEED}|{b1}|{vox}|{i}|{attempt}")
            sk = HoleMarkedLoopSkeleton(GRID, rng=rng)
            sk.generate(spec)
            if sk._validate(spec):
                success = True
                break
        if not success:
            failures += 1
            continue
        attempts_total += attempt
        if attempt == 1:
            first_pass += 1
        if getattr(sk, "template_mode", "?") == "mixed":
            mixed_count += 1
            if len(set(getattr(sk, "ring_planes", []))) >= 2:
                multi_plane += 1
        vox_sum += len(sk.voxels)
        if cubical_betti_1(sk.voxels) == b1:
            b1_exact += 1
    n_ok = N - failures
    print(f"{b1:>3} {vox:>4} {100*first_pass/max(1,n_ok):>5.1f}% "
          f"{attempts_total/max(1,n_ok):>5.2f} {100*mixed_count/max(1,n_ok):>6.1f}% "
          f"{100*multi_plane/max(1,mixed_count) if mixed_count else 0:>12.1f}% "
          f"{vox_sum/max(1,n_ok):>9.2f} {b1_exact:>4}/{n_ok:<4}"
          + (f"  FAILURES={failures}" if failures else ""))
