"""
Shape Visualization for Paper Figures — MindFold 3D

Copyright (c) 2026 Scott N. Hwang, Parviz Safadel. All rights reserved.

Generates a 3x4 grid of shapes: rows = cognitive dimensions,
columns = difficulty tiers (Low → Expert). Each shape varies one dimension
while holding the others at medium.

Usage:
    python visualize_shapes.py              # interactive display
    python visualize_shapes.py --save       # save to docs/latex/figures/Figure_2_Shapes.png at 600 dpi
    python visualize_shapes.py --save -o path/to/out.png --dpi 600
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mindfold3d.cognitive_mapping import get_skeleton_spec
from mindfold3d.skeleton_generation import generate_shape_skeleton

TIERS = ["low", "medium", "high", "expert"]
ALL_DIMS = ["spatial_form", "structural_complexity", "spatial_scale"]

# Row labels and feature annotations
DIM_LABELS = {
    "spatial_form": {
        "title": "Spatial\nForm",
        "features": ["anisotropy_index", "symmetry_score"],
        "short": ["AI", "Sym"],
    },
    "structural_complexity": {
        "title": "Structural\nComplexity",
        "features": ["branching_factor", "number_of_components", "betti_1"],
        "short": ["Br", "Comp", r"$\beta_1$"],
    },
    "spatial_scale": {
        "title": "Spatial\nScale",
        "features": ["voxel_count"],
        "short": ["N"],
    },
}

# Colors for each dimension row
ROW_COLORS = {
    "spatial_form": "#5C6BC0",          # indigo
    "structural_complexity": "#26A69A",  # teal
    "spatial_scale": "#EF5350",          # red
}


# Neutral levels for the non-varied dimensions (matches the Study 1
# protocol: Spatial Form and Scale at Medium, Structural Complexity at
# Low — a mandatory ring would geometrically constrain the form features).
NEUTRAL = {
    "spatial_form": "medium",
    "structural_complexity": "low",
    "spatial_scale": "medium",
}


def generate_shape(vary_dim: str, tier: str) -> dict:
    """Generate a shape varying one dimension, holding others at neutral."""
    difficulties = dict(NEUTRAL)
    difficulties[vary_dim] = tier

    spec = get_skeleton_spec(
        shape_difficulties=difficulties,
        task_difficulties={},
    )
    return generate_shape_skeleton(spec)


# (anisotropy range, symmetry range) of each Spatial Form tier, used to
# pick a representative exemplar rather than an arbitrary first sample.
SF_RANGES = {
    "low":    ((0.00, 0.40), (0.70, 1.00)),
    "medium": ((0.45, 0.65), (0.46, 0.70)),
    "high":   ((0.65, 0.82), (0.10, 0.46)),
    "expert": ((0.85, 0.99), (0.00, 0.20)),
}


def _bbox_longest(result: dict) -> int:
    coords = np.array(result["voxels"])
    return int((coords.max(0) - coords.min(0)).max()) + 1


def _sf_mid_score(f: dict, tier: str) -> float:
    (ai_lo, ai_hi), (sy_lo, sy_hi) = SF_RANGES[tier]
    return (abs(f["anisotropy_index"] - (ai_lo + ai_hi) / 2) / (ai_hi - ai_lo)
            + abs(f["symmetry_score"] - (sy_lo + sy_hi) / 2) / (sy_hi - sy_lo))


def pick_shape(vary_dim: str, tier: str, k: int = 15) -> dict:
    """Sample k candidates and return the most tier-representative one.

    Spatial Form exemplars must land inside both tier ranges, then the most
    compact in-range shape is preferred at Low and the most elongated at
    High/Expert, so the visual progression matches the feature progression
    (a planar shape can score high anisotropy without looking elongated).
    Cyclic Structural Complexity exemplars are chosen so every tunnel is
    visible in the display projection, preferring shapes that are thin
    along the viewing-depth axis. Other cells have exact integer features,
    so the first sample is representative.
    """
    if vary_dim == "spatial_form":
        cands = [generate_shape(vary_dim, tier) for _ in range(k)]
        (ai_lo, ai_hi), (sy_lo, sy_hi) = SF_RANGES[tier]
        in_range = [
            c for c in cands
            if ai_lo <= c["features"]["anisotropy_index"] <= ai_hi
            and sy_lo <= c["features"]["symmetry_score"] <= sy_hi
        ]
        if in_range:
            if tier == "low":
                return min(in_range, key=_bbox_longest)
            if tier in ("high", "expert"):
                return max(in_range, key=_bbox_longest)
            return min(in_range, key=lambda c: _sf_mid_score(c["features"], tier))
        return min(cands, key=lambda c: _sf_mid_score(c["features"], tier))
    if vary_dim == "structural_complexity" and tier != "low":
        best, best_key = None, None
        for _ in range(k):
            result = generate_shape(vary_dim, tier)
            arr = orient_for_display(
                voxels_to_array(result["voxels"], result["grid_size"]))
            target = result["features"]["betti_1"]
            visible = _enclosed_empty_count(arr.any(axis=1))
            depth = int(np.ptp(np.argwhere(arr)[:, 1])) + 1
            key = (0 if visible == target else 1, depth)
            if best_key is None or key < best_key:
                best, best_key = result, key
                best["_display_arr"] = arr
        return best
    return generate_shape(vary_dim, tier)


def voxels_to_array(voxels: list, grid_size: list) -> np.ndarray:
    """Convert list of [x,y,z] voxels to a boolean 3D array."""
    arr = np.zeros(grid_size, dtype=bool)
    for v in voxels:
        arr[v[0], v[1], v[2]] = True
    return arr


def _enclosed_empty_count(mask2d: np.ndarray) -> int:
    """Number of enclosed empty regions (through-holes) in a 2D mask."""
    from collections import deque

    h, w = mask2d.shape
    empty = ~mask2d
    seen = np.zeros_like(empty)
    dq = deque()
    for i in range(h):
        for j in (0, w - 1):
            if empty[i, j] and not seen[i, j]:
                seen[i, j] = True
                dq.append((i, j))
    for j in range(w):
        for i in (0, h - 1):
            if empty[i, j] and not seen[i, j]:
                seen[i, j] = True
                dq.append((i, j))
    while dq:
        i, j = dq.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and empty[ni, nj] and not seen[ni, nj]:
                seen[ni, nj] = True
                dq.append((ni, nj))
    remaining = empty & ~seen
    count = 0
    for i in range(h):
        for j in range(w):
            if remaining[i, j]:
                count += 1
                dq.append((i, j))
                remaining[i, j] = False
                while dq:
                    ci, cj = dq.popleft()
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < h and 0 <= nj < w and remaining[ni, nj]:
                            remaining[ni, nj] = False
                            dq.append((ni, nj))
    return count


def orient_for_display(arr: np.ndarray) -> np.ndarray:
    """Rotate (90-degree steps) so through-tunnels run horizontally.

    Template plane orientation is random, and a vertical tunnel is nearly
    edge-on at the figure's low camera elevation. All annotated features
    are invariant under these rotations.
    """
    axes = [a for a in range(3) if _enclosed_empty_count(arr.any(axis=a)) > 0]
    if axes:
        if 1 in axes:
            return arr
        if 0 in axes:
            return np.rot90(arr, 1, axes=(0, 1))
        return np.rot90(arr, 1, axes=(1, 2))
    # Acyclic: put the longest extent along x, perpendicular to the view,
    # so elongation is not foreshortened.
    coords = np.argwhere(arr)
    longest = int(np.argmax(coords.max(0) - coords.min(0)))
    if longest == 1:
        return np.rot90(arr, 1, axes=(1, 0))
    if longest == 2:
        return np.rot90(arr, 1, axes=(2, 0))
    return arr


def embed_centered(arr: np.ndarray, size: int) -> np.ndarray:
    """Place the shape centered in a cubic grid of the given size."""
    out = np.zeros((size, size, size), dtype=bool)
    coords = np.argwhere(arr)
    mins, maxs = coords.min(0), coords.max(0)
    shift = (np.array([size] * 3) - (maxs - mins + 1)) // 2 - mins
    for c in coords:
        out[tuple(c + shift)] = True
    return out


DEFAULT_OUTPUT = Path("docs/latex/figures/Figure_2_Shapes.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="Save the figure instead of displaying it interactively.")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output path (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--dpi", type=int, default=600, help="Output DPI (default: 600).")
    args = parser.parse_args()

    results = {}
    for dim in ALL_DIMS:
        for tier in TIERS:
            print(f"  Generating {dim}={tier}...", end=" ", flush=True)
            results[(dim, tier)] = pick_shape(dim, tier)
            print("ok")

    # Common frame so one voxel has the same on-page size in every cell.
    frame = max(max(r["grid_size"]) for r in results.values())

    fig = plt.figure(figsize=(18, 14))

    for row, dim in enumerate(ALL_DIMS):
        dim_info = DIM_LABELS[dim]
        color = ROW_COLORS[dim]

        for col, tier in enumerate(TIERS):
            idx = row * 4 + col + 1
            result = results[(dim, tier)]
            features = result["features"]

            arr = result.get("_display_arr")
            if arr is None:
                arr = orient_for_display(
                    voxels_to_array(result["voxels"], result["grid_size"]))
            arr = embed_centered(arr, frame)

            # Color array
            colors = np.empty(arr.shape, dtype=object)
            colors[arr] = color

            ax = fig.add_subplot(3, 4, idx, projection="3d")
            ax.voxels(arr, facecolors=colors, edgecolor="#444444", linewidth=0.3)

            # Consistent viewing angle, swung toward the y axis so the
            # horizontally oriented tunnels of cyclic shapes read as
            # see-through holes rather than shallow recesses.
            ax.view_init(elev=20, azim=110)

            # Equal aspect at the common voxel scale
            ax.set_xlim(0, frame)
            ax.set_ylim(0, frame)
            ax.set_zlim(0, frame)

            # Clean axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")

            # Feature annotation
            feat_strs = []
            for feat, short in zip(dim_info["features"], dim_info["short"]):
                val = features.get(feat)
                if val is not None:
                    if isinstance(val, float):
                        feat_strs.append(f"{short}={val:.2f}")
                    else:
                        feat_strs.append(f"{short}={val}")
            annotation = ", ".join(feat_strs)

            ax.set_title(
                f"{tier.capitalize()}\n{annotation}",
                fontsize=16,
                pad=8,
            )

        # Row label on the left — use text on the first subplot of each row
        ax_first = fig.axes[row * 4]
        ax_first.text2D(
            -0.15, 0.5,
            dim_info["title"],
            transform=ax_first.transAxes,
            fontsize=20,
            fontweight="bold",
            va="center",
            ha="center",
            rotation=90,
            color=color,
        )

    plt.subplots_adjust(
        left=0.08, right=0.97,
        top=0.98, bottom=0.02,
        wspace=0.05, hspace=0.15,
    )

    if args.save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"\nSaved figure to {args.output} at {args.dpi} dpi.")
    else:
        print("\nDisplaying figure. Close window when done.")
        plt.show()


if __name__ == "__main__":
    main()
