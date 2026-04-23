"""
Shape Visualization for Paper Figures — MindFold 3D

Copyright (c) 2024-2026 Scott N. Hwang, Parviz Safadel. All rights reserved.

Generates a 3x4 grid of shapes: rows = cognitive dimensions,
columns = difficulty tiers (Low → Expert). Each shape varies one dimension
while holding the others at medium.

Usage:
    python visualize_shapes.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from cognitive_mapping import get_skeleton_spec
from skeleton_generation import generate_shape_skeleton

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
        "features": ["branching_factor", "number_of_components", "cycle_count"],
        "short": ["Br", "Comp", "Cyc"],
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


def generate_shape(vary_dim: str, tier: str) -> dict:
    """Generate a shape varying one dimension, holding others at medium."""
    difficulties = {d: "medium" for d in ALL_DIMS}
    difficulties[vary_dim] = tier

    spec = get_skeleton_spec(
        shape_difficulties=difficulties,
        task_difficulties={},
    )
    return generate_shape_skeleton(spec)


def voxels_to_array(voxels: list, grid_size: list) -> np.ndarray:
    """Convert list of [x,y,z] voxels to a boolean 3D array."""
    arr = np.zeros(grid_size, dtype=bool)
    for v in voxels:
        arr[v[0], v[1], v[2]] = True
    return arr


def main():
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "MindFold 3D — Shape Difficulty Progression",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    for row, dim in enumerate(ALL_DIMS):
        dim_info = DIM_LABELS[dim]
        color = ROW_COLORS[dim]

        for col, tier in enumerate(TIERS):
            idx = row * 4 + col + 1
            print(f"  Generating {dim}={tier}...", end=" ", flush=True)

            result = generate_shape(dim, tier)
            voxels = result["voxels"]
            grid_size = result["grid_size"]
            features = result["features"]

            arr = voxels_to_array(voxels, grid_size)

            # Color array
            colors = np.empty(arr.shape, dtype=object)
            colors[arr] = color

            ax = fig.add_subplot(3, 4, idx, projection="3d")
            ax.voxels(arr, facecolors=colors, edgecolor="#444444", linewidth=0.3)

            # Consistent viewing angle
            ax.view_init(elev=25, azim=135)

            # Equal aspect — use largest grid dimension across all shapes
            max_dim = max(grid_size)
            ax.set_xlim(0, max_dim)
            ax.set_ylim(0, max_dim)
            ax.set_zlim(0, max_dim)

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
                fontsize=10,
                pad=8,
            )

            print("ok")

        # Row label on the left — use text on the first subplot of each row
        ax_first = fig.axes[row * 4]
        ax_first.text2D(
            -0.15, 0.5,
            dim_info["title"],
            transform=ax_first.transAxes,
            fontsize=13,
            fontweight="bold",
            va="center",
            ha="center",
            rotation=90,
            color=color,
        )

    plt.subplots_adjust(
        left=0.08, right=0.97,
        top=0.92, bottom=0.02,
        wspace=0.05, hspace=0.15,
    )

    print("\nDisplaying figure. Close window when done.")
    plt.show()


if __name__ == "__main__":
    main()
