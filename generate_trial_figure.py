"""
Generate a figure showing a sample assessment trial for the paper.

Displays a target shape and 4 choice shapes (1 correct match + 3 distractors)
arranged as a participant would see them.

Copyright (c) 2024-2026 Scott N. Hwang, Parviz Safadel. All rights reserved.

Usage:
    python generate_trial_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

from cognitive_mapping import get_skeleton_spec, perturb_skeleton_spec
from skeleton_generation import generate_shape_skeleton
from shape_generation import generate_mirror_reflection

random.seed(42)
np.random.seed(42)

SHAPE_COLOR = "#5C6BC0"
MATCH_COLOR = "#5C6BC0"
DISTRACTOR_COLORS = ["#EF5350", "#26A69A", "#FF9800"]
EDGE_COLOR = "#444444"


def voxels_to_array(voxels, grid_size):
    arr = np.zeros(grid_size, dtype=bool)
    for v in voxels:
        arr[v[0], v[1], v[2]] = True
    return arr


def rotate_voxels(voxels, grid_size, angle_deg=None):
    """Rotate voxels by a random 90-degree multiple around a random axis.

    Uses direct axis permutations/negations — no scipy needed.
    """
    if angle_deg is None:
        angle_deg = random.choice([90, 180, 270])
    axis = random.choice(["x", "y", "z"])
    steps = angle_deg // 90  # 1, 2, or 3 quarter-turns

    coords = np.array(voxels, dtype=float)

    for _ in range(steps):
        # Each 90-degree rotation: permute two axes and negate one
        if axis == "x":
            coords = coords[:, [0, 2, 1]]
            coords[:, 1] = -coords[:, 1]
        elif axis == "y":
            coords = coords[:, [2, 1, 0]]
            coords[:, 2] = -coords[:, 2]
        else:  # z
            coords = coords[:, [1, 0, 2]]
            coords[:, 0] = -coords[:, 0]

    # Shift so all coordinates are non-negative
    coords = coords - coords.min(axis=0)
    coords = np.round(coords).astype(int)

    max_coords = coords.max(axis=0)
    new_grid = tuple(int(m + 1) for m in max_coords)
    return [list(v) for v in coords], new_grid


def plot_voxels(ax, voxels, grid_size, color, elev=25, azim=135):
    arr = voxels_to_array(voxels, grid_size)
    colors = np.empty(arr.shape, dtype=object)
    colors[arr] = color
    ax.voxels(arr, facecolors=colors, edgecolor=EDGE_COLOR, linewidth=0.3)
    ax.view_init(elev=elev, azim=azim)
    max_dim = max(grid_size)
    ax.set_xlim(0, max_dim)
    ax.set_ylim(0, max_dim)
    ax.set_zlim(0, max_dim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("w")
    ax.yaxis.pane.set_edgecolor("w")
    ax.zaxis.pane.set_edgecolor("w")


def main():
    # Generate a medium-difficulty chiral target shape
    print("Generating target shape...")
    difficulties = {
        "spatial_form": "medium",
        "structural_complexity": "medium",
        "spatial_scale": "medium",
    }
    spec = get_skeleton_spec(
        shape_difficulties=difficulties,
        task_difficulties={"mirror_discrimination": "high"},
    )
    target = generate_shape_skeleton(spec)
    target_voxels = target["voxels"]
    target_grid = target["grid_size"]

    # Generate correct match (rotated copy)
    print("Generating rotated match...")
    match_voxels, match_grid = rotate_voxels(target_voxels, target_grid)

    # Generate distractors
    choices = []

    # Distractor 1: mirror reflection
    print("Generating mirror distractor...")
    mirror_result = generate_mirror_reflection(
        target_voxels, tuple(target_grid)
    )
    if mirror_result:
        choices.append(("Mirror", [list(v) for v in mirror_result], target_grid))
    else:
        # Fallback: tier 2 subtle distractor
        d_spec = perturb_skeleton_spec(spec, 2)
        d_result = generate_shape_skeleton(d_spec)
        choices.append(("Subtle", d_result["voxels"], d_result["grid_size"]))

    # Distractor 2: moderate (tier 1)
    print("Generating moderate distractor...")
    d_spec = perturb_skeleton_spec(spec, 1)
    d_result = generate_shape_skeleton(d_spec)
    choices.append(("Moderate", d_result["voxels"], d_result["grid_size"]))

    # Distractor 3: subtle (tier 2)
    print("Generating subtle distractor...")
    d_spec = perturb_skeleton_spec(spec, 2)
    d_result = generate_shape_skeleton(d_spec)
    choices.append(("Subtle", d_result["voxels"], d_result["grid_size"]))

    # Insert the correct match at a random position among the 4 choices
    correct_idx = random.randint(0, 3)
    all_choices = list(choices)
    all_choices.insert(correct_idx, ("Match", match_voxels, match_grid))

    # Create figure: target on top, 4 choices below
    fig = plt.figure(figsize=(16, 9))

    # Target shape — centered at top
    ax_target = fig.add_subplot(2, 5, 3, projection="3d")
    plot_voxels(ax_target, target_voxels, target_grid, SHAPE_COLOR)
    ax_target.set_title("Target Shape", fontsize=14, fontweight="bold", pad=12)

    # Choice labels
    choice_labels = ["A", "B", "C", "D"]
    choice_colors = []
    for i in range(4):
        if i == correct_idx:
            choice_colors.append(MATCH_COLOR)
        else:
            ci = len(choice_colors) - (1 if i > correct_idx else 0)
            choice_colors.append(DISTRACTOR_COLORS[ci % len(DISTRACTOR_COLORS)])

    # 4 choices in the bottom row
    for i, (label, voxels, grid) in enumerate(all_choices):
        # Rotate distractors too for realism
        if label != "Match":
            voxels, grid = rotate_voxels(voxels, grid)

        ax = fig.add_subplot(2, 4, 5 + i, projection="3d")
        plot_voxels(ax, voxels, grid, choice_colors[i])

        marker = " *" if label == "Match" else ""
        tier_label = f" ({label})" if label != "Match" else " (Correct)"
        ax.set_title(
            f"Choice {choice_labels[i]}{tier_label}",
            fontsize=12,
            fontweight="bold" if label == "Match" else "normal",
            pad=10,
        )

    plt.suptitle(
        "MindFold 3D \u2014 Sample Assessment Trial",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.subplots_adjust(
        left=0.02, right=0.98,
        top=0.90, bottom=0.02,
        wspace=0.05, hspace=0.15,
    )

    output_path = "docs/figures/Figure_3_Sample_Trial.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
