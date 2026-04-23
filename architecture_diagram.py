"""
Architecture Diagram for Paper — MindFold 3D

Copyright (c) 2024-2026 Scott N. Hwang, Parviz Safadel. All rights reserved.

Generates a publication-quality architecture diagram showing the three-layer
cognitive framework and the bidirectional mapping pipeline.

Usage:
    python architecture_diagram.py
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Colors
C_LAYER1 = "#5C6BC0"   # indigo
C_LAYER2 = "#26A69A"   # teal
C_LAYER3 = "#EF5350"   # red
C_GEN = "#FF8F00"      # amber
C_MAP = "#7E57C2"      # purple
C_LLM = "#66BB6A"      # green
C_ARROW = "#37474F"
C_TEXT = "#212121"

# Layout constants
BOX_FS = 11             # uniform font size in all boxes
BOX_H = 0.028           # single-line box height
BOX_H2 = 0.040          # two-line box height
BOX_PAD = "round,pad=0.003"  # minimal internal padding


def rbox(ax, x, y, w, h, text, color, text_color="white", alpha=0.9):
    """Draw a rounded rectangle with centered text."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=BOX_PAD,
        facecolor=color, edgecolor="none", alpha=alpha, zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=BOX_FS, color=text_color, zorder=3)


def arr(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.5, cs="arc3,rad=0"):
    """Draw an arrow."""
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", color=color,
        lw=lw, connectionstyle=cs, zorder=4, mutation_scale=14,
    ))


def sbg(ax, x, y, w, h, label, color):
    """Draw a section background with label."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.005",
        facecolor=color, edgecolor=color, alpha=0.12, lw=1.5, zorder=0,
    ))
    ax.text(x + 0.015, y + h - 0.008, label,
            fontsize=BOX_FS, fontweight="bold", color=color,
            va="top", ha="left", zorder=1)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    ax.text(0.80, 1.08, "MindFold 3D — System Architecture",
            ha="center", va="top", fontsize=16, fontweight="bold", color=C_TEXT)

    # Bidirectional mapping label
    ax.annotate("", xy=(0.42, 1.02), xytext=(0.84, 1.02),
                arrowprops=dict(arrowstyle="<->", color=C_MAP, lw=2.5))
    ax.text(0.63, 1.04, "Bidirectional Cognitive-Geometric Mapping",
            ha="center", fontsize=12, fontweight="bold", color=C_MAP)

    # =========================================================
    # LEFT: Cognitive Specification
    # =========================================================
    sbg(ax, 0.02, 0.56, 0.36, 0.43, "Cognitive Specification", C_MAP)

    rbox(ax, 0.04, 0.93, 0.32, BOX_H, "Difficulty Settings", C_MAP)

    # Layer 1
    rbox(ax, 0.04, 0.87, 0.15, BOX_H, "Spatial Form", C_LAYER1)
    rbox(ax, 0.21, 0.87, 0.15, BOX_H, "Struct. Complexity", C_LAYER1)
    rbox(ax, 0.04, 0.83, 0.15, BOX_H, "Spatial Scale", C_LAYER1)
    ax.text(0.20, 0.905, "Layer 1: Shape Geometry", fontsize=BOX_FS,
            color=C_LAYER1, ha="center", zorder=5)

    # Layer 2
    rbox(ax, 0.04, 0.76, 0.15, BOX_H, "Mental Rotation", C_LAYER2)
    rbox(ax, 0.21, 0.76, 0.15, BOX_H, "Mirror Discrim.", C_LAYER2)
    rbox(ax, 0.04, 0.72, 0.15, BOX_H, "Working Memory", C_LAYER2)
    rbox(ax, 0.21, 0.72, 0.15, BOX_H, "Config. Binding", C_LAYER2)
    rbox(ax, 0.04, 0.68, 0.15, BOX_H, "Perspective Taking", C_LAYER2)
    ax.text(0.20, 0.80, "Layer 2: Task Design", fontsize=BOX_FS,
            color=C_LAYER2, ha="center", zorder=5)

    # =========================================================
    # CENTER: Generation Pipeline
    # =========================================================
    sbg(ax, 0.42, 0.56, 0.38, 0.43, "Generation Pipeline", C_GEN)

    rbox(ax, 0.44, 0.93, 0.34, BOX_H,
         "Forward Mapping (cognitive → skeleton)", C_MAP)

    rbox(ax, 0.44, 0.87, 0.10, BOX_H, "Tree", C_GEN)
    rbox(ax, 0.56, 0.87, 0.10, BOX_H, "Chiral", C_GEN)
    rbox(ax, 0.68, 0.87, 0.10, BOX_H, "Bridge", C_GEN)

    rbox(ax, 0.44, 0.82, 0.34, BOX_H,
         "Generated 3D Voxel Shape", C_GEN)

    rbox(ax, 0.44, 0.76, 0.34, BOX_H2,
         "Feature Measurement\n(ground-truth vector)", "#455A64")

    rbox(ax, 0.44, 0.69, 0.34, BOX_H2,
         "Reverse Mapping\n(features → cognitive tiers)", C_MAP)

    # Arrows in pipeline
    arr(ax, 0.61, 0.93, 0.61, 0.905)
    arr(ax, 0.61, 0.87, 0.61, 0.855)
    arr(ax, 0.61, 0.82, 0.61, 0.805)
    arr(ax, 0.61, 0.76, 0.61, 0.735)

    # Spec → forward mapping
    arr(ax, 0.38, 0.94, 0.44, 0.94)

    # =========================================================
    # RIGHT: Cognitive Profile & Scoring
    # =========================================================
    sbg(ax, 0.84, 0.56, 0.38, 0.43, "Cognitive Profile & Scoring", C_MAP)

    rbox(ax, 0.86, 0.93, 0.34, BOX_H, "Classified Difficulty Profile", C_MAP)
    rbox(ax, 0.86, 0.87, 0.34, BOX_H, "Fidelity Score (intended vs. realized)", "#7E57C2")
    rbox(ax, 0.86, 0.82, 0.34, BOX_H, "Ground-Truth Feature Vector", "#455A64")
    rbox(ax, 0.86, 0.77, 0.34, BOX_H, "Scoring uses measured features", "#455A64")

    # Reverse map → profile
    arr(ax, 0.78, 0.71, 0.86, 0.944, color=C_MAP, lw=1.5)
    # Profile → fidelity
    arr(ax, 1.03, 0.93, 1.03, 0.905)
    # Measurement → ground truth
    arr(ax, 0.78, 0.78, 0.86, 0.834, color="#455A64", lw=1.5, cs="arc3,rad=-0.1")
    # Ground truth → scoring
    arr(ax, 1.03, 0.82, 1.03, 0.805)

    # =========================================================
    # BOTTOM LEFT: Assessment & Training
    # =========================================================
    sbg(ax, 0.02, 0.05, 0.76, 0.47, "Assessment\n& Training", "#455A64")

    rbox(ax, 0.04, 0.42, 0.23, BOX_H2,
         "Task Presentation\n(Layer 2 parameters)", C_LAYER2)
    rbox(ax, 0.30, 0.42, 0.20, BOX_H2,
         "Distractor Generation\n(mirror, part-permuted)", C_GEN)
    rbox(ax, 0.53, 0.42, 0.22, BOX_H2,
         "3D Presentation\n(browser / VR planned)", "#455A64")

    rbox(ax, 0.04, 0.35, 0.23, BOX_H2,
         "Layer 3: Behavioral Metrics\n(RT, accuracy, rotation)", C_LAYER3)
    rbox(ax, 0.30, 0.35, 0.20, BOX_H2,
         "Performance Data\n(per-trial, per-skill)", C_LAYER3)

    rbox(ax, 0.04, 0.12, 0.46, 0.10,
         "LLM Coaching Layer\n"
         "Cross-dimensional diagnostic synthesis\n"
         "Strategy recommendations · Adaptive sequencing",
         C_LLM)

    # Pipeline → presentation
    arr(ax, 0.61, 0.69, 0.61, 0.46, color=C_GEN, lw=1.5)
    ax.text(0.625, 0.58, "shape +\ndistractors", fontsize=BOX_FS,
            color=C_TEXT)

    arr(ax, 0.20, 0.56, 0.20, 0.46)
    arr(ax, 0.27, 0.44, 0.30, 0.44)
    arr(ax, 0.50, 0.44, 0.53, 0.44)
    arr(ax, 0.64, 0.42, 0.64, 0.395)
    arr(ax, 0.30, 0.37, 0.27, 0.37)
    arr(ax, 0.27, 0.35, 0.27, 0.225)

    # =========================================================
    # BOTTOM RIGHT: Adaptive Loop
    # =========================================================
    sbg(ax, 0.84, 0.05, 0.38, 0.47, "Adaptive Loop", C_LLM)

    rbox(ax, 0.86, 0.42, 0.34, BOX_H2,
         "Difficulty Adjustment\n(skill-targeted adaptation)", C_LLM)
    rbox(ax, 0.86, 0.35, 0.34, BOX_H,
         "Updated Cognitive Specification", C_MAP)
    rbox(ax, 0.86, 0.22, 0.34, BOX_H,
         "Next Trial → Generation Pipeline", C_GEN)

    # LLM → next trial
    arr(ax, 0.50, 0.17, 0.86, 0.23, color=C_LLM, lw=1.2)
    arr(ax, 1.03, 0.42, 1.03, 0.385)
    arr(ax, 1.03, 0.35, 1.03, 0.255)

    # Loop back
    arr(ax, 1.20, 0.25, 1.20, 0.55, color=C_MAP, lw=1.5, cs="arc3,rad=0.25")
    ax.text(1.24, 0.42, "next\ntrial", fontsize=BOX_FS, color=C_MAP,
            rotation=90, ha="left", va="center")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    # Save publication-quality outputs before displaying.
    # PDF is vector (infinite resolution); the PNG is 600 DPI for journals
    # that require raster.
    out_dir = os.path.join(os.path.dirname(__file__), "docs", "figures")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "Figure_1_System_Architecture.pdf")
    png_path = os.path.join(out_dir, "Figure_1_System_Architecture.png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")

    print("Displaying architecture diagram. Close window when done.")
    plt.show()


if __name__ == "__main__":
    main()
