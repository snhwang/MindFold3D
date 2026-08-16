"""
Architecture Diagram for Paper — MindFold 3D

Copyright (c) 2026 Scott N. Hwang, Parviz Safadel. All rights reserved.

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
BOX_FS = 13             # uniform font size in all boxes
BOX_H = 0.030           # single-line box height
BOX_H2 = 0.045          # two-line box height
BOX_PAD = "round,pad=0.003"  # minimal internal padding


def rbox(ax, x, y, w, h, text, color, text_color="white", alpha=0.9,
         fs=None):
    """Draw a rounded rectangle with centered text."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=BOX_PAD,
        facecolor=color, edgecolor="none", alpha=alpha, zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fs if fs is not None else BOX_FS,
            color=text_color, zorder=3)


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
    fig, ax = plt.subplots(1, 1, figsize=(15, 13))
    ax.set_xlim(0, 1.32)
    ax.set_ylim(0, 1.20)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Three-Layer Cognitive Framework key
    ax.text(0.02, 1.16, "Three-Layer Cognitive Framework:",
            ha="left", va="center", fontsize=14, fontweight="bold",
            color=C_TEXT)
    rbox(ax, 0.40, 1.145, 0.26, 0.030,
         "Layer 1: Shape Geometry", C_LAYER1)
    rbox(ax, 0.69, 1.145, 0.24, 0.030,
         "Layer 2: Task Design", C_LAYER2)
    rbox(ax, 0.96, 1.145, 0.30, 0.030,
         "Layer 3: Behavioral Metrics", C_LAYER3)

    # Bidirectional mapping label
    ax.annotate("", xy=(0.42, 1.02), xytext=(0.84, 1.02),
                arrowprops=dict(arrowstyle="<->", color=C_MAP, lw=2.5))
    ax.text(0.63, 1.04, "Bidirectional Cognitive-Geometric Mapping",
            ha="center", fontsize=14, fontweight="bold", color=C_MAP)

    # =========================================================
    # LEFT: Cognitive Specification
    # =========================================================
    sbg(ax, 0.02, 0.56, 0.36, 0.43, "Cognitive Specification", C_MAP)

    rbox(ax, 0.04, 0.93, 0.32, BOX_H, "Difficulty Settings", C_MAP)

    # Layer 1
    rbox(ax, 0.04, 0.87, 0.15, BOX_H, "Spatial Form", C_LAYER1)
    rbox(ax, 0.21, 0.87, 0.15, BOX_H, "Struct. Complexity", C_LAYER1)
    rbox(ax, 0.04, 0.83, 0.15, BOX_H, "Spatial Scale", C_LAYER1)
    ax.text(0.20, 0.905, "Layer 1: Shape Geometry",
            fontsize=BOX_FS + 1, fontweight="bold",
            color=C_LAYER1, ha="center", zorder=5)

    # Layer 2
    rbox(ax, 0.04, 0.76, 0.15, BOX_H, "Mental Rotation", C_LAYER2)
    rbox(ax, 0.21, 0.76, 0.15, BOX_H, "Mirror Discrim.", C_LAYER2)
    rbox(ax, 0.04, 0.72, 0.15, BOX_H, "Working Memory", C_LAYER2)
    rbox(ax, 0.21, 0.72, 0.15, BOX_H, "Config. Binding", C_LAYER2)
    rbox(ax, 0.04, 0.68, 0.15, BOX_H, "Perspective Taking", C_LAYER2)
    ax.text(0.20, 0.80, "Layer 2: Task Design",
            fontsize=BOX_FS + 1, fontweight="bold",
            color=C_LAYER2, ha="center", zorder=5)

    # =========================================================
    # CENTER: Generation Pipeline
    # =========================================================
    sbg(ax, 0.42, 0.56, 0.38, 0.43, "Generation Pipeline", C_GEN)

    rbox(ax, 0.44, 0.93, 0.34, BOX_H,
         "Forward Mapping (cognitive → skeleton)", C_MAP)

    rbox(ax, 0.44, 0.875, 0.10, BOX_H, "Tree", C_GEN)
    rbox(ax, 0.56, 0.875, 0.10, BOX_H, "Chiral", C_GEN)
    rbox(ax, 0.68, 0.875, 0.10, BOX_H, "Hole", C_GEN)

    rbox(ax, 0.44, 0.82, 0.34, BOX_H,
         "Generated 3D Voxel Shape", C_GEN)

    rbox(ax, 0.44, 0.745, 0.34, BOX_H2,
         "Validation\n(certify β₁ + branching; regenerate on miss)", C_GEN,
         fs=BOX_FS - 1.5)

    rbox(ax, 0.44, 0.67, 0.34, BOX_H2,
         "Feature Measurement\n(ground-truth vector)", "#455A64")

    rbox(ax, 0.44, 0.595, 0.34, BOX_H2,
         "Reverse Mapping\n(features → cognitive tiers)", C_MAP)

    # Arrows in pipeline (snap to box edges with uniform 0.03 gaps)
    arr(ax, 0.61, 0.93, 0.61, 0.905)
    arr(ax, 0.61, 0.875, 0.61, 0.85)
    arr(ax, 0.61, 0.82, 0.61, 0.79)
    arr(ax, 0.61, 0.745, 0.61, 0.715)
    arr(ax, 0.61, 0.67, 0.61, 0.64)

    # Validation miss → regenerate (back to the archetype templates)
    arr(ax, 0.44, 0.7675, 0.47, 0.875, color=C_GEN, lw=1.2,
        cs="arc3,rad=0.35")
    ax.text(0.425, 0.845, "miss", fontsize=BOX_FS - 2, color=C_GEN,
            ha="right", style="italic")

    # Spec → forward mapping
    arr(ax, 0.38, 0.94, 0.44, 0.94)

    # =========================================================
    # RIGHT: Cognitive Profile & Scoring
    # =========================================================
    sbg(ax, 0.84, 0.56, 0.38, 0.43, "Cognitive Profile & Scoring", C_MAP)

    rbox(ax, 0.86, 0.93, 0.34, BOX_H, "Classified Difficulty Profile", C_MAP)
    rbox(ax, 0.86, 0.87, 0.34, BOX_H, "Fidelity Score (intended vs. realized)", "#7E57C2")
    rbox(ax, 0.86, 0.81, 0.34, BOX_H, "Ground-Truth Feature Vector", "#455A64")
    rbox(ax, 0.86, 0.75, 0.34, BOX_H, "Scoring uses measured features", "#455A64")

    # Reverse map → Profile (center-y of Profile = 0.945)
    arr(ax, 0.78, 0.6175, 0.86, 0.945, color=C_MAP, lw=1.5)
    # Profile → Fidelity
    arr(ax, 1.03, 0.93, 1.03, 0.90)
    # Measurement → Ground-Truth (center-y of Ground-Truth = 0.825)
    arr(ax, 0.78, 0.6925, 0.86, 0.825, color="#455A64", lw=1.5, cs="arc3,rad=-0.1")
    # Ground-Truth → Scoring
    arr(ax, 1.03, 0.81, 1.03, 0.78)

    # =========================================================
    # BOTTOM LEFT: Assessment Trial
    # =========================================================
    sbg(ax, 0.02, 0.05, 0.76, 0.47, "Assessment Trial", "#455A64")

    rbox(ax, 0.04, 0.42, 0.30, BOX_H2,
         "Task Presentation\n(Layer 2 parameters)", C_LAYER2)
    rbox(ax, 0.45, 0.42, 0.30, BOX_H2,
         "3D Presentation\n(browser / VR planned)", "#78909C")

    rbox(ax, 0.04, 0.30, 0.34, BOX_H2,
         "Layer 3: Behavioral Metrics\n(RT, accuracy, rotation)", C_LAYER3)
    rbox(ax, 0.42, 0.30, 0.34, BOX_H2,
         "Performance Data\n(per-trial, per-feature)", C_LAYER3)

    # Pipeline → 3D Presentation (shape + distractors generated together)
    arr(ax, 0.61, 0.595, 0.61, 0.465, color=C_GEN, lw=1.5)
    ax.text(0.625, 0.525, "shape +\ndistractors", fontsize=BOX_FS,
            color=C_TEXT)

    # Layer 2 (top section) → Task Presentation
    # x=0.30 placed past the section title to avoid obscuring "Trial".
    arr(ax, 0.30, 0.56, 0.30, 0.465)
    # Task Presentation → 3D Presentation
    arr(ax, 0.34, 0.44, 0.45, 0.44)
    # 3D Presentation → Layer 3 Metrics (straight diagonal, user response).
    arr(ax, 0.60, 0.42, 0.38, 0.34,
        color=C_LAYER3, lw=1.5)
    ax.text(0.42, 0.40, "user response", fontsize=BOX_FS - 1,
            color=C_LAYER3, ha="left", style="italic")
    # Layer 3 → Performance Data
    arr(ax, 0.38, 0.32, 0.42, 0.32)

    # =========================================================
    # BOTTOM RIGHT: Adaptive Loop & Optional Coaching
    # =========================================================
    sbg(ax, 0.84, 0.05, 0.38, 0.47, "Adaptive Loop & Optional Coaching", C_LLM)

    rbox(ax, 0.86, 0.40, 0.34, BOX_H2,
         "Difficulty Adjustment\n(deterministic, per-feature)", C_LLM)
    rbox(ax, 0.86, 0.33, 0.34, BOX_H,
         "Updated Cognitive Specification", C_MAP)
    rbox(ax, 0.86, 0.26, 0.34, BOX_H,
         "Next Trial → Generation Pipeline", C_GEN)
    rbox(ax, 0.86, 0.09, 0.34, 0.10,
         "Optional LLM Coaching\n"
         "advisory feedback to the player;\n"
         "never affects difficulty or scoring",
         C_LLM, alpha=0.7)

    # Performance Data → Difficulty Adjustment (the deterministic loop)
    arr(ax, 0.76, 0.33, 0.86, 0.4225,
        color=C_LLM, lw=1.5, cs="arc3,rad=-0.15")
    # Difficulty Adjustment → Updated Spec
    arr(ax, 1.03, 0.40, 1.03, 0.36)
    # Updated Spec → Next Trial
    arr(ax, 1.03, 0.33, 1.03, 0.29)
    # Performance Data → Optional Coaching (advisory side path)
    arr(ax, 0.76, 0.305, 0.86, 0.14,
        color=C_LLM, lw=1.2, cs="arc3,rad=0.15")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    # Save publication-quality outputs before displaying.
    # PDF is vector (infinite resolution); the PNG is 600 DPI for journals
    # that require raster.
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")
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
