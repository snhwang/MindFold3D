"""
Layered Cognitive Framework for MindFold 3D.

Copyright (c) 2026 Scott N. Hwang, Parviz Safadel. All rights reserved.
Patent pending. See docs/PATENT_SPECIFICATION.md for claims.

This module implements the Bidirectional Cognitive-Geometric Mapping Engine,
a core novel component of the MindFold 3D invention (Claims 1, 2, 4, 6).

Key inventive methods:
  - get_difficulty_spec():    Forward cognitive-to-geometric mapping (Claim 4b)
  - get_skeleton_spec():       Direct cognitive-to-skeleton parameterization (Claim 3a)
  - reverse_map_cognitive_profile(): Reverse geometric-to-cognitive mapping (Claim 4d-f)
  - perturb_skeleton_spec():   Tiered distractor specification (Claim 14)

Architecture:
  Layer 1 (Shape Geometry): 3 dimensions that drive the generator's scoring function.
  Layer 2 (Task Design): 5 dimensions that control how shapes are presented.
  Layer 3 (Behavioral Metrics): Measured from responses, not generated.

References:
  - Newcombe & Shipley (2015) Intrinsic/Extrinsic x Static/Dynamic typology
  - Carroll (1993) factor-analytic framework
  - Shepard & Metzler (1971) mental rotation
  - Luck & Vogel (1997) visuospatial working memory
"""

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Any

from shape_features import ShapeFeatureSet


# =============================================================================
# Layer 1: Shape Geometry Dimensions
#
# These map directly to features optimized by generate_shape_advanced().
# Each feature has a scoring weight in the generator and difficulty ranges.
# =============================================================================

SHAPE_DIMENSIONS: Dict[str, Dict[str, Any]] = {
    "spatial_form": {
        "description": "3D elongation and bilateral symmetry of the shape",
        "quadrant": "Intrinsic-Static",
        "rationale": (
            "Anisotropy creates orientation-dependent views that increase rotation difficulty "
            "(Linn & Petersen, 1985; Bethell-Fox & Shepard, 1988). "
            "Symmetry affects encoding ease and mirror detectability: symmetric shapes are "
            "easier to encode and maintain (Wagemans et al., 2012) but cannot be chiral; "
            "asymmetric shapes are required for mirror discrimination and produce stronger "
            "rotational signatures (Corballis, 1988)."
        ),
        "features": {
            "anisotropy_index": {
                # Ranges adjusted to match geometrically achievable values.
                # For 11 connected voxels, a compact (spherical) shape achieves
                # AI ≈ 0.15-0.35 due to discrete-grid effects (off-diagonal
                # covariance from edge-adjacent voxels prevents AI < 0.15).
                # Ranges are non-overlapping with small gaps to avoid ambiguous
                # classification at tier boundaries.
                "low": (0.0, 0.40),
                "medium": (0.45, 0.65),
                "high": (0.65, 0.82),
                "expert": (0.85, 0.99),
                "generator_weight": -150,
            },
            "symmetry_score": {
                # Inverted: high difficulty = LOW symmetry (harder to encode, enables chirality).
                # 0 = fully asymmetric, 1 = perfectly bilaterally symmetric.
                # For N=11 voxels, symmetry values are discrete multiples of 1/11:
                # 4/11=0.364, 5/11=0.455.  Setting the high/medium boundary at 0.46
                # captures the 5/11 step within the high tier, since the perceptual
                # distinction at this granularity is not empirically meaningful.
                "low": (0.7, 1.0),
                "medium": (0.46, 0.7),
                "high": (0.1, 0.46),
                # Note: 0.15 is geometrically impossible for connected high-anisotropy
                # shapes (any midplane-crossing edge creates a mirror pair at min 2/N).
                # For N=11 voxels, the practical minimum is 2/11≈0.182; for N=22 it's
                # 2/22≈0.091.  Setting the upper bound at 0.20 makes the tier achievable
                # at medium scale while still reflecting highly asymmetric shapes.
                "expert": (0.0, 0.20),
                "generator_weight": 0,  # measured post-generation, not actively optimized
            },
        },
    },
    "structural_complexity": {
        "description": "Topological structure: branching, connectivity, and cycles (holes/loops)",
        "quadrant": "Intrinsic-Static",
        "rationale": (
            "Branching creates part-whole decomposition demands (Hoffman & Richards, "
            "1984; Bethell-Fox & Shepard, 1988). Multiple components require parallel "
            "spatial encoding of distinct entities (Just & Carpenter, 1985). "
            "Cyclic structure has two operationalisations: for the graph-cyclic "
            "tiers, the circuit rank μ = |E| − |V| + C of the voxel 6-adjacency "
            "graph (Diestel, 2024) — the connectivity-complexity invariant the "
            "skeleton-first generator directly controls; for the hole tiers, the "
            "cubical first Betti number β₁ — the count of volumetric holes/tunnels "
            "(Kong & Rosenfeld, 1989), enforced by construction plus per-shape "
            "validation. The two invariants are distinct: graph cycles from dense "
            "2×2 face patches carry no volumetric hole, so μ and β₁ diverge on "
            "thick shapes. Both are measured and recorded for every shape."
        ),
        "features": {
            "branching_factor": {
                # Expert upper bound 10: the 4-ring chain template (expert β₁
                # tier) has 9 wall junctions inherently; ranges are calibrated
                # to geometrically achievable values under the all-Betti
                # generator.
                "low": (0, 1),
                "medium": (2, 3),
                "high": (4, 5),
                "expert": (6, 10),
                "generator_weight": -50,
            },
            "number_of_components": {
                "low": 1,
                "medium": 1,
                "high": (2, 3),
                "expert": (2, 4),
                "generator_weight": 1000,
            },
            "betti_1": {
                # Cubical first Betti number: actual volumetric holes/tunnels.
                # Non-overlapping: low=no holes, medium=one, high=2-3, expert=4+.
                # ALL cyclic tiers are generated by the hole-template generator
                # (HoleMarkedLoopSkeleton) with per-shape cubical-β₁ validation;
                # acyclic tiers (tree/chiral) are validated to β₁ = 0. Circuit
                # rank μ remains an internal fill/optimizer control quantity and
                # an unscored diagnostic — it is NOT a scored feature.
                "low": 0,
                "medium": 1,
                "high": (2, 3),
                "expert": 4,
                "generator_weight": -75,
            },
        },
    },
    "spatial_scale": {
        "description": "Overall size of the shape, scaling working memory and rotation load",
        "quadrant": "Intrinsic-Static",
        "rationale": (
            "Voxel count is the best-validated predictor of mental rotation RT "
            "(Bethell-Fox & Shepard, 1988; R²=0.85+) and spatial working memory load. "
            "Performance inflects around 12-15 cubes, consistent with ~4-item visual "
            "working memory capacity (Luck & Vogel, 1997)."
        ),
        "features": {
            "voxel_count": {
                "low": (5, 8),
                "medium": (9, 13),
                "high": (14, 18),
                "expert": (19, 25),
                "generator_weight": 0,  # set as input parameter, not optimized
            },
        },
    },
}


# =============================================================================
# Layer 2: Task Design Dimensions
#
# These control how shapes are presented, not what shapes look like.
# Each maps to a presentation parameter in the frontend/API.
# =============================================================================

TASK_DIMENSIONS: Dict[str, Dict[str, Any]] = {
    "mental_rotation": {
        "description": "Whether shapes are randomly rotated in 3D",
        "quadrant": "Intrinsic-Dynamic",
        "rationale": (
            "RT increases linearly with angular disparity (Shepard & Metzler, 1971). "
            "Corresponds to Carroll's Speeded Rotation factor."
        ),
        "parameter": "rotation_enabled",
        "levels": {
            "low": False,
            "medium": True,
            "high": True,
            "expert": True,
        },
    },
    "mirror_discrimination": {
        "description": "Include mirror-reflected shapes as distractors",
        "quadrant": "Intrinsic-Dynamic",
        "rationale": (
            "Mirror discrimination is neurally and behaviorally distinct from "
            "rotation (Corballis, 1988; Cooper & Shepard, 1973). Requires "
            "chirality/handedness detection."
        ),
        "parameter": "include_mirror",
        "levels": {
            "low": False,
            "medium": False,
            "high": True,
            "expert": True,
        },
    },
    "working_memory": {
        "description": "Temporal paradigm for target presentation",
        "quadrant": "Cross-cutting",
        "rationale": (
            "Delayed matching taxes visuospatial working memory maintenance "
            "(Baddeley, 2003; Luck & Vogel, 1997). Capacity ~4 items (Cowan, 2001)."
        ),
        "parameter": "wm_mode",
        "levels": {
            "low": "simultaneous",
            "medium": "delayed-short",
            "high": "delayed-long",
            "expert": "delayed-long",
        },
    },
    "configural_binding": {
        "description": "Include part-permuted distractors testing spatial arrangement encoding",
        "quadrant": "Extrinsic-Static",
        "rationale": (
            "Part-permuted distractors force encoding of spatial arrangement, not just "
            "part identity (Hummel & Biederman, 1992; Jiang et al., 2000)."
        ),
        "parameter": "include_part_permuted",
        "levels": {
            "low": False,
            "medium": False,
            "high": True,
            "expert": True,
        },
    },
    "perspective_taking": {
        "description": "Viewpoint-dependent matching: target and choices shown from different camera angles",
        "quadrant": "Extrinsic-Dynamic",
        "rationale": (
            "Perspective taking is factorially separable from mental rotation "
            "(Hegarty & Waller, 2004; Kozhevnikov & Hegarty, 2001). "
            "Engages temporo-parietal junction, not posterior parietal cortex. "
            "Difficulty scales with angular distance between viewpoints."
        ),
        "parameter": "perspective_mode",
        "levels": {
            "low": False,
            "medium": False,
            "high": True,
            "expert": True,
        },
    },
}


# =============================================================================
# Output dataclasses
# =============================================================================

@dataclass
class TaskParameters:
    """Parameters that control task presentation (Layer 2)."""
    include_mirror: bool = False
    wm_mode: str = "simultaneous"
    rotation_enabled: bool = True
    include_part_permuted: bool = False
    perspective_mode: bool = False


@dataclass
class DifficultySpec:
    """Combined output: shape generation targets + task presentation parameters."""
    shape_features: ShapeFeatureSet = field(default_factory=lambda: ShapeFeatureSet(voxel_count=8))
    task_params: TaskParameters = field(default_factory=TaskParameters)
    shape_difficulties: Dict[str, str] = field(default_factory=dict)
    task_difficulties: Dict[str, str] = field(default_factory=dict)
    recommended_archetype: str = "tree"


@dataclass
class SkeletonSpec:
    """Direct skeleton parameters from cognitive dimensions.

    Bypasses numeric feature targets — cognitive dimensions directly
    parameterize the skeleton rules. Features are measured post-generation
    for verification only.
    """
    archetype: str = "tree"
    voxel_count: int = 10
    grid_size: Tuple[int, int, int] = (7, 7, 7)
    # From structural_complexity
    num_branches: int = 3           # tree: 2 (low) / 3 (medium) / 5 (high) / 7 (expert)
    num_loops: int = 1              # bridge: 0 (low) / 1 (medium) / 2 (high) / 4 (expert)
    num_components: int = 1         # disconnected voxel clusters: 1/1/2/3
    # From spatial_form
    direction_spread: str = "moderate"  # "planar", "moderate", "elongated_3d"
    planarity: str = "medium"           # "low", "medium", "high"
    # From spatial_scale
    packing: str = "medium"             # "sparse", "medium", "dense"
    # Hole-tier (topological) targets. When target_b1 is not None the shape is
    # generated by HoleMarkedLoopSkeleton and validated against cubical β₁; the
    # post-generation geometric optimizer is bypassed to preserve topology.
    target_b1: Optional[int] = None
    skip_optimizer: bool = False
    # Source cognitive difficulties (for verification)
    shape_difficulties: Dict[str, str] = field(default_factory=dict)
    task_difficulties: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# Main mapping function
# =============================================================================

def _resolve_range(spec: Any) -> float:
    """Pick midpoint from a (min, max) tuple, or return scalar directly."""
    if isinstance(spec, tuple) and len(spec) == 2:
        return (spec[0] + spec[1]) / 2.0
    return spec


def get_difficulty_spec(
    shape_difficulties: Dict[str, Literal["low", "medium", "high", "expert"]],
    task_difficulties: Dict[str, Literal["low", "medium", "high", "expert"]],
    target_voxel_count: int = 10,
) -> DifficultySpec:
    """
    Map cognitive difficulty settings to shape targets and task parameters.

    Args:
        shape_difficulties: Difficulty for each Layer 1 dimension.
            Keys: "spatial_form", "structural_complexity", "spatial_scale"
        task_difficulties: Difficulty for each Layer 2 dimension.
            Keys: "mental_rotation", "mirror_discrimination", "working_memory",
                  "configural_binding", "perspective_taking"
        target_voxel_count: Desired number of voxels (randomized per-question).

    Returns:
        DifficultySpec with shape_features (ShapeFeatureSet) and task_params (TaskParameters).
    """
    # --- Layer 1: Shape targets ---
    # Default to "medium" for any unspecified dimension
    defaults = {dim: "medium" for dim in SHAPE_DIMENSIONS}
    defaults.update(shape_difficulties)

    # β₁ target for the Structural Complexity tier (all-Betti design): any
    # tier with β₁ ≥ 1 is generated by the hole-template generator on the
    # 11³ grid, with a feasibility floor on voxel count (k linked rings at
    # s=3 need 8/14/20/24 skeleton voxels for k=1..4).
    _b1_spec = SHAPE_DIMENSIONS["structural_complexity"]["features"]["betti_1"].get(
        defaults.get("structural_complexity", "medium")
    )
    _b1_target = round(_resolve_range(_b1_spec)) if _b1_spec is not None else 0

    is_expert = any(v == "expert" for v in shape_difficulties.values())
    if _b1_target >= 1:
        grid = HOLE_TIER_GRID
        target_voxel_count = max(target_voxel_count, HOLE_B1_VOXEL_FLOORS.get(_b1_target, 8))
    else:
        grid = (10, 10, 10) if is_expert else (7, 7, 7)
    feature_values: Dict[str, Any] = {
        "voxel_count": target_voxel_count,
        "grid_size": grid,
    }

    for dim_name, dim_config in SHAPE_DIMENSIONS.items():
        difficulty = defaults[dim_name]
        for feature_name, feature_config in dim_config["features"].items():
            range_spec = feature_config.get(difficulty)
            if range_spec is not None:
                value = _resolve_range(range_spec)
                # Round integers for int fields
                if feature_name in ("branching_factor", "number_of_components", "betti_1", "voxel_count"):
                    value = round(value)
                feature_values[feature_name] = value

    # Enforce the hole-template feasibility floor after the spatial_scale
    # range has resolved voxel_count (k rings need 8/14/20/24 voxels).
    if _b1_target >= 1:
        feature_values["voxel_count"] = max(
            feature_values["voxel_count"], HOLE_B1_VOXEL_FLOORS.get(_b1_target, 8)
        )

    # Build ShapeFeatureSet (only set fields that exist)
    valid_keys = ShapeFeatureSet.model_fields.keys()
    filtered = {k: v for k, v in feature_values.items() if k in valid_keys}

    try:
        shape_features = ShapeFeatureSet(**filtered)
    except Exception as e:
        print(f"Error creating ShapeFeatureSet: {e}")
        shape_features = ShapeFeatureSet(voxel_count=target_voxel_count, grid_size=grid)

    # --- Layer 2: Task parameters ---
    task_defaults = {dim: "medium" for dim in TASK_DIMENSIONS}
    task_defaults.update(task_difficulties)

    task_values = {}
    for dim_name, dim_config in TASK_DIMENSIONS.items():
        difficulty = task_defaults[dim_name]
        param_name = dim_config["parameter"]
        task_values[param_name] = dim_config["levels"][difficulty]

    task_params = TaskParameters(**task_values)

    # Determine recommended archetype based on cognitive difficulty.
    # All-Betti design: any tier whose β₁ target is ≥ 1 is generated by the
    # hole-template generator, which guarantees cubical β₁ by construction
    # plus validation. Hole overrides chiral (a hole shape is not guaranteed
    # chiral — see design-review note in the v7 change list).
    archetype = "tree"
    if task_defaults.get("mental_rotation") in ("high", "expert") or task_defaults.get("mirror_discrimination") in ("high", "expert"):
        archetype = "chiral"
    if shape_features.betti_1 and shape_features.betti_1 > 0:
        archetype = "hole"

    return DifficultySpec(
        shape_features=shape_features,
        task_params=task_params,
        shape_difficulties=defaults,
        task_difficulties=task_defaults,
        recommended_archetype=archetype,
    )


# =============================================================================
# Direct Cognitive-to-Skeleton Mapping
#
# Maps cognitive difficulty settings directly to skeleton parameters,
# bypassing numeric feature targets. Features are measured post-generation
# for verification only.
# =============================================================================

def get_skeleton_spec(
    shape_difficulties: Dict[str, str],
    task_difficulties: Dict[str, str],
    target_voxel_count: int = 10,
) -> SkeletonSpec:
    """
    Map cognitive difficulty settings directly to skeleton parameters.

    Unlike get_difficulty_spec() which produces numeric feature targets,
    this produces structural parameters that skeleton rules consume directly.

    Args:
        shape_difficulties: Difficulty for each Layer 1 dimension.
        task_difficulties: Difficulty for each Layer 2 dimension.
        target_voxel_count: Desired number of voxels.

    Returns:
        SkeletonSpec with direct skeleton parameters.
    """
    # Default to "medium" for any unspecified dimension
    sd = {dim: "medium" for dim in SHAPE_DIMENSIONS}
    sd.update(shape_difficulties)
    td = {dim: "medium" for dim in TASK_DIMENSIONS}
    td.update(task_difficulties)

    sc = sd.get("structural_complexity", "medium")
    sf = sd.get("spatial_form", "medium")
    sscale = sd.get("spatial_scale", "medium")

    # --- Archetype selection ---
    # Default archetype selection based on cognitive dimensions.
    # Extended skeletons (lamina, mesh, spiral, foam, bundle, fractal) can be
    # requested explicitly via shape_difficulties or task_difficulties using
    # the key "archetype_hint". The core selection logic below handles the
    # original three archetypes; the hint mechanism allows the new six.
    archetype = "tree"
    # Chiral for high/expert spatial_form: low symmetry requires deliberately
    # asymmetric construction (unequal legs).  Also selected for rotation/mirror tasks.
    if sf in ("high", "expert"):
        archetype = "chiral"
    if td.get("mental_rotation") in ("high", "expert") or td.get("mirror_discrimination") in ("high", "expert"):
        archetype = "chiral"

    # --- spatial_scale → voxel_count and packing ---
    # Resolved first so num_components gating can use it.
    scale_voxel_map = {"low": 7, "medium": 11, "high": 16, "expert": 22}
    resolved_voxel_count = target_voxel_count if target_voxel_count != 10 else scale_voxel_map[sscale]
    packing = {"low": "sparse", "medium": "medium", "high": "dense", "expert": "dense"}[sscale]

    # --- structural_complexity → num_branches, num_loops, num_components ---
    num_branches = {"low": 2, "medium": 3, "high": 5, "expert": 7}[sc]
    num_loops = {"low": 0, "medium": 1, "high": 2, "expert": 4}[sc]

    # All-Betti design: every cyclic tier (num_loops ≥ 1) is generated by the
    # hole-template generator, which guarantees cubical β₁ = num_loops by
    # construction plus per-shape validation. Hole overrides chiral (hole
    # shapes are not guaranteed chiral — see v7 change-list design note).
    # Cyclic tiers are single-component (the hole generator builds one
    # connected shape) and require the template feasibility floor on voxels.
    if num_loops >= 1:
        archetype = "hole"
        resolved_voxel_count = max(
            resolved_voxel_count, HOLE_B1_VOXEL_FLOORS.get(num_loops, 8)
        )
        num_components = 1
    else:
        # Multi-component requires sufficient voxels per component to generate
        # meaningful branching.  With too few voxels, splitting into
        # disconnected blobs eliminates structural complexity rather than
        # adding it.  Thresholds: high SC needs ~8 voxels/component (min 16
        # total for 2); expert SC needs ~7 voxels/component (min 21 for 3).
        raw_num_components = {"low": 1, "medium": 1, "high": 2, "expert": 3}[sc]
        if raw_num_components == 2 and resolved_voxel_count < 16:
            num_components = 1
        elif raw_num_components == 3 and resolved_voxel_count < 21:
            num_components = 2 if resolved_voxel_count >= 14 else 1
        else:
            num_components = raw_num_components

    # --- spatial_form → direction_spread ---
    # High anisotropy (high difficulty) → elongated 3D spread.
    # Low anisotropy (low difficulty) → compact/spherical (NOT planar: a flat
    # shape has one near-zero PCA eigenvalue → high anisotropy, opposite of intent).
    direction_spread_map = {
        "low": "compact",
        "medium": "moderate",
        "high": "elongated_3d",
        "expert": "elongated_3d",
    }
    direction_spread = direction_spread_map[sf]
    planarity = sf  # skeleton rule still uses planarity level for fill strategy

    # Grid size: 11³ for hole shapes (template placement room), otherwise
    # 10×10×10 for expert and 7×7×7 for the rest.
    is_expert = any(v == "expert" for v in sd.values())
    if archetype == "hole":
        grid = HOLE_TIER_GRID
    else:
        grid = (10, 10, 10) if is_expert else (7, 7, 7)

    return SkeletonSpec(
        archetype=archetype,
        voxel_count=resolved_voxel_count,
        grid_size=grid,
        num_branches=num_branches,
        num_loops=num_loops,
        num_components=num_components,
        direction_spread=direction_spread,
        planarity=planarity,
        packing=packing,
        target_b1=num_loops if archetype == "hole" else None,
        shape_difficulties=sd,
        task_difficulties=td,
    )


# =============================================================================
# Hole tiers (topological construction mode)
#
# Hole tiers target cubical β₁ (volumetric holes) rather than graph circuit
# rank. Generated by HoleMarkedLoopSkeleton on an 11³ grid with per-shape
# cubical-β₁ validation; the geometric optimizer is bypassed. Voxel-count
# ranges: minimum = template skeleton size for k linked rings at s=3
# (8, 14, 20, 24 for k = 1..4); maximum = the empirically safe fill ceiling
# below which accidental handles stay rare (see fidelity_study_hole_tiers).
# =============================================================================

HOLE_TIERS: Dict[str, Dict[str, Any]] = {
    "hole_1": {"target_b1": 1, "voxel_range": (8, 25), "default_voxels": 15},
    "hole_2": {"target_b1": 2, "voxel_range": (14, 30), "default_voxels": 20},
    "hole_3": {"target_b1": 3, "voxel_range": (20, 30), "default_voxels": 25},
    "hole_4": {"target_b1": 4, "voxel_range": (24, 35), "default_voxels": 30},
}

HOLE_TIER_GRID: Tuple[int, int, int] = (11, 11, 11)

# Feasibility floor on voxel count for k reserved holes: the minimum skeleton
# size of k linked s=3 rings (single ring 8; measured template minima
# 8/13/18/23 for k=1..4) PLUS fill budget for junction-biased branching.
# k=2 floor is 17 (13-voxel template + 4 fill voxels) so the SC-high tier
# can reach its branching-factor range (4-5) via junction-biased fill.
HOLE_B1_VOXEL_FLOORS: Dict[int, int] = {1: 8, 2: 17, 3: 20, 4: 24}


def get_hole_tier_spec(
    tier: str,
    target_voxel_count: Optional[int] = None,
) -> SkeletonSpec:
    """Build a SkeletonSpec for a hole tier (Hole-1 … Hole-4).

    Args:
        tier: One of "hole_1" … "hole_4" (or an int 1-4).
        target_voxel_count: Optional voxel count; clamped to the tier's safe
            range. Defaults to the tier's default_voxels.

    Returns:
        SkeletonSpec with archetype="hole", target_b1 set, and
        skip_optimizer=True (the geometric optimizer preserves circuit rank
        but can plug reserved-hole tunnels, destroying β₁).
    """
    if isinstance(tier, int):
        tier = f"hole_{tier}"
    if tier not in HOLE_TIERS:
        raise ValueError(f"Unknown hole tier {tier!r}; expected one of {list(HOLE_TIERS)}")
    cfg = HOLE_TIERS[tier]
    lo, hi = cfg["voxel_range"]
    vc = cfg["default_voxels"] if target_voxel_count is None else max(lo, min(hi, target_voxel_count))
    k = cfg["target_b1"]
    return SkeletonSpec(
        archetype="hole",
        voxel_count=vc,
        grid_size=HOLE_TIER_GRID,
        num_branches=max(3, k + 2),
        num_loops=k,
        num_components=1,
        direction_spread="planar",
        planarity="medium",
        packing="medium",
        target_b1=k,
        skip_optimizer=True,
        shape_difficulties={"structural_complexity": tier},
        task_difficulties={},
    )


def perturb_skeleton_spec(base_spec: SkeletonSpec, distractor_tier: int) -> SkeletonSpec:
    """
    Create a perturbed SkeletonSpec for distractor generation.

    Tier 0 = radical (obviously different), Tier 1 = moderate, Tier 2 = subtle.
    Perturbs skeleton parameters directly instead of numeric features.

    Args:
        base_spec: The target shape's SkeletonSpec.
        distractor_tier: 0 (radical), 1 (moderate), or 2 (subtle).

    Returns:
        New SkeletonSpec with perturbed parameters.
    """
    import copy as _copy
    spec = _copy.deepcopy(base_spec)

    if distractor_tier == 0:
        # RADICAL: obviously different structure
        spec.voxel_count = random.randint(5, 7)
        spec.packing = random.choice(["sparse", "dense"])
        spec.num_branches = random.choice([1, 5]) if spec.archetype == "tree" else spec.num_branches
        spec.direction_spread = random.choice(["planar", "moderate", "elongated_3d"])
    else:
        # Build a pool of perturbable parameters
        perturbations = []

        # Voxel count
        perturbations.append(("voxel_count", None))
        # Direction spread
        perturbations.append(("direction_spread", None))
        # Packing
        perturbations.append(("packing", None))
        # Branching (only for tree)
        if spec.archetype == "tree":
            perturbations.append(("num_branches", None))
        # Planarity
        perturbations.append(("planarity", None))

        random.shuffle(perturbations)

        # Tier 1: perturb 2-3 params, Tier 2: perturb 1
        num_perturb = random.randint(2, 3) if distractor_tier == 1 else 1

        for i in range(min(num_perturb, len(perturbations))):
            param = perturbations[i][0]

            if param == "voxel_count":
                if distractor_tier == 1:
                    spec.voxel_count += random.choice([-2, -1, 1, 2])
                else:
                    spec.voxel_count += random.choice([-1, 1])
                max_voxels = 25 if base_spec.grid_size == (10, 10, 10) else 15
                spec.voxel_count = max(5, min(max_voxels, spec.voxel_count))

            elif param == "direction_spread":
                options = ["planar", "moderate", "elongated_3d"]
                options = [o for o in options if o != spec.direction_spread]
                spec.direction_spread = random.choice(options)

            elif param == "packing":
                options = ["sparse", "medium", "dense"]
                options = [o for o in options if o != spec.packing]
                spec.packing = random.choice(options)

            elif param == "num_branches":
                delta = random.choice([-2, -1, 1, 2]) if distractor_tier == 1 else random.choice([-1, 1])
                max_branches = 9 if base_spec.grid_size == (10, 10, 10) else 6
                spec.num_branches = max(1, min(max_branches, spec.num_branches + delta))

            elif param == "planarity":
                options = ["low", "medium", "high"]
                if base_spec.grid_size == (10, 10, 10):
                    options.append("expert")
                options = [o for o in options if o != spec.planarity]
                spec.planarity = random.choice(options)

    return spec


# =============================================================================
# Convenience: list all generator-scored feature names
# =============================================================================

def get_scored_feature_names() -> list:
    """Return names of features that the generator actively optimizes."""
    names = []
    for dim_config in SHAPE_DIMENSIONS.values():
        names.extend(dim_config["features"].keys())
    return names


# =============================================================================
# Reverse Cognitive Mapping: Structure → Cognitive Profile
#
# Maps analyzed shape features back to cognitive difficulty classifications,
# verifying that generated shapes embody their intended cognitive challenges.
# =============================================================================

@dataclass
class FeatureClassification:
    """Classification of a single feature value against defined difficulty ranges."""
    feature_name: str
    value: float
    classified_level: str   # "low", "medium", "high", or "expert"
    confidence: str         # "in_range" or "nearest" (for gap values)


@dataclass
class DimensionProfile:
    """Aggregated difficulty classification for one cognitive dimension."""
    level: str                              # "low", "medium", "high", "expert", or "mixed"
    features: List[FeatureClassification] = field(default_factory=list)
    agreement: float = 0.0                  # 0.0 to 1.0


@dataclass
class CognitiveProfile:
    """Verified cognitive profile derived from analyzed shape features."""
    shape_dimensions: Dict[str, DimensionProfile] = field(default_factory=dict)
    task_suitability: Dict[str, Any] = field(default_factory=dict)
    overall_fidelity: float = 1.0           # 0.0 to 1.0 vs intended spec


def _classify_feature_value(feature_name: str, value: float,
                            feature_config: dict) -> FeatureClassification:
    """Classify a feature value into low/medium/high/expert using SHAPE_DIMENSIONS ranges.

    When a value falls in overlapping ranges (e.g., number_of_components where
    both low=1 and medium=1), prefer 'medium' to avoid systematic low-bias.
    """
    levels = ["low", "medium", "high", "expert"]

    # Collect all matching levels (some ranges overlap)
    matches = []
    for level in levels:
        range_spec = feature_config.get(level)
        if range_spec is None:
            continue
        if isinstance(range_spec, tuple) and len(range_spec) == 2:
            lo, hi = range_spec
            if lo <= value <= hi:
                matches.append(level)
        else:
            if value == range_spec:
                matches.append(level)

    if matches:
        # When ambiguous, prefer the highest matching level (expert > high > medium > low)
        # This avoids systematic low-bias for overlapping ranges
        # (e.g., number_of_components where low=1 and medium=1 overlap,
        #  or compactness_score where high and expert ranges overlap)
        level_priority = {"expert": 3, "high": 2, "medium": 1, "low": 0}
        best_match = max(matches, key=lambda l: level_priority.get(l, 0))
        return FeatureClassification(feature_name, value, best_match, "in_range")

    # Value falls in a gap — find nearest range boundary
    best_level = "medium"
    best_dist = float("inf")
    for level in levels:
        range_spec = feature_config.get(level)
        if range_spec is None:
            continue
        if isinstance(range_spec, tuple) and len(range_spec) == 2:
            lo, hi = range_spec
            dist = min(abs(value - lo), abs(value - hi))
        else:
            dist = abs(value - range_spec)
        if dist < best_dist:
            best_dist = dist
            best_level = level

    return FeatureClassification(feature_name, value, best_level, "nearest")


def _aggregate_dimension(classifications: List[FeatureClassification]) -> DimensionProfile:
    """Aggregate per-feature classifications into a dimension-level profile."""
    if not classifications:
        return DimensionProfile(level="unknown", features=[], agreement=0.0)

    votes = Counter(fc.classified_level for fc in classifications)
    most_common_level, most_common_count = votes.most_common(1)[0]

    total = len(classifications)
    if most_common_count > total / 2:
        level = most_common_level
    else:
        level = "mixed"

    agreement = most_common_count / total
    return DimensionProfile(level=level, features=classifications, agreement=agreement)


def reverse_map_cognitive_profile(
    analyzed_features: ShapeFeatureSet,
    intended_difficulties: Optional[Dict[str, str]] = None,
) -> CognitiveProfile:
    """
    Reverse-map analyzed shape features to cognitive difficulty classifications.

    Takes the output of analyze_shape_features() and classifies each feature
    back into the low/medium/high ranges defined in SHAPE_DIMENSIONS.

    Args:
        analyzed_features: ShapeFeatureSet with computed geometric values.
        intended_difficulties: Optional dict of intended shape dimension difficulties,
                               used to compute overall_fidelity.

    Returns:
        CognitiveProfile with per-dimension and per-feature classifications.
    """
    features_dict = analyzed_features.model_dump()
    shape_dimensions = {}

    for dim_name, dim_config in SHAPE_DIMENSIONS.items():
        classifications = []
        for feature_name, feature_config in dim_config["features"].items():
            value = features_dict.get(feature_name)
            if value is not None:
                classifications.append(
                    _classify_feature_value(feature_name, float(value), feature_config)
                )
        shape_dimensions[dim_name] = _aggregate_dimension(classifications)

    # Task suitability from non-dimension properties
    is_chiral = features_dict.get("is_chiral")
    task_suitability = {
        "is_chiral": is_chiral,
        "mirror_discrimination_suitable": bool(is_chiral),
        "archetype": features_dict.get("archetype"),
    }

    # Compute overall fidelity if intended difficulties provided
    fidelity = 1.0
    if intended_difficulties:
        match_count = 0
        total_dims = 0
        for dim_name, intended_level in intended_difficulties.items():
            if dim_name in shape_dimensions:
                total_dims += 1
                verified = shape_dimensions[dim_name]
                if verified.level == intended_level:
                    match_count += 1
                elif verified.level == "mixed":
                    # Partial credit if intended level is among the feature votes
                    feature_levels = [f.classified_level for f in verified.features]
                    if intended_level in feature_levels:
                        match_count += 0.5
        fidelity = match_count / total_dims if total_dims > 0 else 1.0

    return CognitiveProfile(
        shape_dimensions=shape_dimensions,
        task_suitability=task_suitability,
        overall_fidelity=fidelity,
    )


def cognitive_profile_to_dict(profile: CognitiveProfile) -> dict:
    """Convert CognitiveProfile to a JSON-serializable dict."""
    result = {
        "shape_dimensions": {},
        "task_suitability": profile.task_suitability,
        "overall_fidelity": profile.overall_fidelity,
    }
    for dim_name, dim_profile in profile.shape_dimensions.items():
        result["shape_dimensions"][dim_name] = {
            "level": dim_profile.level,
            "agreement": dim_profile.agreement,
            "features": [
                {
                    "feature": fc.feature_name,
                    "value": fc.value,
                    "level": fc.classified_level,
                    "confidence": fc.confidence,
                }
                for fc in dim_profile.features
            ],
        }
    return result


# =============================================================================
# Test / demo
# =============================================================================

if __name__ == "__main__":
    spec = get_difficulty_spec(
        shape_difficulties={
            "spatial_form": "high",
            "structural_complexity": "high",
            "spatial_scale": "medium",
        },
        task_difficulties={
            "mental_rotation": "high",
            "mirror_discrimination": "high",
            "working_memory": "medium",
        },
        target_voxel_count=12,
    )

    print("Shape targets:")
    for field_name, value in spec.shape_features.model_dump().items():
        default = ShapeFeatureSet.model_fields[field_name].default
        if value != default and value is not None:
            print(f"  {field_name}: {value}")

    print(f"\nTask parameters:")
    print(f"  include_mirror: {spec.task_params.include_mirror}")
    print(f"  wm_mode: {spec.task_params.wm_mode}")
    print(f"  rotation_enabled: {spec.task_params.rotation_enabled}")

    print(f"\nScored features: {get_scored_feature_names()}")
