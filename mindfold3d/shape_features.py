"""
Shape Feature Vector Schema for MindFold 3D.

Copyright (c) 2026 Scott N. Hwang, Parviz Safadel. All rights reserved.
Patent pending. See docs/PATENT_SPECIFICATION.md for claims.

This module defines the ShapeFeatureSet data structure used throughout the
MindFold 3D invention for characterizing 3D voxel shapes across geometric,
topological, and task-level dimensions (Claims 1b, 2, 12, 13).
"""
from typing import List, Literal, Optional, Tuple, Dict, Any

from pydantic import BaseModel, Field


class ShapeFeatureSet(BaseModel):
    """
    Shape feature vector for MindFold 3D.

    Fields fall into three categories:
    - Generator-scored: targeted by difficulty settings and optimized during generation
    - Emergent: computed from geometry post-generation, useful for performance tracking
    - Task-level: describe the task context, not shape geometry
    """

    # --- Input parameters ---
    voxel_count: int = Field(..., ge=3, le=40, description="Total number of voxels in the shape (graph-cyclic tiers ≤25; hole tiers ≤35 plus cavity-closure drift)")
    grid_size: Tuple[int, int, int] = Field((7, 7, 7), description="3D grid bounding box size")

    # --- Generator-scored features (optimized by generate_shape_advanced) ---
    branching_factor: Optional[int] = Field(0, ge=0, description="Number of branch points in the shape graph (weight: -50)")
    compactness_score: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Mean neighbor density, normalized (weight: -100)")
    number_of_components: Optional[int] = Field(1, ge=1, description="Disconnected voxel clusters via flood-fill (weight: +/-1000)")
    planarity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Max fraction of voxels in one axis-aligned plane (weight: -75)")
    anisotropy_index: Optional[float] = Field(None, ge=0.0, le=1.0, description="Directional bias from PCA: 0=isotropic, 1=elongated (weight: -150)")
    shape_form_index: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Anisotropy type from PCA: -1=rod, +1=pancake (weight: -75)")
    circuit_rank: Optional[int] = Field(
        0,
        ge=0,
        description=(
            "Circuit rank μ = |E| − |V| + C of the voxel 6-adjacency graph — the number of "
            "independent cycles in the graph. This is the structural-complexity quantity the "
            "skeleton-first generator directly controls for the graph-cyclic tiers "
            "(TreeSkeleton, LoopSkeleton, AsymmetricSkeleton). A graph cycle does not imply a "
            "volumetric hole: dense 2×2 face patches contribute cycles with no continuum hole "
            "(see betti_1). Scoring weight: -75."
        ),
    )
    betti_1: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "First Betti number β₁ of the shape as a cubical complex — the number of "
            "independent volumetric holes/tunnels. Measured post-generation for every shape "
            "via topology_extras.cubical_betti_1 (Euler-characteristic method). Hole-tier "
            "shapes (HoleMarkedLoopSkeleton) target this feature directly, enforced by "
            "construction plus per-shape validation. Not equal to circuit_rank in general: "
            "μ > β₁ on shapes with 2-thick face patches, and β₁ > μ is possible via "
            "edge-adjacency tunnels invisible to the face-adjacency graph."
        ),
    )

    # --- Emergent features (computed from geometry, not targeted) ---
    surface_area: Optional[int] = Field(None, ge=0, description="Total exposed faces across all voxels")
    symmetry_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Bilateral symmetry: max voxel overlap when reflected across any axis-aligned midplane (0=asymmetric, 1=perfectly symmetric)")
    bounding_box_ratio: Optional[float] = Field(None, ge=1.0, description="Aspect ratio of axis-aligned bounding box")
    center_of_mass_offset: Optional[Tuple[float, float, float]] = Field((0.0, 0.0, 0.0), description="Displacement from grid center")
    dominant_axis: Optional[Literal['x', 'y', 'z', 'balanced']] = Field('balanced', description="Longest spatial direction")
    largest_component_ratio: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Size of largest component / total voxels")
    is_chiral: Optional[bool] = Field(None, description="Whether shape differs from its mirror reflection on at least one axis (True if any axis-aligned mirror image does not match the original)")

    # --- Task-level descriptors (set during question assembly, not shape geometry) ---
    distractor_similarity: Optional[Literal['mirror', 'rotation', 'extra_voxel', 'missing_voxel', 'part_permuted', 'none']] = Field('none', description="How distractors differ from target")
    mental_operation_type: Optional[List[Literal[
        'mirror', 'rotation', 'part-whole', 'sequencing',
        '3D assembly', 'rotate_x', 'mirror_z', 'align_3D']]] = Field(default_factory=lambda: ['rotation'], description="Cognitive skill required")

    # --- Generation Tracking ---
    generation_mode: Optional[str] = Field("greedy", description="Which algorithmic engine produced this shape")
    archetype: Optional[str] = Field(None, description="The structural rule archetype used, if any")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with tuples serialized as lists for JSON."""
        result = self.model_dump()
        for key, value in result.items():
            if isinstance(value, tuple):
                result[key] = list(value)
        return result
