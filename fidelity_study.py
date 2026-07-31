"""Fidelity study: how well do generated shapes match intended cognitive difficulty?

Generates shapes at each difficulty tier (low/medium/high/expert) across all
archetype paths, then reverse-maps features back to cognitive levels and reports
dimension-level match rates.
"""
import random
from collections import defaultdict

from cognitive_mapping import (
    get_skeleton_spec, reverse_map_cognitive_profile, cognitive_profile_to_dict,
    SHAPE_DIMENSIONS,
)
from skeleton_generation import generate_shape_skeleton
from shape_features import ShapeFeatureSet

N_TRIALS = 20

# All tier × archetype-path combinations to study
CONFIGS = [
    # Standard tiers (no rotation/mirror → tree or bridge based on SC level)
    {"label": "low-tree", "shape": {"spatial_form": "low", "structural_complexity": "low", "spatial_density": "low"},
     "task": {"mental_rotation": "low", "mirror_discrimination": "low"}, "voxels": 8},
    {"label": "med-tree", "shape": {"spatial_form": "medium", "structural_complexity": "medium", "spatial_density": "medium"},
     "task": {"mental_rotation": "low", "mirror_discrimination": "low"}, "voxels": 10},
    {"label": "high-bridge", "shape": {"spatial_form": "high", "structural_complexity": "high", "spatial_density": "high"},
     "task": {"mental_rotation": "low", "mirror_discrimination": "low"}, "voxels": 12},
    {"label": "high-chiral", "shape": {"spatial_form": "high", "structural_complexity": "high", "spatial_density": "high"},
     "task": {"mental_rotation": "high", "mirror_discrimination": "high"}, "voxels": 12},
    # Expert tiers
    {"label": "expert-bridge", "shape": {"spatial_form": "expert", "structural_complexity": "expert", "spatial_density": "expert"},
     "task": {"mental_rotation": "low", "mirror_discrimination": "low"}, "voxels": 22},
    {"label": "expert-chiral", "shape": {"spatial_form": "expert", "structural_complexity": "expert", "spatial_density": "expert"},
     "task": {"mental_rotation": "expert", "mirror_discrimination": "expert"}, "voxels": 20},
    # Expert with mixed dimensions (more realistic)
    {"label": "expert-mixed", "shape": {"spatial_form": "high", "structural_complexity": "expert", "spatial_density": "high"},
     "task": {"mental_rotation": "low", "mirror_discrimination": "low"}, "voxels": 18},
]

def run_study():
    print("=" * 80)
    print("FIDELITY STUDY: Generated Shape → Reverse Cognitive Mapping")
    print("=" * 80)

    for config in CONFIGS:
        label = config["label"]
        shape_diffs = config["shape"]
        task_diffs = config["task"]
        target_voxels = config["voxels"]

        spec = get_skeleton_spec(shape_diffs, task_diffs, target_voxel_count=target_voxels)

        dim_match_counts = defaultdict(int)
        dim_feature_values = defaultdict(list)  # Track actual feature values

        for trial in range(N_TRIALS):
            shape = generate_shape_skeleton(spec)
            feats = shape["features"]

            # Build SFS for reverse mapping
            valid_keys = ShapeFeatureSet.model_fields.keys()
            sfs = ShapeFeatureSet(**{k: v for k, v in feats.items() if k in valid_keys and v is not None})
            profile = reverse_map_cognitive_profile(sfs, shape_diffs)

            for dim_name in SHAPE_DIMENSIONS:
                intended = shape_diffs.get(dim_name, "medium")
                actual = profile.shape_dimensions[dim_name].level
                if actual == intended:
                    dim_match_counts[dim_name] += 1

                # Collect per-feature values for debugging
                for fc in profile.shape_dimensions[dim_name].features:
                    dim_feature_values[fc.feature_name].append(fc.value)

        print(f"\n--- {label} (archetype={spec.archetype}, voxels={target_voxels}, grid={spec.grid_size}) ---")
        print(f"    Intended: {shape_diffs}")
        overall_fidelity = 0
        for dim_name in SHAPE_DIMENSIONS:
            intended = shape_diffs.get(dim_name, "medium")
            match_rate = dim_match_counts[dim_name] / N_TRIALS
            overall_fidelity += match_rate
            status = "OK" if match_rate >= 0.7 else "LOW" if match_rate >= 0.4 else "FAIL"
            print(f"    {dim_name}: {match_rate:.0%} match (intended={intended}) [{status}]")

        overall_fidelity /= len(SHAPE_DIMENSIONS)
        print(f"    Overall fidelity: {overall_fidelity:.0%}")

        # Show mean feature values for problem dimensions
        print(f"    Feature means:")
        for feat_name, values in sorted(dim_feature_values.items()):
            mean = sum(values) / len(values)
            print(f"      {feat_name}: {mean:.3f} (n={len(values)})")


if __name__ == "__main__":
    run_study()
