"""Cognitive dimension assessor for spatial cognition training."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..core.datapoint import DataPoint
from ..core.definitions import AssessmentLevel, ThresholdConfig
from ..utils.stats import calculate_mean


@dataclass
class CognitiveDimensionMapping:
    """Mapping of features to cognitive dimensions."""
    name: str
    description: str
    features: List[str]
    difficulty_feature: Optional[str] = None
    weights: Optional[Dict[str, float]] = None

    def get_weight(self, feature: str) -> float:
        """Get weight for a feature."""
        if self.weights and feature in self.weights:
            return self.weights[feature]
        return 1.0


# Default cognitive dimension mappings for spatial cognition training
DEFAULT_COGNITIVE_DIMENSIONS = [
    CognitiveDimensionMapping(
        name="mental_transformation",
        description="Ability to mentally rotate and transform 3D objects",
        features=[
            "rotation_confusability",
            "rotational_symmetry_order",
            "mental_operation_type",
        ],
    ),
    CognitiveDimensionMapping(
        name="mirror_discrimination",
        description="Ability to distinguish mirror images",
        features=[
            "mirror_confusability",
            "reflection_planes_count",
            "symmetry_class",
        ],
    ),
    CognitiveDimensionMapping(
        name="spatial_visualization",
        description="Ability to visualize and manipulate 3D space",
        features=[
            "convexity",
            "planarity_score",
            "number_of_concave_edges",
            "shape_form_index",
        ],
    ),
    CognitiveDimensionMapping(
        name="spatial_navigation",
        description="Ability to navigate in 3D space",
        features=[
            "bounding_box_ratio",
            "center_of_mass_offset",
            "anisotropy_index",
            "dominant_axis",
        ],
    ),
    CognitiveDimensionMapping(
        name="visual_perception",
        description="Visual perceptual processing and integration",
        features=[
            "gestalt_grouping_cues",
            "color_confusability",
            "face_color_pattern",
            "voxel_color_variance",
        ],
    ),
    CognitiveDimensionMapping(
        name="structural_analysis",
        description="Ability to analyze and segment complex structures",
        features=[
            "branching_factor",
            "number_of_components",
            "largest_component_ratio",
            "compactness_score",
        ],
    ),
    CognitiveDimensionMapping(
        name="executive_strategy",
        description="Strategic planning and decision-making",
        features=[
            "difficulty_score",
            "distractor_similarity",
            "prior_error_rate",
        ],
    ),
    CognitiveDimensionMapping(
        name="spatial_working_memory",
        description="Ability to hold and manipulate spatial information",
        features=[
            "voxel_count",
            "surface_area",
            "compactness_score",
        ],
    ),
]


@dataclass
class CognitiveDimensionResult:
    """Assessment result for a cognitive dimension."""
    dimension_name: str
    description: str
    score: float
    level: AssessmentLevel
    feature_scores: Dict[str, float]
    contributing_features: int
    total_attempts: int
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimension_name": self.dimension_name,
            "description": self.description,
            "score": self.score,
            "level": self.level.value,
            "feature_scores": self.feature_scores,
            "contributing_features": self.contributing_features,
            "total_attempts": self.total_attempts,
            "details": self.details,
        }


@dataclass
class CognitiveProfile:
    """Complete cognitive profile assessment."""
    user_id: Optional[str]
    dimensions: Dict[str, CognitiveDimensionResult]
    overall_score: float
    overall_level: AssessmentLevel
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    total_attempts: int
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "dimensions": {k: v.to_dict() for k, v in self.dimensions.items()},
            "overall_score": self.overall_score,
            "overall_level": self.overall_level.value,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations,
            "total_attempts": self.total_attempts,
            "details": self.details,
        }


class CognitiveDimensionAssessor:
    """
    Assessor for cognitive dimension analysis in spatial training.

    Maps feature-level performance to cognitive dimensions and
    provides actionable insights.
    """

    def __init__(
        self,
        dimension_mappings: Optional[List[CognitiveDimensionMapping]] = None,
        thresholds: Optional[ThresholdConfig] = None,
        min_attempts_per_feature: int = 3,
    ):
        """
        Initialize the cognitive dimension assessor.

        Args:
            dimension_mappings: Custom dimension-to-feature mappings
            thresholds: Assessment thresholds
            min_attempts_per_feature: Minimum attempts to consider a feature
        """
        self._dimensions = dimension_mappings or DEFAULT_COGNITIVE_DIMENSIONS
        self._thresholds = thresholds or ThresholdConfig(
            excellent=0.9,
            good=0.7,
            average=0.5,
            below_average=0.3,
            poor=0.1,
        )
        self._min_attempts = min_attempts_per_feature

        # Build feature to dimension lookup
        self._feature_to_dimensions: Dict[str, List[str]] = {}
        for dim in self._dimensions:
            for feature in dim.features:
                if feature not in self._feature_to_dimensions:
                    self._feature_to_dimensions[feature] = []
                self._feature_to_dimensions[feature].append(dim.name)

    def assess_dimension(
        self,
        dimension_name: str,
        feature_stats: Dict[str, Dict[str, Any]],
    ) -> CognitiveDimensionResult:
        """
        Assess a single cognitive dimension.

        Args:
            dimension_name: Name of the dimension
            feature_stats: Feature-level statistics

        Returns:
            Dimension assessment result
        """
        # Find dimension mapping
        dimension = None
        for dim in self._dimensions:
            if dim.name == dimension_name:
                dimension = dim
                break

        if dimension is None:
            return CognitiveDimensionResult(
                dimension_name=dimension_name,
                description="Unknown dimension",
                score=0.0,
                level=AssessmentLevel.CRITICAL,
                feature_scores={},
                contributing_features=0,
                total_attempts=0,
                details={"error": "Dimension not found"},
            )

        # Calculate feature scores
        feature_scores = {}
        total_attempts = 0
        weighted_score_sum = 0.0
        weight_sum = 0.0

        for feature in dimension.features:
            if feature not in feature_stats:
                continue

            feature_data = feature_stats[feature]

            # Aggregate across all values of this feature
            correct = 0
            incorrect = 0

            for value_key, value_stats in feature_data.items():
                if isinstance(value_stats, dict):
                    correct += value_stats.get("correct", 0)
                    incorrect += value_stats.get("incorrect", 0)

            attempts = correct + incorrect
            if attempts >= self._min_attempts:
                success_rate = correct / attempts if attempts > 0 else 0
                feature_scores[feature] = success_rate

                weight = dimension.get_weight(feature)
                weighted_score_sum += success_rate * weight
                weight_sum += weight
                total_attempts += attempts

        # Calculate dimension score
        if weight_sum > 0:
            dimension_score = weighted_score_sum / weight_sum
        else:
            dimension_score = 0.0

        # Determine level
        level = self._thresholds.get_level(dimension_score)

        return CognitiveDimensionResult(
            dimension_name=dimension_name,
            description=dimension.description,
            score=dimension_score,
            level=level,
            feature_scores=feature_scores,
            contributing_features=len(feature_scores),
            total_attempts=total_attempts,
            details={
                "mapped_features": dimension.features,
                "features_with_data": list(feature_scores.keys()),
            },
        )

    def assess_all_dimensions(
        self,
        feature_stats: Dict[str, Dict[str, Any]],
    ) -> Dict[str, CognitiveDimensionResult]:
        """
        Assess all cognitive dimensions.

        Args:
            feature_stats: Feature-level statistics

        Returns:
            Dict of dimension name to assessment result
        """
        results = {}
        for dimension in self._dimensions:
            results[dimension.name] = self.assess_dimension(
                dimension.name,
                feature_stats,
            )
        return results

    def create_cognitive_profile(
        self,
        feature_stats: Dict[str, Dict[str, Any]],
        user_id: Optional[str] = None,
    ) -> CognitiveProfile:
        """
        Create a comprehensive cognitive profile.

        Args:
            feature_stats: Feature-level statistics
            user_id: Optional user identifier

        Returns:
            Complete cognitive profile
        """
        # Assess all dimensions
        dimension_results = self.assess_all_dimensions(feature_stats)

        # Calculate overall score
        valid_scores = [
            r.score for r in dimension_results.values()
            if r.contributing_features > 0
        ]
        overall_score = calculate_mean(valid_scores) if valid_scores else 0.0
        overall_level = self._thresholds.get_level(overall_score)

        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []

        for dim_name, result in dimension_results.items():
            if result.contributing_features == 0:
                continue

            if result.level in [AssessmentLevel.EXCELLENT, AssessmentLevel.GOOD]:
                strengths.append(dim_name)
            elif result.level in [AssessmentLevel.POOR, AssessmentLevel.CRITICAL]:
                weaknesses.append(dim_name)

        # Sort by score
        strengths.sort(
            key=lambda d: dimension_results[d].score,
            reverse=True,
        )
        weaknesses.sort(
            key=lambda d: dimension_results[d].score,
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            dimension_results,
            weaknesses,
        )

        # Calculate total attempts
        total_attempts = sum(
            r.total_attempts for r in dimension_results.values()
        )

        return CognitiveProfile(
            user_id=user_id,
            dimensions=dimension_results,
            overall_score=overall_score,
            overall_level=overall_level,
            strengths=strengths[:3],  # Top 3
            weaknesses=weaknesses[:3],  # Bottom 3
            recommendations=recommendations,
            total_attempts=total_attempts,
            details={
                "dimensions_assessed": len(dimension_results),
                "dimensions_with_data": len(valid_scores),
            },
        )

    def _generate_recommendations(
        self,
        results: Dict[str, CognitiveDimensionResult],
        weaknesses: List[str],
    ) -> List[str]:
        """Generate training recommendations based on results."""
        recommendations = []

        recommendation_map = {
            "mental_transformation": "Practice mental rotation exercises with increasing complexity",
            "mirror_discrimination": "Focus on shapes with subtle mirror differences",
            "spatial_visualization": "Work with more complex 3D structures and concave shapes",
            "spatial_navigation": "Practice with elongated and asymmetric shapes",
            "visual_perception": "Focus on pattern recognition with varying color schemes",
            "structural_analysis": "Work with branching and multi-component structures",
            "executive_strategy": "Increase difficulty gradually and analyze error patterns",
            "spatial_working_memory": "Practice with larger structures and more voxels",
        }

        for weakness in weaknesses[:3]:
            if weakness in recommendation_map:
                recommendations.append(recommendation_map[weakness])

        if not recommendations:
            if results:
                avg_score = calculate_mean([r.score for r in results.values() if r.contributing_features > 0])
                if avg_score < 0.5:
                    recommendations.append("Continue regular practice to build foundational skills")
                elif avg_score < 0.7:
                    recommendations.append("Challenge yourself with higher difficulty levels")
                else:
                    recommendations.append("Excellent progress! Try varying training conditions")

        return recommendations

    def get_feature_to_dimension_map(self) -> Dict[str, List[str]]:
        """Get the mapping of features to cognitive dimensions."""
        return self._feature_to_dimensions.copy()

    def get_dimension_features(self, dimension_name: str) -> List[str]:
        """Get features for a specific dimension."""
        for dim in self._dimensions:
            if dim.name == dimension_name:
                return dim.features.copy()
        return []
