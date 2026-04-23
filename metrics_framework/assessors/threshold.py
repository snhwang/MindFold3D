"""Threshold-based performance assessor."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..core.definitions import (
    ThresholdConfig,
    AssessmentLevel,
    MetricDefinition,
)
from ..core.datapoint import DataPoint
from ..utils.stats import calculate_mean


@dataclass
class AssessmentResult:
    """Result of a threshold assessment."""
    metric_name: str
    value: float
    level: AssessmentLevel
    threshold_used: float
    target: Optional[float] = None
    deviation_from_target: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "level": self.level.value,
            "threshold_used": self.threshold_used,
            "target": self.target,
            "deviation_from_target": self.deviation_from_target,
            "details": self.details,
        }

    @property
    def is_passing(self) -> bool:
        """Check if assessment is at least average."""
        return self.level in [
            AssessmentLevel.EXCELLENT,
            AssessmentLevel.GOOD,
            AssessmentLevel.AVERAGE,
        ]

    @property
    def needs_improvement(self) -> bool:
        """Check if metric needs improvement."""
        return self.level in [
            AssessmentLevel.BELOW_AVERAGE,
            AssessmentLevel.POOR,
            AssessmentLevel.CRITICAL,
        ]


class ThresholdAssessor:
    """
    Assessor that evaluates metrics against configurable thresholds.

    Supports multiple assessment levels and customizable thresholds.
    """

    DEFAULT_THRESHOLDS = ThresholdConfig(
        excellent=0.9,
        good=0.7,
        average=0.5,
        below_average=0.3,
        poor=0.1,
    )

    def __init__(
        self,
        thresholds: Optional[ThresholdConfig] = None,
        higher_is_better: bool = True,
    ):
        """
        Initialize the threshold assessor.

        Args:
            thresholds: Threshold configuration
            higher_is_better: Whether higher values are better
        """
        self._thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self._higher_is_better = higher_is_better

    def assess_value(
        self,
        value: float,
        metric_name: str = "metric",
        target: Optional[float] = None,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ) -> AssessmentResult:
        """
        Assess a single value.

        Args:
            value: Value to assess
            metric_name: Name of the metric
            target: Target value
            min_value: Minimum possible value
            max_value: Maximum possible value

        Returns:
            Assessment result
        """
        # Normalize to 0-1 range
        range_size = max_value - min_value
        if range_size > 0:
            normalized = (value - min_value) / range_size
        else:
            normalized = value

        # Get level
        level = self._thresholds.get_level(normalized, not self._higher_is_better)

        # Determine threshold used
        threshold_map = {
            AssessmentLevel.EXCELLENT: self._thresholds.excellent,
            AssessmentLevel.GOOD: self._thresholds.good,
            AssessmentLevel.AVERAGE: self._thresholds.average,
            AssessmentLevel.BELOW_AVERAGE: self._thresholds.below_average,
            AssessmentLevel.POOR: self._thresholds.poor,
            AssessmentLevel.CRITICAL: 0.0,
        }

        # Calculate deviation from target
        deviation = None
        if target is not None:
            deviation = value - target

        return AssessmentResult(
            metric_name=metric_name,
            value=value,
            level=level,
            threshold_used=threshold_map[level],
            target=target,
            deviation_from_target=deviation,
            details={
                "normalized_value": normalized,
                "higher_is_better": self._higher_is_better,
                "min_value": min_value,
                "max_value": max_value,
            },
        )

    def assess_datapoints(
        self,
        datapoints: List[DataPoint],
        metric_definition: Optional[MetricDefinition] = None,
    ) -> AssessmentResult:
        """
        Assess a collection of datapoints.

        Args:
            datapoints: DataPoints to assess
            metric_definition: Optional metric definition for context

        Returns:
            Assessment result
        """
        if not datapoints:
            return AssessmentResult(
                metric_name="unknown",
                value=0.0,
                level=AssessmentLevel.CRITICAL,
                threshold_used=0.0,
                details={"error": "No datapoints provided"},
            )

        values = [dp.value for dp in datapoints]
        mean_value = calculate_mean(values)
        metric_name = datapoints[0].metric_name

        # Get bounds from metric definition
        min_val = 0.0
        max_val = 1.0
        target = None
        higher_is_better = self._higher_is_better

        if metric_definition:
            if metric_definition.min_value is not None:
                min_val = metric_definition.min_value
            if metric_definition.max_value is not None:
                max_val = metric_definition.max_value
            target = metric_definition.target_value
            higher_is_better = metric_definition.higher_is_better

        result = self.assess_value(
            mean_value,
            metric_name,
            target,
            min_val,
            max_val,
        )

        result.details.update({
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "all_values": values,
        })

        return result

    def assess_multiple(
        self,
        metrics: Dict[str, float],
        definitions: Optional[Dict[str, MetricDefinition]] = None,
    ) -> Dict[str, AssessmentResult]:
        """
        Assess multiple metrics at once.

        Args:
            metrics: Dict of metric_name -> value
            definitions: Optional dict of metric definitions

        Returns:
            Dict of metric_name -> AssessmentResult
        """
        definitions = definitions or {}
        results = {}

        for metric_name, value in metrics.items():
            metric_def = definitions.get(metric_name)

            min_val = 0.0
            max_val = 1.0
            target = None

            if metric_def:
                if metric_def.min_value is not None:
                    min_val = metric_def.min_value
                if metric_def.max_value is not None:
                    max_val = metric_def.max_value
                target = metric_def.target_value

            results[metric_name] = self.assess_value(
                value,
                metric_name,
                target,
                min_val,
                max_val,
            )

        return results

    def get_summary_assessment(
        self,
        results: Dict[str, AssessmentResult],
    ) -> Dict[str, Any]:
        """
        Get a summary assessment from multiple results.

        Args:
            results: Dict of assessment results

        Returns:
            Summary assessment
        """
        if not results:
            return {"overall_level": AssessmentLevel.CRITICAL.value}

        levels = [r.level for r in results.values()]
        level_counts = {}
        for level in AssessmentLevel:
            level_counts[level.value] = sum(1 for l in levels if l == level)

        # Determine overall level (weighted by severity)
        level_weights = {
            AssessmentLevel.CRITICAL: 0,
            AssessmentLevel.POOR: 1,
            AssessmentLevel.BELOW_AVERAGE: 2,
            AssessmentLevel.AVERAGE: 3,
            AssessmentLevel.GOOD: 4,
            AssessmentLevel.EXCELLENT: 5,
        }

        avg_weight = sum(level_weights[l] for l in levels) / len(levels)

        # Map back to level
        overall_level = AssessmentLevel.AVERAGE
        for level, weight in level_weights.items():
            if avg_weight >= weight:
                overall_level = level

        # Find metrics needing attention
        needs_attention = [
            name for name, result in results.items()
            if result.needs_improvement
        ]

        return {
            "overall_level": overall_level.value,
            "metrics_count": len(results),
            "level_distribution": level_counts,
            "needs_attention": needs_attention,
            "passing_count": sum(1 for r in results.values() if r.is_passing),
            "failing_count": sum(1 for r in results.values() if r.needs_improvement),
        }
