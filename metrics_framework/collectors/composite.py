"""Composite collector for multi-dimensional metric tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import MetricCollector
from .counter import SuccessFailureCounter
from .timer import TimerCollector
from .histogram import CategoricalHistogram

if TYPE_CHECKING:
    from ..core.engine import MetricsEngine


@dataclass
class FeaturePerformance:
    """Performance data for a single feature value."""
    feature_name: str
    feature_value: Any
    correct: int = 0
    incorrect: int = 0
    response_times: List[float] = field(default_factory=list)

    @property
    def total_attempts(self) -> int:
        """Get total attempts."""
        return self.correct + self.incorrect

    @property
    def success_rate(self) -> float:
        """Get success rate (0-1)."""
        if self.total_attempts == 0:
            return 0.0
        return self.correct / self.total_attempts

    @property
    def avg_response_time(self) -> float:
        """Get average response time."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "feature_value": self.feature_value,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "total_attempts": self.total_attempts,
            "success_rate": self.success_rate,
            "avg_response_time": self.avg_response_time,
            "response_times": self.response_times,
        }


class CompositeCollector(MetricCollector):
    """
    Composite collector for tracking performance across multiple dimensions.

    Ideal for tracking performance by feature, category, or other dimensions.
    """

    def __init__(
        self,
        metric_name: str,
        engine: Optional["MetricsEngine"] = None,
        dimensions: Optional[Dict[str, Any]] = None,
        auto_record: bool = True,
    ):
        super().__init__(metric_name, engine, dimensions, auto_record)
        self._performance: Dict[str, Dict[str, FeaturePerformance]] = {}
        self._total_correct = 0
        self._total_incorrect = 0
        self._all_response_times: List[float] = []

    def collect(self, value: Any, **kwargs) -> Dict[str, Any]:
        """
        Collect a performance observation.

        Args:
            value: Dict with 'correct' (bool), 'response_time' (float),
                   and feature dimensions
            **kwargs: Additional parameters

        Returns:
            Summary of the observation
        """
        if not isinstance(value, dict):
            raise ValueError("CompositeCollector requires a dict value")

        correct = value.get("correct", False)
        response_time = value.get("response_time", 0.0)
        features = value.get("features", {})

        # Update totals
        if correct:
            self._total_correct += 1
        else:
            self._total_incorrect += 1

        if response_time > 0:
            self._all_response_times.append(response_time)

        # Update per-feature stats
        for feature_name, feature_value in features.items():
            if feature_name not in self._performance:
                self._performance[feature_name] = {}

            value_key = str(feature_value)
            if value_key not in self._performance[feature_name]:
                self._performance[feature_name][value_key] = FeaturePerformance(
                    feature_name=feature_name,
                    feature_value=feature_value,
                )

            perf = self._performance[feature_name][value_key]
            if correct:
                perf.correct += 1
            else:
                perf.incorrect += 1
            if response_time > 0:
                perf.response_times.append(response_time)

        self._state.add(value)
        return {
            "correct": correct,
            "response_time": response_time,
            "features_tracked": len(features),
        }

    def get_value(self) -> float:
        """Get overall success rate."""
        total = self._total_correct + self._total_incorrect
        if total == 0:
            return 0.0
        return self._total_correct / total

    def record_attempt(
        self,
        correct: bool,
        response_time: float,
        features: Dict[str, Any],
        dimensions: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record an attempt with its features.

        Args:
            correct: Whether the attempt was correct
            response_time: Time taken for the attempt
            features: Feature values for this attempt
            dimensions: Additional dimensions
            metadata: Additional metadata

        Returns:
            Collection result
        """
        value = {
            "correct": correct,
            "response_time": response_time,
            "features": features,
        }
        result = self.collect(value)

        if self._auto_record and self._engine:
            merged_dimensions = {**self._dimensions, **(dimensions or {})}
            self._engine.record(
                metric_name=self.metric_name,
                value=1 if correct else 0,
                dimensions={**merged_dimensions, **features},
                metadata={**(metadata or {}), "response_time": response_time},
            )

        return result

    @property
    def total_correct(self) -> int:
        """Get total correct count."""
        return self._total_correct

    @property
    def total_incorrect(self) -> int:
        """Get total incorrect count."""
        return self._total_incorrect

    @property
    def total_attempts(self) -> int:
        """Get total attempts."""
        return self._total_correct + self._total_incorrect

    @property
    def success_rate(self) -> float:
        """Get overall success rate."""
        return self.get_value()

    @property
    def avg_response_time(self) -> float:
        """Get average response time."""
        if not self._all_response_times:
            return 0.0
        return sum(self._all_response_times) / len(self._all_response_times)

    def get_feature_stats(self, feature_name: str) -> Dict[str, FeaturePerformance]:
        """Get stats for a specific feature."""
        return self._performance.get(feature_name, {})

    def get_all_feature_stats(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get stats for all features."""
        result = {}
        for feature_name, values in self._performance.items():
            result[feature_name] = {
                value_key: perf.to_dict()
                for value_key, perf in values.items()
            }
        return result

    def get_weak_areas(
        self,
        min_attempts: int = 3,
        max_success_rate: float = 0.5,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Identify weak areas based on success rate.

        Args:
            min_attempts: Minimum attempts to consider
            max_success_rate: Maximum success rate to be considered weak
            limit: Maximum number of weak areas to return

        Returns:
            List of weak areas sorted by success rate
        """
        weak_areas = []

        for feature_name, values in self._performance.items():
            for value_key, perf in values.items():
                if (perf.total_attempts >= min_attempts and
                    perf.success_rate < max_success_rate):
                    weak_areas.append({
                        "feature_name": feature_name,
                        "feature_value": perf.feature_value,
                        "success_rate": perf.success_rate,
                        "total_attempts": perf.total_attempts,
                        "avg_response_time": perf.avg_response_time,
                    })

        # Sort by success rate (ascending)
        weak_areas.sort(key=lambda x: x["success_rate"])
        return weak_areas[:limit]

    def get_strong_areas(
        self,
        min_attempts: int = 3,
        min_success_rate: float = 0.8,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Identify strong areas based on success rate.

        Args:
            min_attempts: Minimum attempts to consider
            min_success_rate: Minimum success rate to be considered strong
            limit: Maximum number of strong areas to return

        Returns:
            List of strong areas sorted by success rate (descending)
        """
        strong_areas = []

        for feature_name, values in self._performance.items():
            for value_key, perf in values.items():
                if (perf.total_attempts >= min_attempts and
                    perf.success_rate >= min_success_rate):
                    strong_areas.append({
                        "feature_name": feature_name,
                        "feature_value": perf.feature_value,
                        "success_rate": perf.success_rate,
                        "total_attempts": perf.total_attempts,
                        "avg_response_time": perf.avg_response_time,
                    })

        # Sort by success rate (descending)
        strong_areas.sort(key=lambda x: x["success_rate"], reverse=True)
        return strong_areas[:limit]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all performance data."""
        return {
            "total_correct": self._total_correct,
            "total_incorrect": self._total_incorrect,
            "total_attempts": self.total_attempts,
            "success_rate": self.success_rate,
            "avg_response_time": self.avg_response_time,
            "features_tracked": list(self._performance.keys()),
            "weak_areas": self.get_weak_areas(),
            "strong_areas": self.get_strong_areas(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_correct": self._total_correct,
            "total_incorrect": self._total_incorrect,
            "response_times": self._all_response_times,
            "performance": self.get_all_feature_stats(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        metric_name: str = "composite",
        engine: Optional["MetricsEngine"] = None,
    ) -> "CompositeCollector":
        """Create from dictionary."""
        collector = cls(metric_name=metric_name, engine=engine)
        collector._total_correct = data.get("total_correct", 0)
        collector._total_incorrect = data.get("total_incorrect", 0)
        collector._all_response_times = data.get("response_times", [])

        # Restore performance data
        for feature_name, values in data.get("performance", {}).items():
            collector._performance[feature_name] = {}
            for value_key, perf_data in values.items():
                perf = FeaturePerformance(
                    feature_name=feature_name,
                    feature_value=perf_data.get("feature_value", value_key),
                    correct=perf_data.get("correct", 0),
                    incorrect=perf_data.get("incorrect", 0),
                    response_times=perf_data.get("response_times", []),
                )
                collector._performance[feature_name][value_key] = perf

        return collector

    def reset(self) -> None:
        """Reset all collected data."""
        super().reset()
        self._performance.clear()
        self._total_correct = 0
        self._total_incorrect = 0
        self._all_response_times.clear()
