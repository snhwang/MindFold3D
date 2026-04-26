"""Histogram collector for distribution analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import MetricCollector
from ..utils.stats import (
    calculate_mean,
    calculate_median,
    calculate_std,
    calculate_percentile,
)

if TYPE_CHECKING:
    from ..core.engine import MetricsEngine


@dataclass
class HistogramBucket:
    """A histogram bucket."""
    lower_bound: float
    upper_bound: float
    count: int = 0
    sum_values: float = 0.0

    @property
    def mean(self) -> float:
        """Get the mean of values in this bucket."""
        if self.count == 0:
            return 0.0
        return self.sum_values / self.count

    def contains(self, value: float) -> bool:
        """Check if a value belongs in this bucket."""
        return self.lower_bound <= value < self.upper_bound


class HistogramCollector(MetricCollector):
    """
    Collector for histogram metrics (distribution of values).

    Tracks the distribution of values across configurable buckets,
    useful for analyzing response times, sizes, etc.
    """

    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(
        self,
        metric_name: str,
        buckets: Optional[List[float]] = None,
        engine: Optional["MetricsEngine"] = None,
        dimensions: Optional[Dict[str, Any]] = None,
        auto_record: bool = True,
    ):
        super().__init__(metric_name, engine, dimensions, auto_record)

        # Set up buckets
        bucket_boundaries = sorted(buckets or self.DEFAULT_BUCKETS)

        self._buckets: List[HistogramBucket] = []
        prev_bound = float("-inf")
        for bound in bucket_boundaries:
            self._buckets.append(HistogramBucket(prev_bound, bound))
            prev_bound = bound
        # Add final bucket for overflow
        self._buckets.append(HistogramBucket(prev_bound, float("inf")))

        self._values: List[float] = []
        self._sum = 0.0
        self._count = 0

    def collect(self, value: Any, **kwargs) -> float:
        """
        Observe a value.

        Args:
            value: The value to observe
            **kwargs: Additional parameters

        Returns:
            The observed value
        """
        float_value = float(value)
        self._values.append(float_value)
        self._sum += float_value
        self._count += 1

        # Add to appropriate bucket
        for bucket in self._buckets:
            if bucket.contains(float_value):
                bucket.count += 1
                bucket.sum_values += float_value
                break

        self._state.add(float_value)
        return float_value

    def get_value(self) -> float:
        """Get the mean of all observed values."""
        if self._count == 0:
            return 0.0
        return self._sum / self._count

    def observe(self, value: float) -> float:
        """Observe a value (alias for collect)."""
        return self.collect(value)

    @property
    def count(self) -> int:
        """Get the total count of observations."""
        return self._count

    @property
    def sum(self) -> float:
        """Get the sum of all observations."""
        return self._sum

    @property
    def mean(self) -> float:
        """Get the mean of all observations."""
        return self.get_value()

    @property
    def median(self) -> float:
        """Get the median of all observations."""
        return calculate_median(self._values)

    @property
    def std(self) -> float:
        """Get the standard deviation."""
        return calculate_std(self._values)

    @property
    def min(self) -> float:
        """Get the minimum value."""
        return min(self._values) if self._values else 0.0

    @property
    def max(self) -> float:
        """Get the maximum value."""
        return max(self._values) if self._values else 0.0

    def percentile(self, p: float) -> float:
        """Get a specific percentile (0-100)."""
        return calculate_percentile(self._values, p)

    def get_buckets(self) -> List[Dict[str, Any]]:
        """Get bucket information."""
        return [
            {
                "lower_bound": b.lower_bound,
                "upper_bound": b.upper_bound,
                "count": b.count,
                "mean": b.mean,
            }
            for b in self._buckets
        ]

    def get_distribution(self) -> Dict[str, Any]:
        """Get the full distribution statistics."""
        return {
            "count": self._count,
            "sum": self._sum,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "p50": self.percentile(50),
            "p90": self.percentile(90),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
            "buckets": self.get_buckets(),
        }

    def reset(self) -> None:
        """Reset the histogram."""
        super().reset()
        self._values.clear()
        self._sum = 0.0
        self._count = 0
        for bucket in self._buckets:
            bucket.count = 0
            bucket.sum_values = 0.0


class CategoricalHistogram(MetricCollector):
    """
    Histogram for categorical values.

    Tracks the distribution of categorical values,
    useful for tracking error types, feature values, etc.
    """

    def __init__(
        self,
        metric_name: str,
        categories: Optional[List[str]] = None,
        engine: Optional["MetricsEngine"] = None,
        dimensions: Optional[Dict[str, Any]] = None,
        auto_record: bool = True,
    ):
        super().__init__(metric_name, engine, dimensions, auto_record)
        self._categories = set(categories) if categories else set()
        self._counts: Dict[str, int] = {}
        self._total = 0

    def collect(self, value: Any, **kwargs) -> str:
        """
        Observe a categorical value.

        Args:
            value: The category value
            **kwargs: Additional parameters

        Returns:
            The observed category
        """
        category = str(value)

        # Validate if categories are restricted
        if self._categories and category not in self._categories:
            raise ValueError(f"Unknown category: {category}")

        self._counts[category] = self._counts.get(category, 0) + 1
        self._total += 1
        self._state.add(category)
        return category

    def get_value(self) -> Dict[str, int]:
        """Get the category counts."""
        return self._counts.copy()

    @property
    def total(self) -> int:
        """Get the total count."""
        return self._total

    def get_count(self, category: str) -> int:
        """Get the count for a specific category."""
        return self._counts.get(category, 0)

    def get_proportion(self, category: str) -> float:
        """Get the proportion for a specific category."""
        if self._total == 0:
            return 0.0
        return self._counts.get(category, 0) / self._total

    def get_distribution(self) -> Dict[str, Any]:
        """Get the full distribution."""
        return {
            "total": self._total,
            "counts": self._counts.copy(),
            "proportions": {
                k: v / self._total if self._total > 0 else 0
                for k, v in self._counts.items()
            },
            "mode": max(self._counts, key=self._counts.get) if self._counts else None,
        }

    def reset(self) -> None:
        """Reset the histogram."""
        super().reset()
        self._counts.clear()
        self._total = 0
