"""Timer collector for time-based measurements."""

import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .base import MetricCollector
from .histogram import HistogramCollector
from ..utils.stats import (
    calculate_mean,
    calculate_median,
    calculate_std,
    calculate_percentile,
)

if TYPE_CHECKING:
    from ..core.engine import MetricsEngine


class TimerCollector(MetricCollector):
    """
    Collector for timing metrics.

    Tracks duration measurements with support for:
    - Manual start/stop timing
    - Context manager timing
    - Decorator timing
    """

    def __init__(
        self,
        metric_name: str,
        engine: Optional["MetricsEngine"] = None,
        dimensions: Optional[Dict[str, Any]] = None,
        auto_record: bool = True,
        unit: str = "seconds",
    ):
        super().__init__(metric_name, engine, dimensions, auto_record)
        self._unit = unit
        self._start_time: Optional[float] = None
        self._durations: List[float] = []
        self._total_duration = 0.0

    def collect(self, value: Any, **kwargs) -> float:
        """
        Record a duration value.

        Args:
            value: The duration value
            **kwargs: Additional parameters

        Returns:
            The recorded duration
        """
        duration = float(value)
        self._durations.append(duration)
        self._total_duration += duration
        self._state.add(duration)
        return duration

    def get_value(self) -> float:
        """Get the mean duration."""
        if not self._durations:
            return 0.0
        return self._total_duration / len(self._durations)

    def start(self) -> "TimerCollector":
        """Start the timer."""
        self._start_time = time.perf_counter()
        return self

    def stop(
        self,
        dimensions: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Stop the timer and record the duration.

        Returns:
            The measured duration
        """
        if self._start_time is None:
            raise RuntimeError("Timer was not started")

        duration = time.perf_counter() - self._start_time
        self._start_time = None

        # Collect the value
        self.collect(duration)

        # Record to engine if configured
        if self._auto_record and self._engine:
            merged_dimensions = {**self._dimensions, **(dimensions or {})}
            self._engine.record(
                metric_name=self.metric_name,
                value=duration,
                dimensions=merged_dimensions,
                metadata=metadata,
            )

        return duration

    def observe(
        self,
        duration: float,
        dimensions: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Observe a duration value directly.

        Args:
            duration: The duration to record
            dimensions: Additional dimensions
            metadata: Additional metadata

        Returns:
            The recorded duration
        """
        return self.record(duration, dimensions, metadata)

    @property
    def count(self) -> int:
        """Get the number of recorded durations."""
        return len(self._durations)

    @property
    def total(self) -> float:
        """Get the total duration."""
        return self._total_duration

    @property
    def mean(self) -> float:
        """Get the mean duration."""
        return self.get_value()

    @property
    def median(self) -> float:
        """Get the median duration."""
        return calculate_median(self._durations)

    @property
    def std(self) -> float:
        """Get the standard deviation."""
        return calculate_std(self._durations)

    @property
    def min(self) -> float:
        """Get the minimum duration."""
        return min(self._durations) if self._durations else 0.0

    @property
    def max(self) -> float:
        """Get the maximum duration."""
        return max(self._durations) if self._durations else 0.0

    def percentile(self, p: float) -> float:
        """Get a specific percentile (0-100)."""
        return calculate_percentile(self._durations, p)

    def get_stats(self) -> Dict[str, Any]:
        """Get timing statistics."""
        return {
            "count": self.count,
            "total": self.total,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "p50": self.percentile(50),
            "p90": self.percentile(90),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
            "unit": self._unit,
        }

    @contextmanager
    def time(
        self,
        dimensions: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for timing a block of code.

        Usage:
            with timer.time():
                # code to time
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.collect(duration)
            if self._auto_record and self._engine:
                merged_dimensions = {**self._dimensions, **(dimensions or {})}
                self._engine.record(
                    metric_name=self.metric_name,
                    value=duration,
                    dimensions=merged_dimensions,
                    metadata=metadata,
                )

    def timed(
        self,
        dimensions: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """
        Decorator for timing a function.

        Usage:
            @timer.timed()
            def my_function():
                # code to time
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.time(dimensions, metadata):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def reset(self) -> None:
        """Reset the timer."""
        super().reset()
        self._start_time = None
        self._durations.clear()
        self._total_duration = 0.0


class ResponseTimeTracker(TimerCollector):
    """
    Specialized timer for tracking response times.

    Includes additional features for response time analysis:
    - Automatic response time categorization (fast, normal, slow)
    - SLA tracking
    """

    def __init__(
        self,
        metric_name: str,
        fast_threshold: float = 1.0,
        slow_threshold: float = 5.0,
        sla_target: Optional[float] = None,
        engine: Optional["MetricsEngine"] = None,
        dimensions: Optional[Dict[str, Any]] = None,
        auto_record: bool = True,
    ):
        super().__init__(metric_name, engine, dimensions, auto_record, "seconds")
        self._fast_threshold = fast_threshold
        self._slow_threshold = slow_threshold
        self._sla_target = sla_target
        self._fast_count = 0
        self._normal_count = 0
        self._slow_count = 0

    def collect(self, value: Any, **kwargs) -> float:
        """Record a response time and categorize it."""
        duration = float(value)
        super().collect(duration, **kwargs)

        # Categorize
        if duration <= self._fast_threshold:
            self._fast_count += 1
        elif duration <= self._slow_threshold:
            self._normal_count += 1
        else:
            self._slow_count += 1

        return duration

    @property
    def fast_count(self) -> int:
        """Get count of fast responses."""
        return self._fast_count

    @property
    def normal_count(self) -> int:
        """Get count of normal responses."""
        return self._normal_count

    @property
    def slow_count(self) -> int:
        """Get count of slow responses."""
        return self._slow_count

    @property
    def fast_rate(self) -> float:
        """Get the rate of fast responses."""
        if self.count == 0:
            return 0.0
        return self._fast_count / self.count

    @property
    def slow_rate(self) -> float:
        """Get the rate of slow responses."""
        if self.count == 0:
            return 0.0
        return self._slow_count / self.count

    @property
    def sla_compliance(self) -> Optional[float]:
        """Get SLA compliance rate."""
        if self._sla_target is None or self.count == 0:
            return None
        compliant = sum(1 for d in self._durations if d <= self._sla_target)
        return compliant / self.count

    def categorize(self, duration: float) -> str:
        """Categorize a duration."""
        if duration <= self._fast_threshold:
            return "fast"
        elif duration <= self._slow_threshold:
            return "normal"
        else:
            return "slow"

    def get_stats(self) -> Dict[str, Any]:
        """Get response time statistics."""
        stats = super().get_stats()
        stats.update({
            "fast_threshold": self._fast_threshold,
            "slow_threshold": self._slow_threshold,
            "fast_count": self._fast_count,
            "normal_count": self._normal_count,
            "slow_count": self._slow_count,
            "fast_rate": self.fast_rate,
            "slow_rate": self.slow_rate,
            "sla_target": self._sla_target,
            "sla_compliance": self.sla_compliance,
        })
        return stats

    def reset(self) -> None:
        """Reset the tracker."""
        super().reset()
        self._fast_count = 0
        self._normal_count = 0
        self._slow_count = 0
