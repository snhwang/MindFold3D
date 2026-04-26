"""Counter collector for monotonically increasing values."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING

from .base import MetricCollector

if TYPE_CHECKING:
    from ..core.engine import MetricsEngine


class CounterCollector(MetricCollector):
    """
    Collector for counter metrics (monotonically increasing).

    Counters track cumulative values like total requests, total errors, etc.
    They can only increase or be reset to zero.
    """

    def __init__(
        self,
        metric_name: str,
        engine: Optional["MetricsEngine"] = None,
        dimensions: Optional[Dict[str, Any]] = None,
        auto_record: bool = True,
        initial_value: float = 0,
    ):
        super().__init__(metric_name, engine, dimensions, auto_record)
        self._value = initial_value
        self._total_increments = 0

    def collect(self, value: Any = 1, **kwargs) -> float:
        """
        Increment the counter.

        Args:
            value: Amount to increment (default 1)
            **kwargs: Additional parameters

        Returns:
            The new counter value
        """
        if value < 0:
            raise ValueError("Counter can only be incremented with positive values")

        self._value += value
        self._total_increments += 1
        self._state.add(self._value)
        return self._value

    def get_value(self) -> float:
        """Get the current counter value."""
        return self._value

    def increment(self, amount: float = 1) -> float:
        """Increment the counter by a given amount."""
        return self.collect(amount)

    def inc(self) -> float:
        """Increment the counter by 1."""
        return self.collect(1)

    @property
    def total_increments(self) -> int:
        """Get the total number of increments."""
        return self._total_increments

    def reset(self) -> None:
        """Reset the counter to zero."""
        super().reset()
        self._value = 0
        self._total_increments = 0


class SuccessFailureCounter(MetricCollector):
    """
    Collector for tracking success/failure outcomes.

    Tracks correct/incorrect counts and calculates success rate.
    """

    def __init__(
        self,
        metric_name: str,
        engine: Optional["MetricsEngine"] = None,
        dimensions: Optional[Dict[str, Any]] = None,
        auto_record: bool = True,
    ):
        super().__init__(metric_name, engine, dimensions, auto_record)
        self._successes = 0
        self._failures = 0

    def collect(self, value: Any, **kwargs) -> float:
        """
        Record a success or failure.

        Args:
            value: True/1 for success, False/0 for failure
            **kwargs: Additional parameters

        Returns:
            The current success rate
        """
        is_success = bool(value)
        if is_success:
            self._successes += 1
        else:
            self._failures += 1

        self._state.add(is_success)
        return self.success_rate

    def get_value(self) -> float:
        """Get the current success rate."""
        return self.success_rate

    @property
    def successes(self) -> int:
        """Get the number of successes."""
        return self._successes

    @property
    def failures(self) -> int:
        """Get the number of failures."""
        return self._failures

    @property
    def total(self) -> int:
        """Get the total count."""
        return self._successes + self._failures

    @property
    def success_rate(self) -> float:
        """Get the success rate (0-1)."""
        if self.total == 0:
            return 0.0
        return self._successes / self.total

    @property
    def failure_rate(self) -> float:
        """Get the failure rate (0-1)."""
        if self.total == 0:
            return 0.0
        return self._failures / self.total

    def record_success(
        self,
        dimensions: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a success."""
        return self.record(True, dimensions, metadata)

    def record_failure(
        self,
        dimensions: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a failure."""
        return self.record(False, dimensions, metadata)

    def reset(self) -> None:
        """Reset all counts."""
        super().reset()
        self._successes = 0
        self._failures = 0
