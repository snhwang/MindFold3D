"""Gauge collector for point-in-time values."""

from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING

from .base import MetricCollector

if TYPE_CHECKING:
    from ..core.engine import MetricsEngine


class GaugeCollector(MetricCollector):
    """
    Collector for gauge metrics (point-in-time values).

    Gauges track values that can increase or decrease,
    like current memory usage, active connections, etc.
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
        self._min_value = initial_value
        self._max_value = initial_value

    def collect(self, value: Any, **kwargs) -> float:
        """
        Set the gauge value.

        Args:
            value: The new gauge value
            **kwargs: Additional parameters

        Returns:
            The new gauge value
        """
        self._value = float(value)
        self._min_value = min(self._min_value, self._value)
        self._max_value = max(self._max_value, self._value)
        self._state.add(self._value)
        return self._value

    def get_value(self) -> float:
        """Get the current gauge value."""
        return self._value

    def set(self, value: float) -> float:
        """Set the gauge to a specific value."""
        return self.collect(value)

    def increment(self, amount: float = 1) -> float:
        """Increment the gauge."""
        return self.collect(self._value + amount)

    def decrement(self, amount: float = 1) -> float:
        """Decrement the gauge."""
        return self.collect(self._value - amount)

    def inc(self) -> float:
        """Increment by 1."""
        return self.increment(1)

    def dec(self) -> float:
        """Decrement by 1."""
        return self.decrement(1)

    @property
    def min_value(self) -> float:
        """Get the minimum recorded value."""
        return self._min_value

    @property
    def max_value(self) -> float:
        """Get the maximum recorded value."""
        return self._max_value

    def reset(self, value: float = 0) -> None:
        """Reset the gauge."""
        super().reset()
        self._value = value
        self._min_value = value
        self._max_value = value


class BoundedGauge(GaugeCollector):
    """Gauge with min/max bounds."""

    def __init__(
        self,
        metric_name: str,
        min_bound: float = 0,
        max_bound: float = 100,
        engine: Optional["MetricsEngine"] = None,
        dimensions: Optional[Dict[str, Any]] = None,
        auto_record: bool = True,
        initial_value: float = 0,
    ):
        super().__init__(metric_name, engine, dimensions, auto_record, initial_value)
        self._min_bound = min_bound
        self._max_bound = max_bound

    def collect(self, value: Any, **kwargs) -> float:
        """Set the gauge value, clamping to bounds."""
        clamped = max(self._min_bound, min(self._max_bound, float(value)))
        return super().collect(clamped, **kwargs)

    @property
    def min_bound(self) -> float:
        """Get the minimum bound."""
        return self._min_bound

    @property
    def max_bound(self) -> float:
        """Get the maximum bound."""
        return self._max_bound

    @property
    def normalized_value(self) -> float:
        """Get the value normalized to 0-1."""
        range_size = self._max_bound - self._min_bound
        if range_size == 0:
            return 0.0
        return (self._value - self._min_bound) / range_size
