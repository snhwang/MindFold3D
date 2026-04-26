"""Rolling aggregator for moving window analysis."""

from collections import deque
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union

from ..core.definitions import AggregationType
from ..core.datapoint import DataPoint
from ..utils.stats import (
    calculate_mean,
    calculate_median,
    calculate_std,
    calculate_moving_average,
    calculate_exponential_moving_average,
)
from ..utils.time import parse_interval


class RollingAggregator:
    """
    Aggregator for rolling/moving window analysis.

    Supports fixed-size windows and time-based windows
    for real-time metric analysis.
    """

    def __init__(
        self,
        window_size: int = 10,
        aggregation: Union[AggregationType, str] = AggregationType.MEAN,
    ):
        """
        Initialize the rolling aggregator.

        Args:
            window_size: Number of values in the rolling window
            aggregation: Default aggregation method
        """
        if isinstance(aggregation, str):
            aggregation = AggregationType(aggregation)
        self._window_size = window_size
        self._aggregation = aggregation
        self._values: deque = deque(maxlen=window_size)
        self._timestamps: deque = deque(maxlen=window_size)

    def add(self, value: float, timestamp: Optional[datetime] = None) -> float:
        """
        Add a value to the rolling window.

        Args:
            value: Value to add
            timestamp: Optional timestamp

        Returns:
            Current rolling aggregate
        """
        self._values.append(value)
        self._timestamps.append(timestamp or datetime.utcnow())
        return self.get_value()

    def get_value(self, aggregation: Optional[AggregationType] = None) -> float:
        """Get the current rolling aggregate."""
        if not self._values:
            return 0.0

        values = list(self._values)
        agg = aggregation or self._aggregation

        if agg == AggregationType.SUM:
            return sum(values)
        elif agg == AggregationType.MEAN:
            return calculate_mean(values)
        elif agg == AggregationType.MEDIAN:
            return calculate_median(values)
        elif agg == AggregationType.MIN:
            return min(values)
        elif agg == AggregationType.MAX:
            return max(values)
        elif agg == AggregationType.COUNT:
            return float(len(values))
        elif agg == AggregationType.STD:
            return calculate_std(values)
        else:
            return calculate_mean(values)

    @property
    def values(self) -> List[float]:
        """Get current window values."""
        return list(self._values)

    @property
    def is_full(self) -> bool:
        """Check if window is full."""
        return len(self._values) == self._window_size

    @property
    def count(self) -> int:
        """Get current count of values."""
        return len(self._values)

    def clear(self) -> None:
        """Clear the rolling window."""
        self._values.clear()
        self._timestamps.clear()


class TimeWindowAggregator:
    """
    Aggregator for time-based rolling windows.

    Maintains values within a time window rather than a count-based window.
    """

    def __init__(
        self,
        window_duration: Union[str, timedelta] = "1h",
        aggregation: Union[AggregationType, str] = AggregationType.MEAN,
    ):
        """
        Initialize the time window aggregator.

        Args:
            window_duration: Duration of the window (e.g., "1h", "30m")
            aggregation: Default aggregation method
        """
        if isinstance(window_duration, str):
            window_duration = parse_interval(window_duration)
        if isinstance(aggregation, str):
            aggregation = AggregationType(aggregation)

        self._window_duration = window_duration
        self._aggregation = aggregation
        self._entries: List[tuple[datetime, float]] = []

    def add(self, value: float, timestamp: Optional[datetime] = None) -> float:
        """
        Add a value to the window.

        Args:
            value: Value to add
            timestamp: Timestamp (defaults to now)

        Returns:
            Current aggregate
        """
        ts = timestamp or datetime.utcnow()
        self._entries.append((ts, value))
        self._cleanup()
        return self.get_value()

    def _cleanup(self) -> None:
        """Remove expired entries."""
        cutoff = datetime.utcnow() - self._window_duration
        self._entries = [(ts, v) for ts, v in self._entries if ts > cutoff]

    def get_value(self, aggregation: Optional[AggregationType] = None) -> float:
        """Get the current aggregate."""
        self._cleanup()

        if not self._entries:
            return 0.0

        values = [v for _, v in self._entries]
        agg = aggregation or self._aggregation

        if agg == AggregationType.SUM:
            return sum(values)
        elif agg == AggregationType.MEAN:
            return calculate_mean(values)
        elif agg == AggregationType.MEDIAN:
            return calculate_median(values)
        elif agg == AggregationType.MIN:
            return min(values)
        elif agg == AggregationType.MAX:
            return max(values)
        elif agg == AggregationType.COUNT:
            return float(len(values))
        elif agg == AggregationType.STD:
            return calculate_std(values)
        else:
            return calculate_mean(values)

    @property
    def values(self) -> List[float]:
        """Get current window values."""
        self._cleanup()
        return [v for _, v in self._entries]

    @property
    def count(self) -> int:
        """Get current count of values."""
        self._cleanup()
        return len(self._entries)

    def clear(self) -> None:
        """Clear the window."""
        self._entries.clear()


class MovingAverageCalculator:
    """Calculator for various moving average types."""

    @staticmethod
    def simple_moving_average(
        datapoints: List[DataPoint],
        window: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Calculate simple moving average.

        Args:
            datapoints: DataPoints to analyze
            window: Window size

        Returns:
            List of {timestamp, value, sma} dicts
        """
        if not datapoints:
            return []

        # Sort by timestamp
        sorted_dps = sorted(datapoints, key=lambda dp: dp.timestamp)
        values = [dp.value for dp in sorted_dps]
        sma_values = calculate_moving_average(values, window)

        return [
            {
                "timestamp": dp.timestamp.isoformat(),
                "value": dp.value,
                "sma": sma,
            }
            for dp, sma in zip(sorted_dps, sma_values)
        ]

    @staticmethod
    def exponential_moving_average(
        datapoints: List[DataPoint],
        alpha: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Calculate exponential moving average.

        Args:
            datapoints: DataPoints to analyze
            alpha: Smoothing factor (0-1)

        Returns:
            List of {timestamp, value, ema} dicts
        """
        if not datapoints:
            return []

        sorted_dps = sorted(datapoints, key=lambda dp: dp.timestamp)
        values = [dp.value for dp in sorted_dps]
        ema_values = calculate_exponential_moving_average(values, alpha)

        return [
            {
                "timestamp": dp.timestamp.isoformat(),
                "value": dp.value,
                "ema": ema,
            }
            for dp, ema in zip(sorted_dps, ema_values)
        ]

    @staticmethod
    def weighted_moving_average(
        datapoints: List[DataPoint],
        weights: Optional[List[float]] = None,
        window: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Calculate weighted moving average.

        Args:
            datapoints: DataPoints to analyze
            weights: Custom weights (defaults to linear increasing)
            window: Window size if weights not specified

        Returns:
            List of {timestamp, value, wma} dicts
        """
        if not datapoints:
            return []

        sorted_dps = sorted(datapoints, key=lambda dp: dp.timestamp)
        values = [dp.value for dp in sorted_dps]

        # Default weights: linear increasing
        if weights is None:
            weights = list(range(1, window + 1))

        weight_sum = sum(weights)
        result = []

        for i, dp in enumerate(sorted_dps):
            if i < len(weights) - 1:
                wma = dp.value  # Not enough data
            else:
                window_values = values[i - len(weights) + 1:i + 1]
                wma = sum(v * w for v, w in zip(window_values, weights)) / weight_sum

            result.append({
                "timestamp": dp.timestamp.isoformat(),
                "value": dp.value,
                "wma": wma,
            })

        return result
