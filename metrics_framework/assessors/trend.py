"""Trend-based performance assessor."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from ..core.datapoint import DataPoint
from ..aggregators.temporal import TemporalAggregator
from ..utils.stats import calculate_mean, calculate_std


class TrendDirection(str, Enum):
    """Trend direction indicators."""
    STRONGLY_IMPROVING = "strongly_improving"
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    STRONGLY_DECLINING = "strongly_declining"


@dataclass
class TrendResult:
    """Result of trend analysis."""
    metric_name: str
    direction: TrendDirection
    slope: float
    r_squared: float
    change_percent: float
    confidence: float
    period_start: datetime
    period_end: datetime
    data_points: int
    first_value: float
    last_value: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "direction": self.direction.value,
            "slope": self.slope,
            "r_squared": self.r_squared,
            "change_percent": self.change_percent,
            "confidence": self.confidence,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "data_points": self.data_points,
            "first_value": self.first_value,
            "last_value": self.last_value,
            "details": self.details,
        }

    @property
    def is_improving(self) -> bool:
        """Check if trend is improving."""
        return self.direction in [
            TrendDirection.STRONGLY_IMPROVING,
            TrendDirection.IMPROVING,
        ]

    @property
    def is_declining(self) -> bool:
        """Check if trend is declining."""
        return self.direction in [
            TrendDirection.STRONGLY_DECLINING,
            TrendDirection.DECLINING,
        ]


class TrendAssessor:
    """
    Assessor that analyzes metric trends over time.

    Uses linear regression to determine trend direction
    and calculates statistical significance.
    """

    def __init__(
        self,
        strong_threshold: float = 0.1,
        significance_threshold: float = 0.05,
        min_data_points: int = 5,
        higher_is_better: bool = True,
    ):
        """
        Initialize the trend assessor.

        Args:
            strong_threshold: Threshold for strong trend (slope magnitude)
            significance_threshold: P-value threshold for significance
            min_data_points: Minimum data points for trend analysis
            higher_is_better: Whether higher values indicate improvement
        """
        self._strong_threshold = strong_threshold
        self._significance_threshold = significance_threshold
        self._min_data_points = min_data_points
        self._higher_is_better = higher_is_better

    def analyze_trend(
        self,
        datapoints: List[DataPoint],
        interval: str = "day",
    ) -> TrendResult:
        """
        Analyze trend for a set of datapoints.

        Args:
            datapoints: DataPoints to analyze
            interval: Time interval for aggregation

        Returns:
            Trend analysis result
        """
        if not datapoints:
            return self._empty_result("unknown")

        metric_name = datapoints[0].metric_name

        if len(datapoints) < self._min_data_points:
            return self._insufficient_data_result(metric_name, len(datapoints))

        # Sort by timestamp
        sorted_dps = sorted(datapoints, key=lambda dp: dp.timestamp)

        # Aggregate by interval
        aggregator = TemporalAggregator()
        aggregated = aggregator.aggregate_by_interval(sorted_dps, interval)

        if len(aggregated) < 2:
            return self._insufficient_data_result(metric_name, len(datapoints))

        # Extract values and calculate linear regression
        values = [a.value for a in aggregated]
        n = len(values)
        x = list(range(n))

        # Calculate means
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        # Calculate slope and intercept
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator

        intercept = y_mean - slope * x_mean

        # Calculate R-squared
        y_pred = [slope * xi + intercept for xi in x]
        ss_res = sum((values[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((values[i] - y_mean) ** 2 for i in range(n))

        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Calculate change percentage
        first_value = values[0]
        last_value = values[-1]
        change_percent = 0.0
        if first_value != 0:
            change_percent = ((last_value - first_value) / abs(first_value)) * 100

        # Determine trend direction
        normalized_slope = slope
        if y_mean != 0:
            normalized_slope = slope / abs(y_mean)

        # Adjust for higher_is_better
        effective_slope = normalized_slope if self._higher_is_better else -normalized_slope

        if effective_slope > self._strong_threshold:
            direction = TrendDirection.STRONGLY_IMPROVING
        elif effective_slope > 0.01:
            direction = TrendDirection.IMPROVING
        elif effective_slope < -self._strong_threshold:
            direction = TrendDirection.STRONGLY_DECLINING
        elif effective_slope < -0.01:
            direction = TrendDirection.DECLINING
        else:
            direction = TrendDirection.STABLE

        # Calculate confidence based on R-squared and sample size
        confidence = min(1.0, r_squared * (1 - 1 / n))

        return TrendResult(
            metric_name=metric_name,
            direction=direction,
            slope=slope,
            r_squared=r_squared,
            change_percent=change_percent,
            confidence=confidence,
            period_start=sorted_dps[0].timestamp,
            period_end=sorted_dps[-1].timestamp,
            data_points=n,
            first_value=first_value,
            last_value=last_value,
            details={
                "interval": interval,
                "intercept": intercept,
                "normalized_slope": normalized_slope,
                "values": values,
                "higher_is_better": self._higher_is_better,
            },
        )

    def compare_periods(
        self,
        datapoints: List[DataPoint],
        period1_start: datetime,
        period1_end: datetime,
        period2_start: datetime,
        period2_end: datetime,
    ) -> Dict[str, Any]:
        """
        Compare metrics between two time periods.

        Args:
            datapoints: DataPoints to analyze
            period1_start: Start of first period
            period1_end: End of first period
            period2_start: Start of second period
            period2_end: End of second period

        Returns:
            Comparison results
        """
        # Filter datapoints by period
        period1_dps = [
            dp for dp in datapoints
            if period1_start <= dp.timestamp <= period1_end
        ]
        period2_dps = [
            dp for dp in datapoints
            if period2_start <= dp.timestamp <= period2_end
        ]

        if not period1_dps or not period2_dps:
            return {
                "error": "Insufficient data in one or both periods",
                "period1_count": len(period1_dps),
                "period2_count": len(period2_dps),
            }

        # Calculate stats for each period
        period1_values = [dp.value for dp in period1_dps]
        period2_values = [dp.value for dp in period2_dps]

        period1_mean = calculate_mean(period1_values)
        period2_mean = calculate_mean(period2_values)

        period1_std = calculate_std(period1_values)
        period2_std = calculate_std(period2_values)

        # Calculate change
        absolute_change = period2_mean - period1_mean
        percent_change = 0.0
        if period1_mean != 0:
            percent_change = (absolute_change / abs(period1_mean)) * 100

        # Determine if change is significant (simple approach)
        pooled_std = ((period1_std ** 2 + period2_std ** 2) / 2) ** 0.5
        effect_size = abs(absolute_change) / pooled_std if pooled_std > 0 else 0

        is_significant = effect_size > 0.5  # Cohen's d threshold

        # Determine if improvement or decline
        if self._higher_is_better:
            is_improvement = absolute_change > 0
        else:
            is_improvement = absolute_change < 0

        return {
            "period1": {
                "start": period1_start.isoformat(),
                "end": period1_end.isoformat(),
                "count": len(period1_dps),
                "mean": period1_mean,
                "std": period1_std,
            },
            "period2": {
                "start": period2_start.isoformat(),
                "end": period2_end.isoformat(),
                "count": len(period2_dps),
                "mean": period2_mean,
                "std": period2_std,
            },
            "change": {
                "absolute": absolute_change,
                "percent": percent_change,
                "effect_size": effect_size,
                "is_significant": is_significant,
                "is_improvement": is_improvement,
            },
        }

    def _empty_result(self, metric_name: str) -> TrendResult:
        """Create an empty result."""
        now = datetime.utcnow()
        return TrendResult(
            metric_name=metric_name,
            direction=TrendDirection.STABLE,
            slope=0.0,
            r_squared=0.0,
            change_percent=0.0,
            confidence=0.0,
            period_start=now,
            period_end=now,
            data_points=0,
            first_value=0.0,
            last_value=0.0,
            details={"error": "No datapoints provided"},
        )

    def _insufficient_data_result(
        self,
        metric_name: str,
        count: int,
    ) -> TrendResult:
        """Create a result for insufficient data."""
        now = datetime.utcnow()
        return TrendResult(
            metric_name=metric_name,
            direction=TrendDirection.STABLE,
            slope=0.0,
            r_squared=0.0,
            change_percent=0.0,
            confidence=0.0,
            period_start=now,
            period_end=now,
            data_points=count,
            first_value=0.0,
            last_value=0.0,
            details={
                "error": f"Insufficient data points ({count} < {self._min_data_points})"
            },
        )
