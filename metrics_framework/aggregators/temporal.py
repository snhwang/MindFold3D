"""Temporal aggregator for time-based analysis."""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from ..core.definitions import AggregationType
from ..core.datapoint import DataPoint, AggregatedDataPoint
from ..utils.stats import (
    calculate_mean,
    calculate_median,
    calculate_std,
    calculate_percentile,
    calculate_variance,
    calculate_mode,
)
from ..utils.time import get_time_bucket, parse_interval


class TemporalAggregator:
    """
    Aggregator for time-based metric analysis.

    Supports aggregating metrics over time periods and calculating
    time-series data.
    """

    def __init__(
        self,
        aggregation: Union[AggregationType, str] = AggregationType.MEAN,
        percentile_value: float = 95,
    ):
        """
        Initialize the temporal aggregator.

        Args:
            aggregation: Default aggregation method
            percentile_value: Percentile value for PERCENTILE aggregation
        """
        if isinstance(aggregation, str):
            aggregation = AggregationType(aggregation)
        self._aggregation = aggregation
        self._percentile_value = percentile_value

    def aggregate(
        self,
        values: List[float],
        aggregation: Optional[Union[AggregationType, str]] = None,
    ) -> float:
        """
        Aggregate a list of values.

        Args:
            values: Values to aggregate
            aggregation: Aggregation method (uses default if None)

        Returns:
            Aggregated value
        """
        if not values:
            return 0.0

        agg = aggregation or self._aggregation
        if isinstance(agg, str):
            agg = AggregationType(agg)

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
        elif agg == AggregationType.FIRST:
            return values[0]
        elif agg == AggregationType.LAST:
            return values[-1]
        elif agg == AggregationType.STD:
            return calculate_std(values)
        elif agg == AggregationType.VARIANCE:
            return calculate_variance(values)
        elif agg == AggregationType.PERCENTILE:
            return calculate_percentile(values, self._percentile_value)
        elif agg == AggregationType.MODE:
            mode = calculate_mode(values)
            return mode if mode is not None else calculate_mean(values)
        else:
            return calculate_mean(values)

    def aggregate_datapoints(
        self,
        datapoints: List[DataPoint],
        aggregation: Optional[Union[AggregationType, str]] = None,
    ) -> AggregatedDataPoint:
        """
        Aggregate a list of datapoints.

        Args:
            datapoints: DataPoints to aggregate
            aggregation: Aggregation method

        Returns:
            Aggregated datapoint
        """
        if not datapoints:
            return AggregatedDataPoint(
                metric_name="",
                aggregation_type=str(aggregation or self._aggregation),
                value=0.0,
                count=0,
                timestamp_from=datetime.utcnow(),
                timestamp_to=datetime.utcnow(),
            )

        values = [dp.value for dp in datapoints]
        timestamps = [dp.timestamp for dp in datapoints]

        return AggregatedDataPoint(
            metric_name=datapoints[0].metric_name,
            aggregation_type=str(aggregation or self._aggregation),
            value=self.aggregate(values, aggregation),
            count=len(values),
            timestamp_from=min(timestamps),
            timestamp_to=max(timestamps),
            raw_values=values,
        )

    def aggregate_by_interval(
        self,
        datapoints: List[DataPoint],
        interval: str = "hour",
        aggregation: Optional[Union[AggregationType, str]] = None,
    ) -> List[AggregatedDataPoint]:
        """
        Aggregate datapoints by time interval.

        Args:
            datapoints: DataPoints to aggregate
            interval: Time interval (minute, hour, day, week, month)
            aggregation: Aggregation method

        Returns:
            List of aggregated datapoints per interval
        """
        if not datapoints:
            return []

        # Group by time bucket
        buckets: Dict[datetime, List[DataPoint]] = defaultdict(list)
        for dp in datapoints:
            bucket = get_time_bucket(dp.timestamp, interval)
            buckets[bucket].append(dp)

        # Aggregate each bucket
        result = []
        for bucket_time in sorted(buckets.keys()):
            bucket_dps = buckets[bucket_time]
            values = [dp.value for dp in bucket_dps]

            # Calculate next bucket time
            delta = parse_interval(interval)
            next_bucket = bucket_time + delta

            result.append(AggregatedDataPoint(
                metric_name=bucket_dps[0].metric_name,
                aggregation_type=str(aggregation or self._aggregation),
                value=self.aggregate(values, aggregation),
                count=len(values),
                timestamp_from=bucket_time,
                timestamp_to=next_bucket,
                raw_values=values,
            ))

        return result

    def calculate_trend(
        self,
        datapoints: List[DataPoint],
        interval: str = "day",
    ) -> Dict[str, Any]:
        """
        Calculate trend over time.

        Args:
            datapoints: DataPoints to analyze
            interval: Time interval for trend calculation

        Returns:
            Trend analysis results
        """
        aggregated = self.aggregate_by_interval(datapoints, interval)

        if len(aggregated) < 2:
            return {
                "trend": "insufficient_data",
                "slope": 0.0,
                "change_percent": 0.0,
                "data_points": len(aggregated),
            }

        values = [a.value for a in aggregated]
        first_value = values[0]
        last_value = values[-1]

        # Simple linear regression for slope
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 0.0

        # Change percentage
        change_percent = 0.0
        if first_value != 0:
            change_percent = ((last_value - first_value) / abs(first_value)) * 100

        # Determine trend direction
        if abs(slope) < 0.01:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"

        return {
            "trend": trend,
            "slope": slope,
            "change_percent": change_percent,
            "first_value": first_value,
            "last_value": last_value,
            "data_points": n,
            "interval": interval,
            "values": values,
        }

    def get_time_series(
        self,
        datapoints: List[DataPoint],
        interval: str = "hour",
        fill_gaps: bool = True,
        fill_value: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Get time series data.

        Args:
            datapoints: DataPoints to convert
            interval: Time interval
            fill_gaps: Whether to fill gaps with fill_value
            fill_value: Value to use for gaps

        Returns:
            List of time series points
        """
        if not datapoints:
            return []

        aggregated = self.aggregate_by_interval(datapoints, interval)

        if not fill_gaps or len(aggregated) < 2:
            return [
                {
                    "timestamp": a.timestamp_from.isoformat(),
                    "value": a.value,
                    "count": a.count,
                }
                for a in aggregated
            ]

        # Fill gaps
        result = []
        delta = parse_interval(interval)
        start_time = aggregated[0].timestamp_from
        end_time = aggregated[-1].timestamp_from

        # Create lookup
        time_values = {a.timestamp_from: a for a in aggregated}

        current_time = start_time
        while current_time <= end_time:
            if current_time in time_values:
                a = time_values[current_time]
                result.append({
                    "timestamp": current_time.isoformat(),
                    "value": a.value,
                    "count": a.count,
                })
            else:
                result.append({
                    "timestamp": current_time.isoformat(),
                    "value": fill_value,
                    "count": 0,
                })
            current_time += delta

        return result
