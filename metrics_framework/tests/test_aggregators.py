"""Tests for aggregators."""

import pytest
from datetime import datetime, timedelta
from metrics_framework.core.datapoint import DataPoint
from metrics_framework.aggregators.temporal import TemporalAggregator
from metrics_framework.aggregators.dimensional import DimensionalAggregator
from metrics_framework.aggregators.rolling import RollingAggregator


def create_datapoint(value, **kwargs):
    """Helper to create a datapoint."""
    return DataPoint(
        metric_name=kwargs.get("metric_name", "test"),
        value=value,
        timestamp=kwargs.get("timestamp", datetime.utcnow()),
        dimensions=kwargs.get("dimensions", {}),
    )


class TestTemporalAggregator:
    """Test TemporalAggregator functionality."""

    def test_aggregate_by_hour(self):
        """Test aggregation by hour."""
        aggregator = TemporalAggregator()

        base_time = datetime(2024, 6, 15, 10, 0, 0)
        datapoints = [
            create_datapoint(1.0, timestamp=base_time + timedelta(minutes=10)),
            create_datapoint(2.0, timestamp=base_time + timedelta(minutes=20)),
            create_datapoint(3.0, timestamp=base_time + timedelta(minutes=30)),
            create_datapoint(4.0, timestamp=base_time + timedelta(hours=1, minutes=10)),
            create_datapoint(5.0, timestamp=base_time + timedelta(hours=1, minutes=20)),
        ]

        result = aggregator.aggregate(datapoints, interval="hour", function="mean")

        assert len(result) == 2
        # First hour: mean of 1, 2, 3 = 2.0
        # Second hour: mean of 4, 5 = 4.5

    def test_aggregate_by_day(self):
        """Test aggregation by day."""
        aggregator = TemporalAggregator()

        datapoints = [
            create_datapoint(1.0, timestamp=datetime(2024, 6, 15, 10, 0)),
            create_datapoint(2.0, timestamp=datetime(2024, 6, 15, 14, 0)),
            create_datapoint(3.0, timestamp=datetime(2024, 6, 16, 10, 0)),
        ]

        result = aggregator.aggregate(datapoints, interval="day", function="sum")

        assert len(result) == 2


class TestDimensionalAggregator:
    """Test DimensionalAggregator functionality."""

    def test_aggregate_by_dimension(self):
        """Test aggregation by a single dimension."""
        aggregator = DimensionalAggregator()

        datapoints = [
            create_datapoint(1.0, dimensions={"level": "easy"}),
            create_datapoint(1.0, dimensions={"level": "easy"}),
            create_datapoint(0.0, dimensions={"level": "hard"}),
            create_datapoint(1.0, dimensions={"level": "hard"}),
        ]

        result = aggregator.aggregate(datapoints, group_by=["level"], function="mean")

        assert "easy" in result
        assert "hard" in result
        assert result["easy"] == 1.0
        assert result["hard"] == 0.5

    def test_aggregate_multiple_dimensions(self):
        """Test aggregation by multiple dimensions."""
        aggregator = DimensionalAggregator()

        datapoints = [
            create_datapoint(1.0, dimensions={"level": "easy", "category": "A"}),
            create_datapoint(0.0, dimensions={"level": "easy", "category": "A"}),
            create_datapoint(1.0, dimensions={"level": "easy", "category": "B"}),
            create_datapoint(1.0, dimensions={"level": "hard", "category": "A"}),
        ]

        result = aggregator.aggregate(
            datapoints,
            group_by=["level", "category"],
            function="mean"
        )

        assert ("easy", "A") in result or "easy|A" in str(result)


class TestRollingAggregator:
    """Test RollingAggregator functionality."""

    def test_rolling_window(self):
        """Test rolling window aggregation."""
        aggregator = RollingAggregator(window_size=3)

        datapoints = [
            create_datapoint(1.0),
            create_datapoint(2.0),
            create_datapoint(3.0),
            create_datapoint(4.0),
            create_datapoint(5.0),
        ]

        result = aggregator.aggregate(datapoints, function="mean")

        # Rolling mean of window 3:
        # [1,2,3] -> 2.0
        # [2,3,4] -> 3.0
        # [3,4,5] -> 4.0
        assert len(result) == 3
        assert result[0].value == 2.0
        assert result[1].value == 3.0
        assert result[2].value == 4.0

    def test_rolling_sum(self):
        """Test rolling sum."""
        aggregator = RollingAggregator(window_size=2)

        datapoints = [
            create_datapoint(1.0),
            create_datapoint(2.0),
            create_datapoint(3.0),
        ]

        result = aggregator.aggregate(datapoints, function="sum")

        assert len(result) == 2
        assert result[0].value == 3.0  # 1 + 2
        assert result[1].value == 5.0  # 2 + 3
