"""Tests for the MetricsEngine core functionality."""

import pytest
from metrics_framework import MetricsEngine, MetricDefinition
from metrics_framework.core.definitions import ThresholdConfig


class TestMetricsEngine:
    """Test suite for MetricsEngine."""

    def test_create_engine(self, engine):
        """Test engine creation."""
        assert engine is not None
        assert engine.metrics == {}

    def test_register_metric(self, engine):
        """Test metric registration."""
        metric = MetricDefinition(
            name="test_metric",
            metric_type="gauge",
            description="A test metric",
        )
        engine.register_metric(metric)

        assert "test_metric" in engine.metrics
        assert engine.metrics["test_metric"].name == "test_metric"

    def test_register_multiple_metrics(self, engine):
        """Test registering multiple metrics at once."""
        metrics = [
            MetricDefinition(name="metric_a", metric_type="gauge"),
            MetricDefinition(name="metric_b", metric_type="counter"),
            MetricDefinition(name="metric_c", metric_type="timer"),
        ]
        engine.register_metrics(metrics)

        assert len(engine.metrics) == 3
        assert all(f"metric_{x}" in engine.metrics for x in ["a", "b", "c"])

    def test_record_metric(self, engine_with_metrics):
        """Test recording a metric value."""
        result = engine_with_metrics.record("accuracy", 0.85)

        assert result is not None
        # Query the data back
        datapoints = engine_with_metrics.query(metric_name="accuracy")
        assert len(datapoints) == 1
        assert datapoints[0].value == 0.85

    def test_record_with_dimensions(self, engine_with_metrics):
        """Test recording with dimensional data."""
        engine_with_metrics.record(
            "accuracy", 0.9,
            dimensions={"level": "easy", "category": "A"}
        )
        engine_with_metrics.record(
            "accuracy", 0.7,
            dimensions={"level": "hard", "category": "B"}
        )

        datapoints = engine_with_metrics.query(metric_name="accuracy")
        assert len(datapoints) == 2

    def test_record_unregistered_metric_fails(self, engine):
        """Test that recording to unregistered metric fails."""
        with pytest.raises(ValueError):
            engine.record("nonexistent", 1.0)

    def test_query_by_metric_name(self, sample_data):
        """Test querying by metric name."""
        accuracy_data = sample_data.query(metric_name="accuracy")
        response_data = sample_data.query(metric_name="response_time")

        assert len(accuracy_data) == 6
        assert len(response_data) == 4

    def test_query_with_dimension_filter(self, sample_data):
        """Test querying with dimension filters."""
        easy_data = sample_data.query(
            metric_name="accuracy",
            filters={"level": "easy"}
        )

        assert len(easy_data) == 3
        assert all(dp.dimensions.get("level") == "easy" for dp in easy_data)


class TestAggregation:
    """Test aggregation functionality."""

    def test_aggregate_mean(self, sample_data):
        """Test mean aggregation."""
        result = sample_data.aggregate("accuracy")

        # 4 correct out of 6 = 0.667
        assert abs(result - 0.5) < 0.01

    def test_aggregate_with_filter(self, sample_data):
        """Test aggregation with dimension filter."""
        result = sample_data.aggregate(
            "accuracy",
            filters={"level": "easy"}
        )

        # 2 correct out of 3 on easy = 0.667
        assert abs(result - 0.667) < 0.01

    def test_aggregate_group_by(self, sample_data):
        """Test aggregation with grouping."""
        result = sample_data.aggregate(
            "accuracy",
            group_by=["level"]
        )

        assert "easy" in result
        assert "hard" in result
        # Easy: 2/3 correct
        assert abs(result["easy"] - 0.667) < 0.01
        # Hard: 1/3 correct
        assert abs(result["hard"] - 0.333) < 0.01


class TestAssessment:
    """Test metric assessment functionality."""

    def test_assess_all_metrics(self, sample_data):
        """Test assessing all metrics."""
        result = sample_data.assess()

        assert "metrics" in result
        assert "accuracy" in result["metrics"]

    def test_assess_single_metric(self, sample_data):
        """Test assessing a single metric."""
        result = sample_data.assess(metric_name="accuracy")

        assert "level" in result
        assert result["level"] in ["excellent", "good", "average", "below_average", "poor"]


class TestSession:
    """Test session management."""

    def test_start_session(self, engine_with_metrics):
        """Test starting a session."""
        session = engine_with_metrics.start_session(name="test_session")

        assert session is not None
        assert session.name == "test_session"
        assert session.is_active

    def test_end_session(self, engine_with_metrics):
        """Test ending a session."""
        engine_with_metrics.start_session(name="test_session")
        engine_with_metrics.record("accuracy", 0.9)
        session = engine_with_metrics.end_session()

        assert session is not None
        assert not session.is_active
        assert session.end_time is not None

    def test_session_records_linked(self, engine_with_metrics):
        """Test that records are linked to session."""
        session = engine_with_metrics.start_session(name="test_session")
        engine_with_metrics.record("accuracy", 0.8)
        engine_with_metrics.record("accuracy", 0.9)
        engine_with_metrics.end_session()

        datapoints = engine_with_metrics.query(metric_name="accuracy")
        assert all(dp.session_id == session.id for dp in datapoints)
