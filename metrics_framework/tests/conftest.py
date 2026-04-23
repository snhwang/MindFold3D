"""Pytest fixtures for metrics framework tests."""

import pytest
from metrics_framework import MetricsEngine, MetricDefinition
from metrics_framework.core.definitions import ThresholdConfig
from metrics_framework.storage import MemoryStorage


@pytest.fixture
def memory_storage():
    """Create a fresh memory storage instance."""
    return MemoryStorage()


@pytest.fixture
def engine(memory_storage):
    """Create a metrics engine with memory storage."""
    return MetricsEngine(storage_backend=memory_storage)


@pytest.fixture
def engine_with_metrics(engine):
    """Create an engine with pre-registered metrics."""
    engine.register_metrics([
        MetricDefinition(
            name="accuracy",
            metric_type="percentage",
            description="Accuracy rate",
            aggregation="mean",
            higher_is_better=True,
            thresholds=ThresholdConfig(
                excellent=0.9,
                good=0.7,
                average=0.5,
                below_average=0.3,
                poor=0.1,
            ),
        ),
        MetricDefinition(
            name="response_time",
            metric_type="timer",
            description="Response time in seconds",
            unit="seconds",
            aggregation="mean",
            higher_is_better=False,
        ),
        MetricDefinition(
            name="score",
            metric_type="gauge",
            description="Score value",
            aggregation="sum",
            higher_is_better=True,
        ),
    ])
    return engine


@pytest.fixture
def sample_data(engine_with_metrics):
    """Engine with sample data recorded."""
    # Record accuracy data with dimensions
    engine_with_metrics.record("accuracy", 1.0, dimensions={"level": "easy", "category": "A"})
    engine_with_metrics.record("accuracy", 1.0, dimensions={"level": "easy", "category": "A"})
    engine_with_metrics.record("accuracy", 0.0, dimensions={"level": "easy", "category": "B"})
    engine_with_metrics.record("accuracy", 1.0, dimensions={"level": "hard", "category": "A"})
    engine_with_metrics.record("accuracy", 0.0, dimensions={"level": "hard", "category": "B"})
    engine_with_metrics.record("accuracy", 0.0, dimensions={"level": "hard", "category": "B"})

    # Record response times
    engine_with_metrics.record("response_time", 1.5, dimensions={"level": "easy"})
    engine_with_metrics.record("response_time", 2.0, dimensions={"level": "easy"})
    engine_with_metrics.record("response_time", 3.5, dimensions={"level": "hard"})
    engine_with_metrics.record("response_time", 4.0, dimensions={"level": "hard"})

    return engine_with_metrics
