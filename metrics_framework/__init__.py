"""
Modular Metrics Framework
=========================

A flexible, reusable framework for tracking, analyzing, and visualizing
performance metrics across applications.

Features:
- Structured metric tracking with multiple data types
- Temporal and dimensional aggregation
- Performance assessment with configurable thresholds
- Visualization generators (heatmaps, charts, trends)
- Pluggable storage backends

Usage:
    from metrics_framework import MetricsEngine, MetricDefinition

    # Define metrics
    accuracy = MetricDefinition(
        name="accuracy",
        metric_type="percentage",
        aggregation="mean"
    )

    # Create engine
    engine = MetricsEngine()
    engine.register_metric(accuracy)

    # Record metrics
    engine.record("accuracy", 0.85, dimensions={"feature": "rotation"})

    # Get assessment
    report = engine.assess()
"""

__version__ = "1.0.0"
__author__ = "MindFold3D"

from .core.engine import MetricsEngine
from .core.definitions import MetricDefinition, DimensionDefinition
from .core.session import MetricSession
from .collectors.base import MetricCollector
from .collectors.counter import CounterCollector
from .collectors.gauge import GaugeCollector
from .collectors.histogram import HistogramCollector
from .collectors.timer import TimerCollector
from .aggregators.temporal import TemporalAggregator
from .aggregators.dimensional import DimensionalAggregator
from .assessors.threshold import ThresholdAssessor
from .assessors.trend import TrendAssessor
from .assessors.comparative import ComparativeAssessor
from .visualizers.heatmap import HeatmapVisualizer
from .visualizers.chart import ChartVisualizer
from .visualizers.trend import TrendVisualizer
from .storage.base import StorageBackend
from .storage.memory import MemoryStorage
from .storage.sqlalchemy_storage import SQLAlchemyStorage
from .storage.json_storage import JSONStorage

__all__ = [
    # Core
    "MetricsEngine",
    "MetricDefinition",
    "DimensionDefinition",
    "MetricSession",
    # Collectors
    "MetricCollector",
    "CounterCollector",
    "GaugeCollector",
    "HistogramCollector",
    "TimerCollector",
    # Aggregators
    "TemporalAggregator",
    "DimensionalAggregator",
    # Assessors
    "ThresholdAssessor",
    "TrendAssessor",
    "ComparativeAssessor",
    # Visualizers
    "HeatmapVisualizer",
    "ChartVisualizer",
    "TrendVisualizer",
    # Storage
    "StorageBackend",
    "MemoryStorage",
    "SQLAlchemyStorage",
    "JSONStorage",
]
