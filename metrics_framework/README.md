# Modular Performance Metrics Framework

## Specification Document

**Version:** 1.0.0
**Author:** MindFold3D
**License:** MIT

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Core Components](#core-components)
5. [Collectors](#collectors)
6. [Aggregators](#aggregators)
7. [Assessors](#assessors)
8. [Visualizers](#visualizers)
9. [Storage Backends](#storage-backends)
10. [API Reference](#api-reference)
11. [Integration Guide](#integration-guide)
12. [Examples](#examples)
13. [Extending the Framework](#extending-the-framework)

---

## Overview

The Modular Performance Metrics Framework is a Python library designed to provide comprehensive tracking, analysis, and visualization of performance metrics across applications. Originally developed for spatial cognition training (MindFold3D), it is architected for reuse in any domain requiring performance analytics.

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-type Metrics** | Counters, gauges, histograms, timers, percentages |
| **Dimensional Analysis** | Track metrics across unlimited dimensions |
| **Temporal Aggregation** | Time-based rollups (minute, hour, day, week, month) |
| **Performance Assessment** | Threshold-based grading with configurable levels |
| **Trend Analysis** | Detect improving/declining patterns over time |
| **Cognitive Profiling** | Map features to cognitive dimensions |
| **Visualization Ready** | Generate Chart.js/D3.js compatible data structures |
| **Pluggable Storage** | Memory, JSON, SQLAlchemy backends |
| **Thread-Safe** | Safe for concurrent access |

### Design Principles

1. **Modularity**: Each component is independent and replaceable
2. **Extensibility**: Easy to add custom collectors, assessors, visualizers
3. **Zero Dependencies** (core): Only Python standard library required
4. **Optional Dependencies**: SQLAlchemy for database storage
5. **Framework Agnostic**: Works with FastAPI, Flask, Django, or standalone

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MetricsEngine                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Sessions   │  │   Metrics    │  │   Storage    │          │
│  │  Management  │  │ Definitions  │  │   Backend    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Collectors    │  │   Aggregators   │  │    Assessors    │
│  ─────────────  │  │  ─────────────  │  │  ─────────────  │
│  • Counter      │  │  • Temporal     │  │  • Threshold    │
│  • Gauge        │  │  • Dimensional  │  │  • Trend        │
│  • Histogram    │  │  • Rolling      │  │  • Comparative  │
│  • Timer        │  │                 │  │  • Cognitive    │
│  • Composite    │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │   Visualizers   │
                    │  ─────────────  │
                    │  • Heatmap      │
                    │  • Chart        │
                    │  • Trend        │
                    │  • Radar        │
                    └─────────────────┘
```

### Directory Structure

```
metrics_framework/
├── __init__.py              # Public API exports
├── README.md                # This document
│
├── core/                    # Core components
│   ├── __init__.py
│   ├── engine.py           # MetricsEngine class
│   ├── definitions.py      # MetricDefinition, enums, configs
│   ├── session.py          # MetricSession class
│   └── datapoint.py        # DataPoint, AggregatedDataPoint
│
├── collectors/              # Metric collectors
│   ├── __init__.py
│   ├── base.py             # MetricCollector ABC
│   ├── counter.py          # CounterCollector, SuccessFailureCounter
│   ├── gauge.py            # GaugeCollector, BoundedGauge
│   ├── histogram.py        # HistogramCollector, CategoricalHistogram
│   ├── timer.py            # TimerCollector, ResponseTimeTracker
│   └── composite.py        # CompositeCollector
│
├── aggregators/             # Data aggregators
│   ├── __init__.py
│   ├── temporal.py         # TemporalAggregator
│   ├── dimensional.py      # DimensionalAggregator
│   └── rolling.py          # RollingAggregator, TimeWindowAggregator
│
├── assessors/               # Performance assessors
│   ├── __init__.py
│   ├── threshold.py        # ThresholdAssessor
│   ├── trend.py            # TrendAssessor
│   ├── comparative.py      # ComparativeAssessor
│   └── cognitive.py        # CognitiveDimensionAssessor
│
├── visualizers/             # Visualization generators
│   ├── __init__.py
│   ├── heatmap.py          # HeatmapVisualizer
│   ├── chart.py            # ChartVisualizer
│   ├── trend.py            # TrendVisualizer
│   └── radar.py            # RadarVisualizer
│
├── storage/                 # Storage backends
│   ├── __init__.py
│   ├── base.py             # StorageBackend ABC
│   ├── memory.py           # MemoryStorage
│   ├── json_storage.py     # JSONStorage
│   └── sqlalchemy_storage.py # SQLAlchemyStorage
│
├── integrations/            # Application integrations
│   ├── __init__.py
│   └── mindfold.py         # MindFoldMetrics
│
└── utils/                   # Utilities
    ├── __init__.py
    ├── stats.py            # Statistical functions
    └── time.py             # Time utilities
```

---

## Installation

### Basic Installation (No Dependencies)

```bash
# Copy the metrics_framework directory to your project
cp -r metrics_framework /path/to/your/project/
```

### With Database Support

```bash
pip install sqlalchemy
```

### Full Installation

```bash
pip install sqlalchemy numpy
```

---

## Core Components

### MetricsEngine

The central orchestrator for all metrics operations.

```python
from metrics_framework import MetricsEngine

engine = MetricsEngine(
    storage_backend=None,  # Defaults to MemoryStorage
    auto_persist=True,     # Auto-save datapoints
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `register_metric(metric)` | Register a metric definition |
| `record(metric_name, value, ...)` | Record a metric value |
| `query(metric_name, ...)` | Query stored datapoints |
| `aggregate(metric_name, ...)` | Aggregate metric values |
| `assess(metric_name, ...)` | Generate assessment report |
| `start_session(name, user_id)` | Start a new session |
| `end_session()` | End current session |

### MetricDefinition

Defines the structure and behavior of a metric.

```python
from metrics_framework import MetricDefinition, MetricType, AggregationType

metric = MetricDefinition(
    name="accuracy",
    metric_type=MetricType.PERCENTAGE,
    description="Percentage of correct responses",
    unit="%",
    aggregation=AggregationType.MEAN,
    higher_is_better=True,
    min_value=0,
    max_value=1,
    thresholds=ThresholdConfig(
        excellent=0.9,
        good=0.7,
        average=0.5,
        below_average=0.3,
        poor=0.1,
    ),
)
```

#### Metric Types

| Type | Description | Use Case |
|------|-------------|----------|
| `COUNTER` | Monotonically increasing | Total requests, errors |
| `GAUGE` | Point-in-time value | Current memory, active users |
| `HISTOGRAM` | Value distribution | Response times, sizes |
| `TIMER` | Time measurements | Request duration |
| `PERCENTAGE` | 0-1 or 0-100 values | Accuracy, completion rate |
| `RATE` | Value per time unit | Requests per second |
| `BOOLEAN` | True/False outcomes | Success/failure |

#### Aggregation Types

| Type | Description |
|------|-------------|
| `SUM` | Sum of all values |
| `MEAN` | Arithmetic mean |
| `MEDIAN` | Middle value |
| `MIN` / `MAX` | Minimum / Maximum |
| `COUNT` | Number of values |
| `STD` | Standard deviation |
| `PERCENTILE` | Nth percentile |

### DataPoint

Represents a single metric recording.

```python
from metrics_framework.core import DataPoint

datapoint = DataPoint(
    metric_name="accuracy",
    value=0.85,
    timestamp=datetime.utcnow(),
    dimensions={"feature": "rotation", "difficulty": "hard"},
    metadata={"response_time": 2.5},
    session_id="session-123",
    user_id="user-456",
)
```

### MetricSession

Scoped metric collection with lifecycle management.

```python
# Context manager usage
with engine.start_session(name="training", user_id="user-123") as session:
    session.record("accuracy", 0.85, dimensions={"level": 1})
    session.record("accuracy", 0.90, dimensions={"level": 2})
# Session automatically ended

# Manual usage
session = engine.start_session()
# ... record metrics ...
engine.end_session()
```

---

## Collectors

Collectors are specialized classes for gathering specific types of metrics.

### CounterCollector

For monotonically increasing values.

```python
from metrics_framework.collectors import CounterCollector

counter = CounterCollector("requests", engine=engine)
counter.inc()           # Increment by 1
counter.increment(5)    # Increment by 5
print(counter.get_value())  # Current count
```

### SuccessFailureCounter

Track success/failure outcomes with automatic rate calculation.

```python
from metrics_framework.collectors import SuccessFailureCounter

tracker = SuccessFailureCounter("attempts", engine=engine)
tracker.record_success()
tracker.record_failure()

print(tracker.success_rate)  # 0.5
print(tracker.total)         # 2
```

### GaugeCollector

For point-in-time values that can increase or decrease.

```python
from metrics_framework.collectors import GaugeCollector

gauge = GaugeCollector("active_users", engine=engine)
gauge.set(100)
gauge.increment(10)
gauge.decrement(5)
print(gauge.get_value())  # 105
```

### HistogramCollector

For tracking value distributions.

```python
from metrics_framework.collectors import HistogramCollector

histogram = HistogramCollector(
    "response_size",
    buckets=[100, 500, 1000, 5000, 10000],
    engine=engine,
)

histogram.observe(250)
histogram.observe(1500)
histogram.observe(750)

stats = histogram.get_distribution()
# {
#     "count": 3,
#     "mean": 833.33,
#     "median": 750,
#     "p95": 1500,
#     "buckets": [...]
# }
```

### TimerCollector

For measuring durations with multiple timing methods.

```python
from metrics_framework.collectors import TimerCollector

timer = TimerCollector("request_duration", engine=engine)

# Method 1: Manual start/stop
timer.start()
# ... do work ...
duration = timer.stop()

# Method 2: Context manager
with timer.time():
    # ... do work ...

# Method 3: Decorator
@timer.timed()
def my_function():
    # ... do work ...

# Method 4: Direct observation
timer.observe(2.5)  # Record 2.5 seconds
```

### CompositeCollector

For multi-dimensional performance tracking.

```python
from metrics_framework.collectors import CompositeCollector

collector = CompositeCollector("performance", engine=engine)

collector.record_attempt(
    correct=True,
    response_time=2.5,
    features={
        "symmetry_class": "high",
        "voxel_count": 5,
        "difficulty_score": 0.7,
    }
)

# Get statistics
stats = collector.get_all_feature_stats()
weak_areas = collector.get_weak_areas(min_attempts=3, max_success_rate=0.5)
strong_areas = collector.get_strong_areas(min_attempts=3, min_success_rate=0.8)
```

---

## Aggregators

Aggregators combine multiple datapoints into summary statistics.

### TemporalAggregator

Time-based aggregation and trend calculation.

```python
from metrics_framework.aggregators import TemporalAggregator

aggregator = TemporalAggregator(aggregation="mean")

# Aggregate by time interval
hourly = aggregator.aggregate_by_interval(datapoints, interval="hour")
daily = aggregator.aggregate_by_interval(datapoints, interval="day")

# Calculate trend
trend = aggregator.calculate_trend(datapoints, interval="day")
# {
#     "trend": "improving",
#     "slope": 0.05,
#     "change_percent": 15.2,
#     "r_squared": 0.87,
# }

# Get time series for charting
time_series = aggregator.get_time_series(
    datapoints,
    interval="hour",
    fill_gaps=True,
    fill_value=0.0,
)
```

### DimensionalAggregator

Cross-dimensional analysis and comparison.

```python
from metrics_framework.aggregators import DimensionalAggregator

aggregator = DimensionalAggregator(aggregation="mean")

# Aggregate by single dimension
by_feature = aggregator.aggregate_by_dimension(datapoints, "feature")
# {"rotation": 0.75, "mirror": 0.82, ...}

# Aggregate by multiple dimensions
by_multiple = aggregator.aggregate_grouped(
    datapoints,
    group_by=["feature", "difficulty"]
)
# {("rotation", "easy"): 0.90, ("rotation", "hard"): 0.65, ...}

# Cross-dimension matrix (for heatmaps)
matrix = aggregator.get_cross_dimension_matrix(
    datapoints,
    x_dimension="feature",
    y_dimension="difficulty",
)

# Ranking
ranking = aggregator.get_dimension_ranking(
    datapoints,
    dimension="feature",
    ascending=False,  # Best first
    limit=10,
)
```

### RollingAggregator

Moving window calculations for real-time analysis.

```python
from metrics_framework.aggregators import RollingAggregator, TimeWindowAggregator

# Fixed-size window
rolling = RollingAggregator(window_size=10, aggregation="mean")
rolling.add(0.85)
rolling.add(0.90)
print(rolling.get_value())  # Rolling mean

# Time-based window
time_window = TimeWindowAggregator(window_duration="1h", aggregation="mean")
time_window.add(0.85)
print(time_window.get_value())  # Mean of last hour
```

---

## Assessors

Assessors evaluate performance and generate insights.

### ThresholdAssessor

Grade performance against configurable thresholds.

```python
from metrics_framework.assessors import ThresholdAssessor
from metrics_framework.core.definitions import ThresholdConfig

assessor = ThresholdAssessor(
    thresholds=ThresholdConfig(
        excellent=0.9,
        good=0.7,
        average=0.5,
        below_average=0.3,
        poor=0.1,
    ),
    higher_is_better=True,
)

result = assessor.assess_value(0.75, metric_name="accuracy")
# AssessmentResult(
#     metric_name="accuracy",
#     value=0.75,
#     level=AssessmentLevel.GOOD,
#     ...
# )

# Assess multiple metrics
results = assessor.assess_multiple({
    "accuracy": 0.85,
    "response_time": 2.5,
    "completion_rate": 0.92,
})

summary = assessor.get_summary_assessment(results)
# {
#     "overall_level": "good",
#     "passing_count": 3,
#     "failing_count": 0,
#     "needs_attention": [],
# }
```

#### Assessment Levels

| Level | Typical Threshold | Description |
|-------|-------------------|-------------|
| `EXCELLENT` | ≥ 90% | Outstanding performance |
| `GOOD` | ≥ 70% | Above average |
| `AVERAGE` | ≥ 50% | Acceptable |
| `BELOW_AVERAGE` | ≥ 30% | Needs improvement |
| `POOR` | ≥ 10% | Significant concerns |
| `CRITICAL` | < 10% | Immediate attention needed |

### TrendAssessor

Analyze performance trends over time.

```python
from metrics_framework.assessors import TrendAssessor

assessor = TrendAssessor(
    strong_threshold=0.1,
    min_data_points=5,
    higher_is_better=True,
)

result = assessor.analyze_trend(datapoints, interval="day")
# TrendResult(
#     direction=TrendDirection.IMPROVING,
#     slope=0.02,
#     change_percent=15.5,
#     confidence=0.85,
# )

# Compare two periods
comparison = assessor.compare_periods(
    datapoints,
    period1_start=datetime(2024, 1, 1),
    period1_end=datetime(2024, 1, 15),
    period2_start=datetime(2024, 1, 16),
    period2_end=datetime(2024, 1, 31),
)
```

#### Trend Directions

| Direction | Description |
|-----------|-------------|
| `STRONGLY_IMPROVING` | Significant positive trend |
| `IMPROVING` | Positive trend |
| `STABLE` | No significant change |
| `DECLINING` | Negative trend |
| `STRONGLY_DECLINING` | Significant negative trend |

### ComparativeAssessor

Compare performance across dimensions.

```python
from metrics_framework.assessors import ComparativeAssessor

assessor = ComparativeAssessor(higher_is_better=True)

# Compare two dimension values
comparison = assessor.compare_dimension_values(
    datapoints,
    dimension="difficulty",
    value1="easy",
    value2="hard",
)
# ComparisonResult with statistical significance

# Rank all values
ranking = assessor.rank_dimension_values(
    datapoints,
    dimension="feature",
    limit=10,
)

# Identify outliers
outliers = assessor.identify_outliers(
    datapoints,
    dimension="feature",
    method="iqr",
    threshold=1.5,
)
```

### CognitiveDimensionAssessor

Map features to cognitive dimensions (specialized for spatial cognition).

```python
from metrics_framework.assessors import CognitiveDimensionAssessor

assessor = CognitiveDimensionAssessor(
    min_attempts_per_feature=3,
)

# Assess single dimension
result = assessor.assess_dimension("mental_transformation", feature_stats)

# Create full cognitive profile
profile = assessor.create_cognitive_profile(feature_stats, user_id="user-123")
# CognitiveProfile(
#     overall_score=0.72,
#     overall_level=AssessmentLevel.GOOD,
#     dimensions={
#         "mental_transformation": CognitiveDimensionResult(...),
#         "mirror_discrimination": CognitiveDimensionResult(...),
#         ...
#     },
#     strengths=["visual_perception", "structural_analysis"],
#     weaknesses=["mental_transformation"],
#     recommendations=["Practice mental rotation exercises..."],
# )
```

#### Default Cognitive Dimensions

| Dimension | Description | Key Features |
|-----------|-------------|--------------|
| `mental_transformation` | Mental rotation ability | rotation_confusability, rotational_symmetry_order |
| `mirror_discrimination` | Mirror image recognition | mirror_confusability, reflection_planes_count |
| `spatial_visualization` | 3D space manipulation | convexity, planarity_score |
| `spatial_navigation` | Navigating 3D space | bounding_box_ratio, dominant_axis |
| `visual_perception` | Visual processing | gestalt_grouping_cues, color_confusability |
| `structural_analysis` | Structure segmentation | branching_factor, number_of_components |
| `executive_strategy` | Strategic planning | difficulty_score, distractor_similarity |
| `spatial_working_memory` | Spatial information retention | voxel_count, surface_area |

---

## Visualizers

Generate data structures ready for frontend visualization libraries.

### HeatmapVisualizer

Generate heatmap data for D3.js or similar libraries.

```python
from metrics_framework.visualizers import HeatmapVisualizer, HeatmapConfig

config = HeatmapConfig(
    color_thresholds=[
        (0.0, "#ca0020"),   # Red: 0-30%
        (0.3, "#f4a582"),   # Light red: 30-50%
        (0.5, "#f7f7f7"),   # White: 50-70%
        (0.7, "#92c5de"),   # Light blue: 70-90%
        (0.9, "#0571b0"),   # Blue: 90-100%
    ],
    show_values=True,
    min_count_threshold=3,
)

visualizer = HeatmapVisualizer(
    x_dimension="feature",
    y_dimension="value",
    aggregation="mean",
    config=config,
)

heatmap_data = visualizer.generate(datapoints)
# {
#     "type": "heatmap",
#     "x_labels": ["rotation", "mirror", ...],
#     "y_labels": ["low", "medium", "high"],
#     "cells": [
#         {"x": 0, "y": 0, "value": 0.85, "color": "#92c5de", ...},
#         ...
#     ],
#     "config": {...},
# }
```

### ChartVisualizer

Generate Chart.js compatible data structures.

```python
from metrics_framework.visualizers import ChartVisualizer, ChartConfig

visualizer = ChartVisualizer(chart_type="bar")

# Simple chart
chart_data = visualizer.generate(datapoints, group_by="feature")

# Multi-series chart
multi_series = visualizer.generate_multi_series({
    "Session 1": session1_datapoints,
    "Session 2": session2_datapoints,
})

# Stacked bar chart
stacked = visualizer.generate_stacked_bar(
    datapoints,
    stack_by="outcome",
    group_by="feature",
)
```

#### Chart Types

- `bar` - Vertical bar chart
- `line` - Line chart
- `pie` - Pie chart
- `doughnut` - Doughnut chart
- `scatter` - Scatter plot

### TrendVisualizer

Generate time-series trend data with moving averages.

```python
from metrics_framework.visualizers import TrendVisualizer

visualizer = TrendVisualizer(interval="day")

trend_data = visualizer.generate(datapoints)
# Includes: actual values, SMA, EMA, trend analysis

# Cumulative trend
cumulative = visualizer.generate_cumulative_trend(datapoints)

# Progress over sessions
progress = visualizer.generate_progress_trend(session_summaries)
```

### RadarVisualizer

Generate radar/spider chart data for cognitive profiles.

```python
from metrics_framework.visualizers import RadarVisualizer

visualizer = RadarVisualizer(max_value=1.0)

# Single profile
radar_data = visualizer.generate({
    "Mental Transformation": 0.75,
    "Mirror Discrimination": 0.82,
    "Spatial Visualization": 0.68,
}, title="Cognitive Profile")

# Profile comparison
comparison = visualizer.generate_comparison({
    "Current": current_scores,
    "Previous": previous_scores,
})

# Strength/weakness visualization
strength_weakness = visualizer.generate_strength_weakness(scores)
```

---

## Storage Backends

Pluggable storage for metrics persistence.

### MemoryStorage

In-memory storage (non-persistent, fast).

```python
from metrics_framework.storage import MemoryStorage

storage = MemoryStorage()
engine = MetricsEngine(storage_backend=storage)
```

### JSONStorage

File-based JSON storage.

```python
from metrics_framework.storage import JSONStorage

storage = JSONStorage(
    file_path="metrics_data.json",
    auto_save=True,
    pretty_print=True,
)
engine = MetricsEngine(storage_backend=storage)
```

### SQLAlchemyStorage

Database storage using SQLAlchemy.

```python
from metrics_framework.storage import SQLAlchemyStorage

# SQLite
storage = SQLAlchemyStorage("sqlite:///metrics.db")

# PostgreSQL
storage = SQLAlchemyStorage("postgresql://user:pass@localhost/metrics")

# MySQL
storage = SQLAlchemyStorage("mysql://user:pass@localhost/metrics")

engine = MetricsEngine(storage_backend=storage)
```

### Custom Storage Backend

Implement the `StorageBackend` interface:

```python
from metrics_framework.storage import StorageBackend

class MyCustomStorage(StorageBackend):
    def store(self, datapoint):
        # Store single datapoint
        pass

    def store_batch(self, datapoints):
        # Store multiple datapoints
        pass

    def query(self, filters, limit=None):
        # Query datapoints
        return []

    def delete(self, filters):
        # Delete datapoints
        return 0

    def count(self, filters=None):
        # Count datapoints
        return 0

    def clear(self):
        # Clear all data
        pass
```

---

## API Reference

### MetricsEngine API

```python
class MetricsEngine:
    # Metric Registration
    def register_metric(metric: MetricDefinition) -> MetricsEngine
    def register_metrics(metrics: List[MetricDefinition]) -> MetricsEngine
    def get_metric(name: str) -> Optional[MetricDefinition]
    def list_metrics() -> List[str]

    # Recording
    def record(
        metric_name: str,
        value: Any,
        dimensions: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        timestamp: Optional[datetime] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> DataPoint

    def record_batch(records: List[Dict]) -> List[DataPoint]

    # Sessions
    def start_session(name: str = "", user_id: Optional[str] = None) -> MetricSession
    def end_session(session_id: Optional[str] = None) -> Optional[MetricSession]
    def get_session(session_id: str) -> Optional[MetricSession]

    # Querying
    def query(
        metric_name: Optional[str] = None,
        dimensions: Optional[Dict] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[DataPoint]

    # Aggregation
    def aggregate(
        metric_name: str,
        aggregation: Optional[AggregationType] = None,
        group_by: Optional[List[str]] = None,
        **query_params,
    ) -> Union[float, Dict]

    # Assessment
    def assess(
        metric_name: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]

    # Visualization
    def get_heatmap_data(...) -> Dict
    def get_chart_data(...) -> Dict
    def get_trend_data(...) -> Dict

    # Event Hooks
    def on_record(callback: Callable[[DataPoint], None]) -> MetricsEngine
    def on_session_start(callback: Callable[[MetricSession], None]) -> MetricsEngine
    def on_session_end(callback: Callable[[MetricSession], None]) -> MetricsEngine

    # Persistence
    def persist() -> None
    def clear(confirm: bool = False) -> None
    def export(format: str = "json") -> Any
```

---

## Integration Guide

### FastAPI Integration

```python
from fastapi import FastAPI, Depends
from metrics_framework.integrations import MindFoldMetrics

app = FastAPI()
metrics = MindFoldMetrics()

@app.post("/submit-response")
async def submit_response(data: ResponseSubmission):
    result = metrics.record_attempt(
        correct=data.correct,
        response_time=data.response_time,
        target_features=data.target_features,
        user_id=data.user_id,
    )
    return result

@app.get("/scorecard")
async def get_scorecard():
    return metrics.get_scorecard()

@app.get("/heatmap")
async def get_heatmap():
    return metrics.get_heatmap_data()
```

### Flask Integration

```python
from flask import Flask, jsonify, request
from metrics_framework.integrations import MindFoldMetrics

app = Flask(__name__)
metrics = MindFoldMetrics()

@app.route("/submit-response", methods=["POST"])
def submit_response():
    data = request.json
    result = metrics.record_attempt(
        correct=data["correct"],
        response_time=data["response_time"],
        target_features=data["target_features"],
    )
    return jsonify(result)

@app.route("/scorecard")
def get_scorecard():
    return jsonify(metrics.get_scorecard())
```

### Standalone Usage

```python
from metrics_framework import MetricsEngine, MetricDefinition
from metrics_framework.storage import SQLAlchemyStorage

# Initialize
engine = MetricsEngine(
    storage_backend=SQLAlchemyStorage("sqlite:///my_app.db")
)

# Define metrics for your domain
engine.register_metric(MetricDefinition(
    name="task_completion",
    metric_type="percentage",
    higher_is_better=True,
))

engine.register_metric(MetricDefinition(
    name="error_rate",
    metric_type="percentage",
    higher_is_better=False,
))

# Use throughout your application
engine.record("task_completion", 0.95, dimensions={"task_type": "upload"})
engine.record("error_rate", 0.02, dimensions={"endpoint": "/api/users"})

# Generate reports
report = engine.assess()
```

---

## Examples

### Example 1: Basic Metrics Tracking

```python
from metrics_framework import MetricsEngine, MetricDefinition

engine = MetricsEngine()

# Register metrics
engine.register_metric(MetricDefinition(name="score", metric_type="gauge"))
engine.register_metric(MetricDefinition(name="attempts", metric_type="counter"))

# Record
engine.record("score", 85)
engine.record("attempts", 1)

# Query
scores = engine.query(metric_name="score")
print(f"Average score: {engine.aggregate('score')}")
```

### Example 2: Session-Based Tracking

```python
from metrics_framework import MetricsEngine

engine = MetricsEngine()

# Start session
session = engine.start_session(name="Game Session", user_id="player123")

# Record metrics during session
for question in questions:
    answer = get_user_answer(question)
    engine.record(
        "accuracy",
        1.0 if answer.correct else 0.0,
        dimensions={"difficulty": question.difficulty},
    )

# End session
engine.end_session()

# Get session summary
summary = session.get_summary()
```

### Example 3: Visualization Pipeline

```python
from metrics_framework import MetricsEngine
from metrics_framework.visualizers import HeatmapVisualizer, ChartVisualizer

engine = MetricsEngine()
# ... record metrics ...

# Generate heatmap
heatmap_viz = HeatmapVisualizer(x_dimension="feature", y_dimension="level")
heatmap_data = heatmap_viz.generate(engine.query())

# Generate chart
chart_viz = ChartVisualizer(chart_type="bar")
chart_data = chart_viz.generate(engine.query(), group_by="feature")

# Send to frontend
return {
    "heatmap": heatmap_data,
    "chart": chart_data,
}
```

### Example 4: Custom Cognitive Dimensions

```python
from metrics_framework.assessors.cognitive import (
    CognitiveDimensionAssessor,
    CognitiveDimensionMapping,
)

# Define custom cognitive dimensions for your domain
custom_dimensions = [
    CognitiveDimensionMapping(
        name="problem_solving",
        description="Ability to solve complex problems",
        features=["complexity_level", "steps_required", "time_pressure"],
    ),
    CognitiveDimensionMapping(
        name="attention_to_detail",
        description="Accuracy in detail-oriented tasks",
        features=["error_type", "precision_required"],
    ),
]

assessor = CognitiveDimensionAssessor(dimension_mappings=custom_dimensions)
profile = assessor.create_cognitive_profile(feature_stats)
```

---

## Extending the Framework

### Custom Collector

```python
from metrics_framework.collectors import MetricCollector

class RateCollector(MetricCollector):
    def __init__(self, metric_name, window_seconds=60, **kwargs):
        super().__init__(metric_name, **kwargs)
        self._window = window_seconds
        self._events = []

    def collect(self, value=1, **kwargs):
        now = datetime.utcnow()
        self._events.append((now, value))
        self._cleanup()
        return self.get_value()

    def get_value(self):
        self._cleanup()
        return sum(v for _, v in self._events) / self._window

    def _cleanup(self):
        cutoff = datetime.utcnow() - timedelta(seconds=self._window)
        self._events = [(t, v) for t, v in self._events if t > cutoff]
```

### Custom Assessor

```python
from metrics_framework.assessors.threshold import ThresholdAssessor

class CustomAssessor(ThresholdAssessor):
    def assess_with_context(self, value, context):
        # Add domain-specific logic
        adjusted_value = self._adjust_for_context(value, context)
        return super().assess_value(adjusted_value)

    def _adjust_for_context(self, value, context):
        # Custom adjustment logic
        if context.get("is_first_attempt"):
            return value * 1.1  # Bonus for first attempt
        return value
```

### Custom Visualizer

```python
from metrics_framework.visualizers.chart import ChartVisualizer

class CustomVisualizer(ChartVisualizer):
    def generate_dashboard(self, datapoints):
        return {
            "summary_cards": self._generate_summary_cards(datapoints),
            "main_chart": self.generate(datapoints, group_by="category"),
            "trend_sparklines": self._generate_sparklines(datapoints),
        }

    def _generate_summary_cards(self, datapoints):
        # Custom summary card generation
        pass

    def _generate_sparklines(self, datapoints):
        # Custom sparkline generation
        pass
```

---

## Performance Considerations

| Scenario | Recommendation |
|----------|----------------|
| < 10,000 datapoints | MemoryStorage is fine |
| 10,000 - 100,000 datapoints | JSONStorage with periodic cleanup |
| > 100,000 datapoints | SQLAlchemyStorage with indexing |
| High write throughput | Batch writes with `store_batch()` |
| Real-time dashboards | Use RollingAggregator for live stats |

---

## Changelog

### Version 1.0.0
- Initial release
- Core engine with sessions and datapoints
- 5 collector types
- 3 aggregator types
- 4 assessor types
- 4 visualizer types
- 3 storage backends
- MindFold3D integration

---

## License

MIT License - Free for commercial and personal use.
