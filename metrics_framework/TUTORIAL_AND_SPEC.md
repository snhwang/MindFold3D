# Modular Performance Metrics Framework

## Tutorial & Specification

---

# Part 1: Tutorial

## Introduction

The Modular Performance Metrics Framework is a Python library for tracking, analyzing, and visualizing performance metrics. It follows a simple pipeline:

```
Record → Store → Aggregate → Assess → Visualize
```

This tutorial walks through each step with practical examples.

---

## Quick Start (5 Minutes)

### Installation

```bash
# Copy the metrics_framework directory to your project
# No dependencies required for basic usage

# Optional: For database storage
pip install sqlalchemy
```

### Basic Usage

```python
from metrics_framework import MetricsEngine, MetricDefinition

# 1. Create engine
engine = MetricsEngine()

# 2. Define a metric
engine.register_metric(MetricDefinition(
    name="score",
    metric_type="percentage",
    higher_is_better=True,
))

# 3. Record data points
engine.record("score", 0.85, dimensions={"level": "easy"})
engine.record("score", 0.72, dimensions={"level": "hard"})
engine.record("score", 0.91, dimensions={"level": "easy"})

# 4. Analyze
print(engine.aggregate("score"))  # → 0.827 (mean)
print(engine.aggregate("score", group_by=["level"]))  # → {"easy": 0.88, "hard": 0.72}

# 5. Assess
report = engine.assess()
print(report)  # → {"score": {"value": 0.827, "level": "good", ...}}
```

---

## Tutorial 1: Understanding Data Flow

### What Gets Stored

Every `record()` call creates a **DataPoint**:

```python
engine.record("accuracy", 0.85, dimensions={"category": "A"})
```

Creates:

```python
DataPoint(
    id="auto-generated-uuid",
    metric_name="accuracy",
    value=0.85,
    timestamp=datetime.now(),  # Auto-set
    dimensions={"category": "A"},
    metadata={},
    session_id=None,
    user_id=None,
)
```

### Storage Options

```python
# In-memory (default) - fast, not persistent
from metrics_framework.storage import MemoryStorage
engine = MetricsEngine(storage_backend=MemoryStorage())

# JSON file - persistent, human-readable
from metrics_framework.storage import JSONStorage
engine = MetricsEngine(storage_backend=JSONStorage("metrics.json"))

# Database - scalable, queryable
from metrics_framework.storage import SQLAlchemyStorage
engine = MetricsEngine(storage_backend=SQLAlchemyStorage("sqlite:///metrics.db"))
```

---

## Tutorial 2: Defining Metrics

Metrics have properties that control how they're processed and assessed:

```python
from metrics_framework import MetricDefinition, MetricType, AggregationType
from metrics_framework.core.definitions import ThresholdConfig

metric = MetricDefinition(
    # Required
    name="response_time",

    # Type (affects validation and display)
    metric_type=MetricType.TIMER,  # or "timer"

    # Description
    description="Time to complete a request",
    unit="seconds",

    # How to combine multiple values
    aggregation=AggregationType.MEAN,  # or "mean", "sum", "median", etc.

    # Assessment direction
    higher_is_better=False,  # Lower response time is better

    # Value constraints
    min_value=0,
    max_value=None,  # No upper limit

    # Thresholds for grading (normalized 0-1 scale)
    thresholds=ThresholdConfig(
        excellent=0.9,   # Top 10% → "excellent"
        good=0.7,        # Top 30% → "good"
        average=0.5,     # Top 50% → "average"
        below_average=0.3,
        poor=0.1,
    ),

    # Optional target
    target_value=2.0,  # Target: 2 seconds
)
```

### Metric Types

| Type | Use Case | Example Values |
|------|----------|----------------|
| `counter` | Cumulative totals | 1, 2, 3, 4... (only increases) |
| `gauge` | Point-in-time values | 42, 38, 45, 41 (can go up/down) |
| `histogram` | Distributions | Response times, file sizes |
| `timer` | Durations | 0.5s, 1.2s, 0.8s |
| `percentage` | Ratios 0-1 or 0-100 | 0.85, 0.92, 0.78 |
| `rate` | Value per time | 100 req/sec |
| `boolean` | True/False outcomes | 1, 0, 1, 1, 0 |

### Aggregation Types

| Type | Formula | Use Case |
|------|---------|----------|
| `sum` | Σ values | Total sales, total errors |
| `mean` | Σ values / n | Average score |
| `median` | Middle value | Typical response time |
| `min` | Lowest value | Best time |
| `max` | Highest value | Peak memory |
| `count` | Number of values | Total attempts |
| `std` | Standard deviation | Consistency measure |
| `percentile` | Nth percentile | p95 latency |

---

## Tutorial 3: Recording with Dimensions

Dimensions let you slice data for analysis:

```python
# Without dimensions - just raw values
engine.record("sales", 100)
engine.record("sales", 150)
engine.record("sales", 200)
engine.aggregate("sales")  # → 150 (mean)

# With dimensions - can analyze by category
engine.record("sales", 100, dimensions={"region": "north", "product": "widget"})
engine.record("sales", 150, dimensions={"region": "south", "product": "widget"})
engine.record("sales", 200, dimensions={"region": "north", "product": "gadget"})

# Now you can slice
engine.aggregate("sales", group_by=["region"])
# → {"north": 150, "south": 150}

engine.aggregate("sales", group_by=["product"])
# → {"widget": 125, "gadget": 200}

engine.aggregate("sales", group_by=["region", "product"])
# → {("north", "widget"): 100, ("south", "widget"): 150, ("north", "gadget"): 200}
```

### Common Dimension Patterns

**E-commerce:**
```python
dimensions={"traffic_source": "google", "device": "mobile", "country": "US"}
```

**Gaming:**
```python
dimensions={"level": 5, "difficulty": "hard", "character_class": "warrior"}
```

**SaaS:**
```python
dimensions={"plan": "premium", "feature": "export", "user_segment": "enterprise"}
```

**Training/Education:**
```python
dimensions={"topic": "algebra", "question_type": "multiple_choice", "difficulty": 3}
```

---

## Tutorial 4: Sessions

Sessions group related recordings:

```python
# Start a session
session = engine.start_session(name="Training Session", user_id="user-123")

# All recordings are associated with this session
engine.record("accuracy", 0.8)
engine.record("accuracy", 0.85)
engine.record("accuracy", 0.9)

# End session
engine.end_session()

# Query by session
session_data = engine.query(session_id=session.session_id)

# Get session summary
print(session.get_summary())
# {
#     "session_id": "...",
#     "duration_seconds": 300,
#     "total_datapoints": 3,
#     "metrics_count": {"accuracy": 3}
# }
```

### Context Manager

```python
with engine.start_session(name="Game Round", user_id="player-1") as session:
    engine.record("score", 100)
    engine.record("score", 150)
    # Session automatically ends when block exits
```

---

## Tutorial 5: Querying Data

### Basic Queries

```python
# All data for a metric
datapoints = engine.query(metric_name="accuracy")

# Filter by dimension
datapoints = engine.query(
    metric_name="accuracy",
    dimensions={"difficulty": "hard"}
)

# Filter by time
from datetime import datetime, timedelta
datapoints = engine.query(
    metric_name="accuracy",
    from_time=datetime.now() - timedelta(days=7),
    to_time=datetime.now(),
)

# Filter by user
datapoints = engine.query(user_id="user-123")

# Limit results
datapoints = engine.query(metric_name="accuracy", limit=100)
```

### Just Get Values

```python
values = engine.get_values("accuracy")  # → [0.85, 0.72, 0.91, ...]
```

---

## Tutorial 6: Aggregation

### Simple Aggregation

```python
# Uses metric's default aggregation (usually mean)
avg = engine.aggregate("accuracy")

# Specify aggregation type
total = engine.aggregate("sales", aggregation="sum")
median = engine.aggregate("response_time", aggregation="median")
p95 = engine.aggregate("response_time", aggregation="percentile")  # 95th percentile
```

### Grouped Aggregation

```python
# Single dimension
by_region = engine.aggregate("sales", group_by=["region"])
# {"north": 1500, "south": 2000, "west": 1200}

# Multiple dimensions
by_region_product = engine.aggregate("sales", group_by=["region", "product"])
# {("north", "widget"): 500, ("north", "gadget"): 1000, ...}
```

### Time-Based Aggregation

```python
from metrics_framework.aggregators import TemporalAggregator

aggregator = TemporalAggregator()
datapoints = engine.query(metric_name="accuracy")

# Aggregate by hour
hourly = aggregator.aggregate_by_interval(datapoints, interval="hour")

# Aggregate by day
daily = aggregator.aggregate_by_interval(datapoints, interval="day")

# Available intervals: minute, hour, day, week, month
```

---

## Tutorial 7: Assessment

Assessment grades your metrics against thresholds:

```python
# Assess all metrics
report = engine.assess()

# Assess specific metric
report = engine.assess(metric_name="accuracy")

# Assess for specific user
report = engine.assess(user_id="user-123")
```

### Assessment Output

```python
{
    "timestamp": "2024-12-13T10:30:00",
    "metrics": {
        "accuracy": {
            "value": 0.75,
            "count": 100,
            "level": "good",           # Assessment grade
            "higher_is_better": True,
            "min": 0.45,
            "max": 0.98,
            "target": 0.8,             # If defined
        },
        "response_time": {
            "value": 2.3,
            "level": "average",
            "higher_is_better": False,
            ...
        }
    }
}
```

### Assessment Levels

| Level | Meaning | Typical Action |
|-------|---------|----------------|
| `excellent` | Exceptional | Celebrate! |
| `good` | Above average | Keep it up |
| `average` | Acceptable | Room for improvement |
| `below_average` | Needs work | Focus here |
| `poor` | Concerning | Priority attention |
| `critical` | Severe | Immediate action |

---

## Tutorial 8: Visualization

The framework generates data structures ready for charting libraries.

### Heatmap

```python
heatmap_data = engine.get_heatmap_data(
    metric_name="accuracy",
    x_dimension="feature",
    y_dimension="difficulty",
)
```

Output:
```python
{
    "type": "heatmap",
    "x_labels": ["rotation", "mirror", "assembly"],
    "y_labels": ["easy", "medium", "hard"],
    "cells": [
        {"x": 0, "y": 0, "value": 0.92, "color": "#0571b0", "count": 45},
        {"x": 0, "y": 1, "value": 0.78, "color": "#92c5de", "count": 38},
        {"x": 0, "y": 2, "value": 0.61, "color": "#f7f7f7", "count": 22},
        ...
    ],
    "config": {
        "color_thresholds": [...],
        "show_values": True,
    }
}
```

### Bar/Line Chart

```python
chart_data = engine.get_chart_data(
    metric_name="accuracy",
    chart_type="bar",
    group_by="feature",
)
```

Output (Chart.js compatible):
```python
{
    "type": "bar",
    "data": {
        "labels": ["rotation", "mirror", "assembly"],
        "datasets": [{
            "label": "accuracy",
            "data": [0.85, 0.72, 0.68],
            "backgroundColor": ["#0571b0", "#92c5de", "#f7f7f7"],
        }]
    },
    "options": {...}
}
```

### Trend

```python
trend_data = engine.get_trend_data(
    metric_name="accuracy",
    interval="day",
)
```

Output:
```python
{
    "type": "trend",
    "data": {
        "labels": ["2024-12-01", "2024-12-02", ...],
        "datasets": [
            {"label": "Actual", "data": [0.75, 0.78, 0.82, ...]},
            {"label": "Moving Avg", "data": [0.75, 0.76, 0.78, ...]},
        ]
    },
    "trend_analysis": {
        "direction": "improving",
        "slope": 0.02,
        "change_percent": 12.5,
    }
}
```

---

## Tutorial 9: Collectors (Advanced)

Collectors are specialized helpers for common patterns:

### Counter

```python
from metrics_framework.collectors import CounterCollector

counter = CounterCollector("page_views", engine=engine)
counter.inc()        # +1
counter.inc()        # +1
counter.increment(5) # +5
print(counter.get_value())  # 7
```

### Timer

```python
from metrics_framework.collectors import TimerCollector

timer = TimerCollector("request_duration", engine=engine)

# Method 1: Context manager
with timer.time():
    do_something()

# Method 2: Decorator
@timer.timed()
def my_function():
    pass

# Method 3: Manual
timer.start()
do_something()
timer.stop()

# Stats
print(timer.mean)    # Average duration
print(timer.median)  # Median duration
print(timer.p95)     # 95th percentile
```

### Success/Failure Tracker

```python
from metrics_framework.collectors import SuccessFailureCounter

tracker = SuccessFailureCounter("api_calls", engine=engine)
tracker.record_success()
tracker.record_failure()
tracker.record_success()

print(tracker.success_rate)  # 0.667
print(tracker.total)         # 3
```

### Composite (Multi-dimensional)

```python
from metrics_framework.collectors import CompositeCollector

collector = CompositeCollector("performance", engine=engine)

# Record attempts with features
collector.record_attempt(
    correct=True,
    response_time=2.5,
    features={"difficulty": "hard", "type": "math"}
)

# Get weak areas
weak = collector.get_weak_areas(min_attempts=5, max_success_rate=0.5)
# [{"feature_name": "type", "feature_value": "geometry", "success_rate": 0.35}, ...]
```

---

## Tutorial 10: Building a Domain Integration

For your specific application, create a wrapper class:

```python
from metrics_framework import MetricsEngine, MetricDefinition
from metrics_framework.storage import SQLAlchemyStorage

class MyAppMetrics:
    """Metrics for my specific application."""

    def __init__(self, db_url="sqlite:///metrics.db"):
        self._engine = MetricsEngine(
            storage_backend=SQLAlchemyStorage(db_url)
        )
        self._setup_metrics()

    def _setup_metrics(self):
        """Define all metrics for this application."""
        self._engine.register_metrics([
            MetricDefinition(
                name="task_completion",
                metric_type="percentage",
                higher_is_better=True,
            ),
            MetricDefinition(
                name="error_count",
                metric_type="counter",
                higher_is_better=False,
            ),
            # Add more metrics...
        ])

    # Domain-specific methods
    def record_task_result(self, success: bool, task_type: str, user_id: str):
        self._engine.record(
            "task_completion",
            1.0 if success else 0.0,
            dimensions={"task_type": task_type},
            user_id=user_id,
        )

    def record_error(self, error_type: str, endpoint: str):
        self._engine.record(
            "error_count",
            1,
            dimensions={"error_type": error_type, "endpoint": endpoint},
        )

    def get_dashboard(self):
        return {
            "task_completion_rate": self._engine.aggregate("task_completion"),
            "total_errors": self._engine.aggregate("error_count", aggregation="sum"),
            "by_task_type": self._engine.aggregate("task_completion", group_by=["task_type"]),
            "assessment": self._engine.assess(),
        }
```

---

# Part 2: Specification

## Data Structures

### DataPoint

The fundamental unit of stored data.

```python
@dataclass
class DataPoint:
    id: str                      # Unique identifier (UUID)
    metric_name: str             # Name of the metric
    value: float                 # The recorded value
    timestamp: datetime          # When recorded
    dimensions: Dict[str, Any]   # Categorical attributes
    metadata: Dict[str, Any]     # Additional data (not for grouping)
    session_id: Optional[str]    # Associated session
    user_id: Optional[str]       # Associated user
```

**JSON Representation:**
```json
{
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "metric_name": "accuracy",
    "value": 0.85,
    "timestamp": "2024-12-13T10:30:00.000Z",
    "dimensions": {
        "difficulty": "hard",
        "category": "math"
    },
    "metadata": {
        "response_time": 2.5
    },
    "session_id": "session-123",
    "user_id": "user-456"
}
```

### MetricDefinition

Configuration for a metric type.

```python
@dataclass
class MetricDefinition:
    name: str                           # Unique metric name
    metric_type: MetricType             # counter, gauge, histogram, timer, percentage, rate, boolean
    description: str                    # Human-readable description
    unit: str                           # Unit of measurement (%, seconds, USD, etc.)
    aggregation: AggregationType        # Default aggregation method
    higher_is_better: bool              # Direction for assessment
    min_value: Optional[float]          # Minimum valid value
    max_value: Optional[float]          # Maximum valid value
    thresholds: ThresholdConfig         # Assessment thresholds
    target_value: Optional[float]       # Target/goal value
    dimensions: List[DimensionDef]      # Expected dimensions
    required_dimensions: List[str]      # Required dimension names
    group: str                          # Grouping for organization
    tags: List[str]                     # Tags for filtering
    metadata: Dict[str, Any]            # Additional configuration
```

**JSON Representation:**
```json
{
    "name": "accuracy",
    "metric_type": "percentage",
    "description": "Percentage of correct responses",
    "unit": "%",
    "aggregation": "mean",
    "higher_is_better": true,
    "min_value": 0,
    "max_value": 1,
    "thresholds": {
        "excellent": 0.9,
        "good": 0.7,
        "average": 0.5,
        "below_average": 0.3,
        "poor": 0.1
    },
    "target_value": 0.85,
    "dimensions": [
        {"name": "difficulty", "values": ["easy", "medium", "hard"]},
        {"name": "category", "values": null}
    ],
    "required_dimensions": [],
    "group": "performance",
    "tags": ["core", "user-facing"]
}
```

### ThresholdConfig

Defines assessment grade boundaries.

```python
@dataclass
class ThresholdConfig:
    excellent: float      # >= this is "excellent"
    good: float           # >= this is "good"
    average: float        # >= this is "average"
    below_average: float  # >= this is "below_average"
    poor: float           # >= this is "poor"
                          # < poor is "critical"
```

**Note:** Values are on a normalized 0-1 scale. The framework normalizes actual values using `min_value` and `max_value` before comparison.

### DimensionDefinition

Describes an expected dimension.

```python
@dataclass
class DimensionDefinition:
    name: str                    # Dimension name
    description: str             # Human-readable description
    values: Optional[List[Any]]  # Allowed values (None = any)
    default_value: Optional[Any] # Default if not provided
    is_required: bool            # Must be provided
    group: Optional[str]         # For grouping related dimensions
```

---

## Enumerations

### MetricType

```python
class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    PERCENTAGE = "percentage"
    RATE = "rate"
    BOOLEAN = "boolean"
```

### AggregationType

```python
class AggregationType(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    FIRST = "first"
    LAST = "last"
    STD = "std"
    VARIANCE = "variance"
    PERCENTILE = "percentile"
    MODE = "mode"
    RATE = "rate"
```

### AssessmentLevel

```python
class AssessmentLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    BELOW_AVERAGE = "below_average"
    POOR = "poor"
    CRITICAL = "critical"
```

### TrendDirection

```python
class TrendDirection(str, Enum):
    STRONGLY_IMPROVING = "strongly_improving"
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    STRONGLY_DECLINING = "strongly_declining"
```

---

## API Specification

### MetricsEngine

#### Constructor

```python
MetricsEngine(
    storage_backend: Optional[StorageBackend] = None,  # Default: MemoryStorage
    auto_persist: bool = True,
)
```

#### Metric Registration

```python
# Register single metric
register_metric(metric: MetricDefinition) -> MetricsEngine

# Register multiple metrics
register_metrics(metrics: List[MetricDefinition]) -> MetricsEngine

# Get metric definition
get_metric(name: str) -> Optional[MetricDefinition]

# List all registered metrics
list_metrics() -> List[str]
```

#### Recording

```python
# Record a single value
record(
    metric_name: str,
    value: Any,
    dimensions: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: Optional[datetime] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> DataPoint

# Record multiple values
record_batch(
    records: List[Dict[str, Any]],  # Each dict has metric_name, value, etc.
) -> List[DataPoint]
```

#### Sessions

```python
# Start session
start_session(
    name: str = "",
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MetricSession

# End session
end_session(
    session_id: Optional[str] = None,  # Default: active session
) -> Optional[MetricSession]

# Get session
get_session(session_id: str) -> Optional[MetricSession]

# Get active session
active_session -> Optional[MetricSession]  # Property
```

#### Querying

```python
# Query datapoints
query(
    metric_name: Optional[str] = None,
    dimensions: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    from_time: Optional[datetime] = None,
    to_time: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> List[DataPoint]

# Get just values
get_values(
    metric_name: str,
    **query_params,
) -> List[float]
```

#### Aggregation

```python
# Aggregate values
aggregate(
    metric_name: str,
    aggregation: Optional[AggregationType] = None,  # Default: metric's default
    group_by: Optional[List[str]] = None,
    **query_params,
) -> Union[float, Dict[tuple, float]]
```

#### Assessment

```python
# Generate assessment report
assess(
    metric_name: Optional[str] = None,  # None = all metrics
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]
```

#### Visualization

```python
# Heatmap data
get_heatmap_data(
    metric_name: str,
    x_dimension: str,
    y_dimension: str,
    aggregation: Optional[AggregationType] = None,
    **query_params,
) -> Dict[str, Any]

# Chart data
get_chart_data(
    metric_name: str,
    chart_type: str = "bar",  # bar, line, pie, doughnut, scatter
    group_by: Optional[str] = None,
    **query_params,
) -> Dict[str, Any]

# Trend data
get_trend_data(
    metric_name: str,
    interval: str = "day",  # minute, hour, day, week, month
    **query_params,
) -> Dict[str, Any]
```

#### Event Hooks

```python
# Called when a datapoint is recorded
on_record(callback: Callable[[DataPoint], None]) -> MetricsEngine

# Called when session starts
on_session_start(callback: Callable[[MetricSession], None]) -> MetricsEngine

# Called when session ends
on_session_end(callback: Callable[[MetricSession], None]) -> MetricsEngine
```

#### Persistence

```python
# Set storage backend
set_storage(backend: StorageBackend) -> MetricsEngine

# Flush to storage
persist() -> None

# Clear all data
clear(confirm: bool = False) -> None  # Must pass confirm=True

# Export data
export(format: str = "json") -> Any
```

---

## Storage Backend Interface

All storage backends implement:

```python
class StorageBackend(ABC):
    @abstractmethod
    def store(self, datapoint: DataPoint) -> None:
        """Store a single datapoint."""

    @abstractmethod
    def store_batch(self, datapoints: List[DataPoint]) -> None:
        """Store multiple datapoints."""

    @abstractmethod
    def query(self, filters: Dict[str, Any], limit: Optional[int]) -> List[DataPoint]:
        """Query datapoints matching filters."""

    @abstractmethod
    def delete(self, filters: Dict[str, Any]) -> int:
        """Delete matching datapoints. Returns count deleted."""

    @abstractmethod
    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count matching datapoints."""

    @abstractmethod
    def clear(self) -> None:
        """Delete all datapoints."""

    def flush(self) -> None:
        """Flush buffered data (optional)."""

    def close(self) -> None:
        """Close connections (optional)."""
```

### Query Filters

Supported filter keys:

| Key | Type | Description |
|-----|------|-------------|
| `metric_name` | str | Exact metric name match |
| `session_id` | str | Exact session ID match |
| `user_id` | str | Exact user ID match |
| `timestamp_from` | datetime | Minimum timestamp (inclusive) |
| `timestamp_to` | datetime | Maximum timestamp (inclusive) |
| `{dimension_name}` | Any | Exact dimension value match |

---

## Visualization Output Formats

### Heatmap

```python
{
    "type": "heatmap",
    "x_dimension": str,          # Dimension name for X axis
    "y_dimension": str,          # Dimension name for Y axis
    "x_labels": List[str],       # X axis labels
    "y_labels": List[str],       # Y axis labels
    "cells": [
        {
            "x": int,            # X index
            "y": int,            # Y index
            "x_label": str,      # X label
            "y_label": str,      # Y label
            "value": float,      # Aggregated value
            "count": int,        # Number of datapoints
            "color": str,        # Hex color code
            "display_value": str # Formatted value
        },
        ...
    ],
    "value_range": {
        "min": float,
        "max": float
    },
    "aggregation": str,
    "config": {
        "color_thresholds": List[Tuple[float, str]],
        "show_values": bool,
        "show_counts": bool,
        "cell_width": int,
        "cell_height": int
    },
    "total_datapoints": int
}
```

### Chart (Chart.js Compatible)

```python
{
    "type": str,                 # bar, line, pie, doughnut, scatter
    "data": {
        "labels": List[str],
        "datasets": [
            {
                "label": str,
                "data": List[float],
                "backgroundColor": Union[str, List[str]],
                "borderColor": Union[str, List[str]],
                "fill": bool,           # For line charts
                "tension": float,       # Line smoothing
            },
            ...
        ]
    },
    "options": {
        "responsive": bool,
        "plugins": {
            "legend": {"display": bool},
            "title": {"display": bool, "text": str}
        },
        "scales": {
            "x": {...},
            "y": {...}
        }
    }
}
```

### Trend

```python
{
    "type": "trend",
    "data": {
        "labels": List[str],     # Timestamps
        "datasets": [
            {"label": "Actual", "data": List[float], ...},
            {"label": "Moving Avg", "data": List[float], ...},
            {"label": "EMA", "data": List[float], ...}
        ]
    },
    "trend_analysis": {
        "trend": str,            # improving, stable, declining
        "slope": float,
        "change_percent": float,
        "r_squared": float,      # Fit quality
        "first_value": float,
        "last_value": float,
        "data_points": int
    },
    "options": {...}
}
```

### Radar

```python
{
    "type": "radar",
    "data": {
        "labels": List[str],     # Dimension names
        "datasets": [
            {
                "label": str,
                "data": List[float],
                "backgroundColor": str,  # With alpha
                "borderColor": str,
                "pointBackgroundColor": str
            },
            ...
        ]
    },
    "options": {
        "scales": {
            "r": {
                "suggestedMin": 0,
                "suggestedMax": float
            }
        }
    }
}
```

---

## Assessment Output Format

```python
{
    "timestamp": str,            # ISO format
    "session_id": Optional[str],
    "user_id": Optional[str],
    "metrics": {
        "{metric_name}": {
            "value": float,          # Aggregated value
            "count": int,            # Number of datapoints
            "level": str,            # Assessment level
            "higher_is_better": bool,
            "min": float,
            "max": float,
            "target": Optional[float]
        },
        ...
    }
}
```

---

## Configuration File Format (Optional)

For storing metric definitions in a file:

```yaml
# metrics_config.yaml
version: "1.0"

metrics:
  - name: accuracy
    type: percentage
    description: Percentage of correct responses
    unit: "%"
    aggregation: mean
    higher_is_better: true
    min_value: 0
    max_value: 1
    thresholds:
      excellent: 0.9
      good: 0.7
      average: 0.5
      below_average: 0.3
      poor: 0.1
    target: 0.85
    dimensions:
      - name: difficulty
        values: [easy, medium, hard]
      - name: category
        required: false

  - name: response_time
    type: timer
    description: Time to respond
    unit: seconds
    aggregation: median
    higher_is_better: false
    min_value: 0
    thresholds:
      excellent: 0.2   # Inverted because lower is better
      good: 0.4
      average: 0.6
      below_average: 0.8
      poor: 0.9
```

---

## Error Handling

### Validation Errors

```python
# Invalid value for metric
engine.record("accuracy", 1.5)  # max_value is 1.0
# Raises: ValueError("Value 1.5 above maximum 1.0")

# Missing required dimension
engine.record("metric_with_required_dim", 0.5, dimensions={})
# Raises: ValueError("Missing required dimension: category")

# Invalid dimension value
engine.record("accuracy", 0.5, dimensions={"difficulty": "extreme"})
# Raises: ValueError("Invalid value for dimension difficulty: extreme")
```

### Query Errors

```python
# Unknown metric (no error, just returns empty)
engine.query(metric_name="nonexistent")
# Returns: []

# Invalid aggregation
engine.aggregate("accuracy", aggregation="invalid")
# Raises: ValueError("Unknown aggregation type: invalid")
```

---

## Thread Safety

- `MetricsEngine` uses `threading.RLock` for all operations
- Safe to record from multiple threads
- Safe to query while recording
- Sessions are per-engine (not per-thread)

---

## Performance Guidelines

| Data Volume | Recommended Storage | Notes |
|-------------|---------------------|-------|
| < 10K datapoints | MemoryStorage | Fast, but lost on restart |
| 10K - 100K | JSONStorage | Persistent, single file |
| 100K - 1M | SQLAlchemyStorage (SQLite) | Good balance |
| > 1M | SQLAlchemyStorage (PostgreSQL) | Scalable, add indexes |

### Optimization Tips

1. **Batch recording**: Use `record_batch()` for bulk inserts
2. **Limit queries**: Always use `limit` for large datasets
3. **Index dimensions**: In SQL, index frequently-queried dimensions
4. **Periodic cleanup**: Delete old data periodically
5. **Aggregation caching**: Cache aggregation results for dashboards

---

## Version History

### 1.0.0 (Initial Release)
- Core engine with sessions
- 7 metric types
- 12 aggregation types
- 4 collector types
- 3 aggregator types
- 4 assessor types
- 4 visualizer types
- 3 storage backends
