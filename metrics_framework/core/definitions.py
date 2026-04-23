"""
Metric and Dimension Definitions
================================

Provides structured definitions for metrics and their dimensions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class MetricType(str, Enum):
    """Supported metric types."""
    COUNTER = "counter"  # Monotonically increasing value
    GAUGE = "gauge"  # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution of values
    TIMER = "timer"  # Time-based measurements
    PERCENTAGE = "percentage"  # 0-100 or 0-1 values
    RATE = "rate"  # Value per time unit
    BOOLEAN = "boolean"  # True/False outcomes


class AggregationType(str, Enum):
    """Supported aggregation methods."""
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    FIRST = "first"
    LAST = "last"
    STD = "std"  # Standard deviation
    VARIANCE = "variance"
    PERCENTILE = "percentile"
    MODE = "mode"
    RATE = "rate"  # Change per time unit


class AssessmentLevel(str, Enum):
    """Performance assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    BELOW_AVERAGE = "below_average"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class ThresholdConfig:
    """Configuration for assessment thresholds."""
    excellent: float = 0.9
    good: float = 0.7
    average: float = 0.5
    below_average: float = 0.3
    # Below below_average is poor, below poor_threshold is critical
    poor: float = 0.1

    def get_level(self, value: float, inverted: bool = False) -> AssessmentLevel:
        """Get assessment level for a value."""
        if inverted:
            value = 1.0 - value

        if value >= self.excellent:
            return AssessmentLevel.EXCELLENT
        elif value >= self.good:
            return AssessmentLevel.GOOD
        elif value >= self.average:
            return AssessmentLevel.AVERAGE
        elif value >= self.below_average:
            return AssessmentLevel.BELOW_AVERAGE
        elif value >= self.poor:
            return AssessmentLevel.POOR
        else:
            return AssessmentLevel.CRITICAL


@dataclass
class DimensionDefinition:
    """Definition of a metric dimension."""
    name: str
    description: str = ""
    values: Optional[List[Any]] = None  # Allowed values (None = any)
    default_value: Optional[Any] = None
    is_required: bool = False
    group: Optional[str] = None  # For grouping related dimensions

    def validate(self, value: Any) -> bool:
        """Validate a value against this dimension."""
        if self.values is not None:
            return value in self.values
        return True


@dataclass
class MetricDefinition:
    """Definition of a trackable metric."""
    name: str
    metric_type: Union[MetricType, str] = MetricType.GAUGE
    description: str = ""
    unit: str = ""

    # Aggregation settings
    aggregation: Union[AggregationType, str] = AggregationType.MEAN
    aggregation_params: Dict[str, Any] = field(default_factory=dict)

    # Assessment settings
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    higher_is_better: bool = True
    target_value: Optional[float] = None

    # Dimension settings
    dimensions: List[DimensionDefinition] = field(default_factory=list)
    required_dimensions: List[str] = field(default_factory=list)

    # Value constraints
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    # Metadata
    group: str = "default"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Transformation
    transform: Optional[Callable[[Any], float]] = None

    def __post_init__(self):
        """Convert string enums to actual enum values."""
        if isinstance(self.metric_type, str):
            self.metric_type = MetricType(self.metric_type)
        if isinstance(self.aggregation, str):
            self.aggregation = AggregationType(self.aggregation)

    def validate_value(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a metric value."""
        if self.transform:
            try:
                value = self.transform(value)
            except Exception as e:
                return False, f"Transform failed: {e}"

        if not isinstance(value, (int, float, bool)):
            return False, f"Value must be numeric, got {type(value)}"

        if self.min_value is not None and value < self.min_value:
            return False, f"Value {value} below minimum {self.min_value}"

        if self.max_value is not None and value > self.max_value:
            return False, f"Value {value} above maximum {self.max_value}"

        return True, None

    def validate_dimensions(self, dimensions: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate dimensions for a metric recording."""
        for req_dim in self.required_dimensions:
            if req_dim not in dimensions:
                return False, f"Missing required dimension: {req_dim}"

        dim_defs = {d.name: d for d in self.dimensions}
        for dim_name, dim_value in dimensions.items():
            if dim_name in dim_defs:
                if not dim_defs[dim_name].validate(dim_value):
                    return False, f"Invalid value for dimension {dim_name}: {dim_value}"

        return True, None

    def assess(self, value: float) -> AssessmentLevel:
        """Assess a value against thresholds."""
        # Normalize to 0-1 range if needed
        if self.max_value is not None and self.min_value is not None:
            normalized = (value - self.min_value) / (self.max_value - self.min_value)
        elif self.metric_type == MetricType.PERCENTAGE:
            normalized = value if value <= 1 else value / 100
        else:
            normalized = value

        return self.thresholds.get_level(normalized, not self.higher_is_better)


# Common metric definition templates
COMMON_METRICS = {
    "accuracy": MetricDefinition(
        name="accuracy",
        metric_type=MetricType.PERCENTAGE,
        description="Percentage of correct responses",
        unit="%",
        aggregation=AggregationType.MEAN,
        higher_is_better=True,
        min_value=0,
        max_value=1,
        thresholds=ThresholdConfig(excellent=0.9, good=0.7, average=0.5, below_average=0.3, poor=0.1),
    ),
    "response_time": MetricDefinition(
        name="response_time",
        metric_type=MetricType.TIMER,
        description="Time to respond to a question",
        unit="seconds",
        aggregation=AggregationType.MEAN,
        higher_is_better=False,
        min_value=0,
        thresholds=ThresholdConfig(excellent=0.2, good=0.4, average=0.6, below_average=0.8, poor=0.9),
    ),
    "completion_rate": MetricDefinition(
        name="completion_rate",
        metric_type=MetricType.PERCENTAGE,
        description="Percentage of tasks completed",
        unit="%",
        aggregation=AggregationType.MEAN,
        higher_is_better=True,
        min_value=0,
        max_value=1,
    ),
    "error_count": MetricDefinition(
        name="error_count",
        metric_type=MetricType.COUNTER,
        description="Number of errors",
        unit="count",
        aggregation=AggregationType.SUM,
        higher_is_better=False,
        min_value=0,
    ),
    "attempts": MetricDefinition(
        name="attempts",
        metric_type=MetricType.COUNTER,
        description="Number of attempts",
        unit="count",
        aggregation=AggregationType.SUM,
        min_value=0,
    ),
}
