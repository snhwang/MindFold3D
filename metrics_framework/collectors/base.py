"""Base metric collector class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.engine import MetricsEngine
    from ..core.datapoint import DataPoint


@dataclass
class CollectorState:
    """State for a metric collector."""
    metric_name: str
    values: List[Any] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    dimensions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_value: Optional[Any] = None
    last_timestamp: Optional[datetime] = None

    def add(self, value: Any, timestamp: Optional[datetime] = None) -> None:
        """Add a value to the collector state."""
        ts = timestamp or datetime.utcnow()
        self.values.append(value)
        self.timestamps.append(ts)
        self.last_value = value
        self.last_timestamp = ts

    def clear(self) -> None:
        """Clear the collector state."""
        self.values.clear()
        self.timestamps.clear()
        self.last_value = None
        self.last_timestamp = None


class MetricCollector(ABC):
    """Abstract base class for metric collectors."""

    def __init__(
        self,
        metric_name: str,
        engine: Optional["MetricsEngine"] = None,
        dimensions: Optional[Dict[str, Any]] = None,
        auto_record: bool = True,
    ):
        """
        Initialize the collector.

        Args:
            metric_name: Name of the metric to collect
            engine: MetricsEngine to record to
            dimensions: Default dimensions for this collector
            auto_record: Whether to automatically record to engine
        """
        self.metric_name = metric_name
        self._engine = engine
        self._dimensions = dimensions or {}
        self._auto_record = auto_record
        self._state = CollectorState(metric_name=metric_name, dimensions=self._dimensions)

    @abstractmethod
    def collect(self, value: Any, **kwargs) -> Any:
        """
        Collect a value.

        Args:
            value: The value to collect
            **kwargs: Additional collection parameters

        Returns:
            The processed value
        """
        pass

    @abstractmethod
    def get_value(self) -> Any:
        """
        Get the current collector value.

        Returns:
            The current value
        """
        pass

    def record(
        self,
        value: Any,
        dimensions: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Optional["DataPoint"]:
        """
        Record a value to the engine.

        Args:
            value: The value to record
            dimensions: Additional dimensions
            metadata: Additional metadata
            timestamp: Recording timestamp

        Returns:
            The created DataPoint if auto_record is enabled
        """
        processed_value = self.collect(value)

        if self._auto_record and self._engine:
            merged_dimensions = {**self._dimensions, **(dimensions or {})}
            return self._engine.record(
                metric_name=self.metric_name,
                value=processed_value,
                dimensions=merged_dimensions,
                metadata=metadata,
                timestamp=timestamp,
            )
        return None

    def reset(self) -> None:
        """Reset the collector state."""
        self._state.clear()

    @property
    def state(self) -> CollectorState:
        """Get the collector state."""
        return self._state

    def set_engine(self, engine: "MetricsEngine") -> "MetricCollector":
        """Set the metrics engine."""
        self._engine = engine
        return self

    def set_dimensions(self, dimensions: Dict[str, Any]) -> "MetricCollector":
        """Set default dimensions."""
        self._dimensions.update(dimensions)
        return self

    def __enter__(self) -> "MetricCollector":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        pass
