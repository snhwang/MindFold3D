"""
DataPoint
=========

Represents a single metric recording with value, timestamp, and dimensions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import uuid


@dataclass
class DataPoint:
    """A single metric data point."""
    metric_name: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    dimensions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "metric_name": self.metric_name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "dimensions": self.dimensions,
            "metadata": self.metadata,
            "session_id": self.session_id,
            "user_id": self.user_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataPoint":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            metric_name=data["metric_name"],
            value=data["value"],
            timestamp=timestamp,
            dimensions=data.get("dimensions", {}),
            metadata=data.get("metadata", {}),
            session_id=data.get("session_id"),
            user_id=data.get("user_id"),
        )

    def matches_filter(self, filters: Dict[str, Any]) -> bool:
        """Check if this datapoint matches given filters."""
        for key, value in filters.items():
            if key == "metric_name" and self.metric_name != value:
                return False
            elif key == "session_id" and self.session_id != value:
                return False
            elif key == "user_id" and self.user_id != value:
                return False
            elif key == "timestamp_from" and self.timestamp < value:
                return False
            elif key == "timestamp_to" and self.timestamp > value:
                return False
            elif key in self.dimensions:
                if isinstance(value, list):
                    if self.dimensions[key] not in value:
                        return False
                elif self.dimensions[key] != value:
                    return False
        return True

    def get_dimension_key(self, dimension_names: list[str]) -> tuple:
        """Get a tuple key based on specified dimensions."""
        return tuple(self.dimensions.get(d) for d in dimension_names)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, DataPoint):
            return False
        return self.id == other.id


@dataclass
class AggregatedDataPoint:
    """An aggregated metric data point."""
    metric_name: str
    aggregation_type: str
    value: float
    count: int
    timestamp_from: datetime
    timestamp_to: datetime
    dimensions: Dict[str, Any] = field(default_factory=dict)
    raw_values: list[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def time_range(self) -> float:
        """Get the time range in seconds."""
        return (self.timestamp_to - self.timestamp_from).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "aggregation_type": self.aggregation_type,
            "value": self.value,
            "count": self.count,
            "timestamp_from": self.timestamp_from.isoformat(),
            "timestamp_to": self.timestamp_to.isoformat(),
            "dimensions": self.dimensions,
            "raw_values": self.raw_values,
            "metadata": self.metadata,
        }
