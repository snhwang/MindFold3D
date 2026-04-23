"""
Metric Session
==============

Manages a metric collection session with scoped data and lifecycle.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import uuid

from .datapoint import DataPoint

if TYPE_CHECKING:
    from .engine import MetricsEngine


@dataclass
class MetricSession:
    """A metric collection session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    name: str = ""
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _engine: Optional["MetricsEngine"] = field(default=None, repr=False)
    _datapoints: List[DataPoint] = field(default_factory=list, repr=False)
    _is_active: bool = True

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self._is_active

    @property
    def duration(self) -> float:
        """Get session duration in seconds."""
        end = self.ended_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()

    @property
    def datapoints(self) -> List[DataPoint]:
        """Get all datapoints in this session."""
        return self._datapoints.copy()

    def record(
        self,
        metric_name: str,
        value: Any,
        dimensions: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> DataPoint:
        """Record a metric value in this session."""
        if not self._is_active:
            raise RuntimeError("Cannot record to inactive session")

        datapoint = DataPoint(
            metric_name=metric_name,
            value=value,
            timestamp=timestamp or datetime.utcnow(),
            dimensions=dimensions or {},
            metadata=metadata or {},
            session_id=self.session_id,
            user_id=self.user_id,
        )

        self._datapoints.append(datapoint)

        # Also record to engine if attached
        if self._engine:
            self._engine._store_datapoint(datapoint)

        return datapoint

    def get_metrics(self, metric_name: Optional[str] = None) -> List[DataPoint]:
        """Get datapoints, optionally filtered by metric name."""
        if metric_name:
            return [dp for dp in self._datapoints if dp.metric_name == metric_name]
        return self._datapoints.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get session summary statistics."""
        metrics_count: Dict[str, int] = {}
        for dp in self._datapoints:
            metrics_count[dp.metric_name] = metrics_count.get(dp.metric_name, 0) + 1

        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "name": self.name,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration,
            "is_active": self._is_active,
            "total_datapoints": len(self._datapoints),
            "metrics_count": metrics_count,
            "metadata": self.metadata,
        }

    def end(self) -> "MetricSession":
        """End the session."""
        if self._is_active:
            self.ended_at = datetime.utcnow()
            self._is_active = False
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "name": self.name,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "metadata": self.metadata,
            "is_active": self._is_active,
            "datapoints": [dp.to_dict() for dp in self._datapoints],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], engine: Optional["MetricsEngine"] = None) -> "MetricSession":
        """Create session from dictionary."""
        started_at = data.get("started_at")
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at)
        elif started_at is None:
            started_at = datetime.utcnow()

        ended_at = data.get("ended_at")
        if isinstance(ended_at, str):
            ended_at = datetime.fromisoformat(ended_at)

        session = cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            user_id=data.get("user_id"),
            name=data.get("name", ""),
            started_at=started_at,
            ended_at=ended_at,
            metadata=data.get("metadata", {}),
            _engine=engine,
            _is_active=data.get("is_active", ended_at is None),
        )

        # Restore datapoints
        for dp_data in data.get("datapoints", []):
            session._datapoints.append(DataPoint.from_dict(dp_data))

        return session

    def __enter__(self) -> "MetricSession":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - end session."""
        self.end()
