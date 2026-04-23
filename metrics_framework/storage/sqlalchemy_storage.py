"""SQLAlchemy storage backend."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import StorageBackend
from ..core.datapoint import DataPoint

# SQLAlchemy imports are optional
try:
    from sqlalchemy import create_engine, Column, String, Float, DateTime, Text, Integer
    from sqlalchemy.orm import sessionmaker, declarative_base
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()

    class MetricDataPoint(Base):
        """SQLAlchemy model for metric datapoints."""
        __tablename__ = "metric_datapoints"

        id = Column(String(36), primary_key=True)
        metric_name = Column(String(255), index=True, nullable=False)
        value = Column(Float, nullable=False)
        timestamp = Column(DateTime, index=True, nullable=False)
        session_id = Column(String(36), index=True, nullable=True)
        user_id = Column(String(255), index=True, nullable=True)
        dimensions = Column(Text, nullable=True)  # JSON string
        metadata = Column(Text, nullable=True)  # JSON string

        def to_datapoint(self) -> DataPoint:
            """Convert to DataPoint."""
            return DataPoint(
                id=self.id,
                metric_name=self.metric_name,
                value=self.value,
                timestamp=self.timestamp,
                session_id=self.session_id,
                user_id=self.user_id,
                dimensions=json.loads(self.dimensions) if self.dimensions else {},
                metadata=json.loads(self.metadata) if self.metadata else {},
            )

        @classmethod
        def from_datapoint(cls, dp: DataPoint) -> "MetricDataPoint":
            """Create from DataPoint."""
            return cls(
                id=dp.id,
                metric_name=dp.metric_name,
                value=dp.value,
                timestamp=dp.timestamp,
                session_id=dp.session_id,
                user_id=dp.user_id,
                dimensions=json.dumps(dp.dimensions) if dp.dimensions else None,
                metadata=json.dumps(dp.metadata) if dp.metadata else None,
            )


class SQLAlchemyStorage(StorageBackend):
    """
    SQLAlchemy storage backend.

    Provides persistent storage using any SQLAlchemy-supported database.
    """

    def __init__(
        self,
        connection_string: str = "sqlite:///metrics.db",
        echo: bool = False,
    ):
        """
        Initialize SQLAlchemy storage.

        Args:
            connection_string: Database connection string
            echo: Whether to echo SQL queries
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy is required for SQLAlchemyStorage. "
                "Install it with: pip install sqlalchemy"
            )

        self._engine = create_engine(connection_string, echo=echo)
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)

    def store(self, datapoint: DataPoint) -> None:
        """Store a single datapoint."""
        session = self._Session()
        try:
            model = MetricDataPoint.from_datapoint(datapoint)
            session.add(model)
            session.commit()
        finally:
            session.close()

    def store_batch(self, datapoints: List[DataPoint]) -> None:
        """Store multiple datapoints."""
        session = self._Session()
        try:
            models = [MetricDataPoint.from_datapoint(dp) for dp in datapoints]
            session.add_all(models)
            session.commit()
        finally:
            session.close()

    def query(
        self,
        filters: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> List[DataPoint]:
        """Query stored datapoints."""
        session = self._Session()
        try:
            query = session.query(MetricDataPoint)

            # Apply filters
            if "metric_name" in filters:
                query = query.filter(MetricDataPoint.metric_name == filters["metric_name"])
            if "session_id" in filters:
                query = query.filter(MetricDataPoint.session_id == filters["session_id"])
            if "user_id" in filters:
                query = query.filter(MetricDataPoint.user_id == filters["user_id"])
            if "timestamp_from" in filters:
                query = query.filter(MetricDataPoint.timestamp >= filters["timestamp_from"])
            if "timestamp_to" in filters:
                query = query.filter(MetricDataPoint.timestamp <= filters["timestamp_to"])

            # Order by timestamp descending
            query = query.order_by(MetricDataPoint.timestamp.desc())

            if limit:
                query = query.limit(limit)

            results = query.all()

            # Filter by dimensions (needs post-processing)
            datapoints = [m.to_datapoint() for m in results]

            # Apply dimension filters
            dimension_filters = {
                k: v for k, v in filters.items()
                if k not in ["metric_name", "session_id", "user_id", "timestamp_from", "timestamp_to"]
            }
            if dimension_filters:
                datapoints = [
                    dp for dp in datapoints
                    if all(dp.dimensions.get(k) == v for k, v in dimension_filters.items())
                ]

            return datapoints
        finally:
            session.close()

    def delete(self, filters: Dict[str, Any]) -> int:
        """Delete datapoints matching filters."""
        session = self._Session()
        try:
            query = session.query(MetricDataPoint)

            if "metric_name" in filters:
                query = query.filter(MetricDataPoint.metric_name == filters["metric_name"])
            if "session_id" in filters:
                query = query.filter(MetricDataPoint.session_id == filters["session_id"])
            if "user_id" in filters:
                query = query.filter(MetricDataPoint.user_id == filters["user_id"])

            count = query.delete(synchronize_session=False)
            session.commit()
            return count
        finally:
            session.close()

    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count datapoints matching filters."""
        session = self._Session()
        try:
            query = session.query(MetricDataPoint)

            if filters:
                if "metric_name" in filters:
                    query = query.filter(MetricDataPoint.metric_name == filters["metric_name"])
                if "session_id" in filters:
                    query = query.filter(MetricDataPoint.session_id == filters["session_id"])
                if "user_id" in filters:
                    query = query.filter(MetricDataPoint.user_id == filters["user_id"])

            return query.count()
        finally:
            session.close()

    def clear(self) -> None:
        """Clear all stored data."""
        session = self._Session()
        try:
            session.query(MetricDataPoint).delete()
            session.commit()
        finally:
            session.close()

    def close(self) -> None:
        """Close the engine."""
        self._engine.dispose()
