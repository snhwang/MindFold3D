"""
Metrics Engine
==============

The central component that orchestrates metric collection, storage, and analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Type, Union
import threading

from .definitions import (
    MetricDefinition,
    MetricType,
    AggregationType,
    AssessmentLevel,
    COMMON_METRICS,
)
from .datapoint import DataPoint, AggregatedDataPoint
from .session import MetricSession


class MetricsEngine:
    """Central metrics collection and analysis engine."""

    def __init__(
        self,
        storage_backend: Optional["StorageBackend"] = None,
        auto_persist: bool = True,
    ):
        """
        Initialize the metrics engine.

        Args:
            storage_backend: Backend for persisting metrics (defaults to MemoryStorage)
            auto_persist: Whether to automatically persist datapoints
        """
        self._metrics: Dict[str, MetricDefinition] = {}
        self._sessions: Dict[str, MetricSession] = {}
        self._active_session: Optional[MetricSession] = None
        self._storage: "StorageBackend" = storage_backend
        self._auto_persist = auto_persist
        self._lock = threading.RLock()

        # Lazy import to avoid circular dependency
        if self._storage is None:
            from ..storage.memory import MemoryStorage
            self._storage = MemoryStorage()

        # Event hooks
        self._on_record: List[Callable[[DataPoint], None]] = []
        self._on_session_start: List[Callable[[MetricSession], None]] = []
        self._on_session_end: List[Callable[[MetricSession], None]] = []

    # ==================== Metric Definition ====================

    def register_metric(self, metric: MetricDefinition) -> "MetricsEngine":
        """Register a metric definition."""
        with self._lock:
            self._metrics[metric.name] = metric
        return self

    def register_metrics(self, metrics: List[MetricDefinition]) -> "MetricsEngine":
        """Register multiple metric definitions."""
        for metric in metrics:
            self.register_metric(metric)
        return self

    def get_metric(self, name: str) -> Optional[MetricDefinition]:
        """Get a metric definition by name."""
        return self._metrics.get(name)

    def list_metrics(self) -> List[str]:
        """List all registered metric names."""
        return list(self._metrics.keys())

    def use_common_metric(self, name: str) -> "MetricsEngine":
        """Use a predefined common metric template."""
        if name in COMMON_METRICS:
            self.register_metric(COMMON_METRICS[name])
        else:
            raise ValueError(f"Unknown common metric: {name}")
        return self

    # ==================== Recording ====================

    def record(
        self,
        metric_name: str,
        value: Any,
        dimensions: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> DataPoint:
        """
        Record a metric value.

        Args:
            metric_name: Name of the metric
            value: The metric value
            dimensions: Dimensional attributes for the metric
            metadata: Additional metadata
            timestamp: Recording timestamp (defaults to now)
            user_id: User identifier
            session_id: Session identifier (or uses active session)

        Returns:
            The created DataPoint
        """
        with self._lock:
            # Validate against metric definition if registered
            metric_def = self._metrics.get(metric_name)
            if metric_def:
                # Validate value
                is_valid, error = metric_def.validate_value(value)
                if not is_valid:
                    raise ValueError(f"Invalid value for {metric_name}: {error}")

                # Validate dimensions
                if dimensions:
                    is_valid, error = metric_def.validate_dimensions(dimensions)
                    if not is_valid:
                        raise ValueError(f"Invalid dimensions for {metric_name}: {error}")

                # Transform value if needed
                if metric_def.transform:
                    value = metric_def.transform(value)

            # Determine session
            actual_session_id = session_id
            actual_user_id = user_id
            if self._active_session and not session_id:
                actual_session_id = self._active_session.session_id
                actual_user_id = actual_user_id or self._active_session.user_id

            datapoint = DataPoint(
                metric_name=metric_name,
                value=value,
                timestamp=timestamp or datetime.utcnow(),
                dimensions=dimensions or {},
                metadata=metadata or {},
                session_id=actual_session_id,
                user_id=actual_user_id,
            )

            # Store the datapoint
            self._store_datapoint(datapoint)

            # Fire event hooks
            for hook in self._on_record:
                try:
                    hook(datapoint)
                except Exception:
                    pass  # Don't let hooks break recording

            return datapoint

    def record_batch(
        self,
        records: List[Dict[str, Any]],
    ) -> List[DataPoint]:
        """
        Record multiple metrics at once.

        Args:
            records: List of dicts with metric_name, value, and optional dimensions/metadata

        Returns:
            List of created DataPoints
        """
        datapoints = []
        for record in records:
            dp = self.record(
                metric_name=record["metric_name"],
                value=record["value"],
                dimensions=record.get("dimensions"),
                metadata=record.get("metadata"),
                timestamp=record.get("timestamp"),
                user_id=record.get("user_id"),
                session_id=record.get("session_id"),
            )
            datapoints.append(dp)
        return datapoints

    def _store_datapoint(self, datapoint: DataPoint) -> None:
        """Store a datapoint to the backend."""
        if self._auto_persist and self._storage:
            self._storage.store(datapoint)

    # ==================== Sessions ====================

    def start_session(
        self,
        name: str = "",
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MetricSession:
        """Start a new metric collection session."""
        with self._lock:
            session = MetricSession(
                name=name,
                user_id=user_id,
                metadata=metadata or {},
                _engine=self,
            )
            self._sessions[session.session_id] = session
            self._active_session = session

            for hook in self._on_session_start:
                try:
                    hook(session)
                except Exception:
                    pass

            return session

    def end_session(self, session_id: Optional[str] = None) -> Optional[MetricSession]:
        """End a session."""
        with self._lock:
            target_id = session_id or (self._active_session.session_id if self._active_session else None)
            if not target_id:
                return None

            session = self._sessions.get(target_id)
            if session:
                session.end()

                for hook in self._on_session_end:
                    try:
                        hook(session)
                    except Exception:
                        pass

                if self._active_session and self._active_session.session_id == target_id:
                    self._active_session = None

            return session

    def get_session(self, session_id: str) -> Optional[MetricSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    @property
    def active_session(self) -> Optional[MetricSession]:
        """Get the current active session."""
        return self._active_session

    # ==================== Querying ====================

    def query(
        self,
        metric_name: Optional[str] = None,
        dimensions: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[DataPoint]:
        """
        Query stored datapoints.

        Args:
            metric_name: Filter by metric name
            dimensions: Filter by dimension values
            session_id: Filter by session
            user_id: Filter by user
            from_time: Filter from timestamp
            to_time: Filter to timestamp
            limit: Maximum number of results

        Returns:
            List of matching DataPoints
        """
        filters = {}
        if metric_name:
            filters["metric_name"] = metric_name
        if session_id:
            filters["session_id"] = session_id
        if user_id:
            filters["user_id"] = user_id
        if from_time:
            filters["timestamp_from"] = from_time
        if to_time:
            filters["timestamp_to"] = to_time
        if dimensions:
            filters.update(dimensions)

        return self._storage.query(filters, limit)

    def get_values(
        self,
        metric_name: str,
        **query_params,
    ) -> List[Any]:
        """Get just the values for a metric."""
        datapoints = self.query(metric_name=metric_name, **query_params)
        return [dp.value for dp in datapoints]

    # ==================== Aggregation ====================

    def aggregate(
        self,
        metric_name: str,
        aggregation: Optional[Union[AggregationType, str]] = None,
        group_by: Optional[List[str]] = None,
        **query_params,
    ) -> Union[float, Dict[tuple, float]]:
        """
        Aggregate metric values.

        Args:
            metric_name: Name of the metric to aggregate
            aggregation: Aggregation type (uses metric default if not specified)
            group_by: Dimension names to group by
            **query_params: Additional query parameters

        Returns:
            Aggregated value, or dict of grouped values
        """
        from ..aggregators.dimensional import DimensionalAggregator

        metric_def = self._metrics.get(metric_name)
        if not aggregation and metric_def:
            aggregation = metric_def.aggregation
        aggregation = aggregation or AggregationType.MEAN

        datapoints = self.query(metric_name=metric_name, **query_params)

        if group_by:
            aggregator = DimensionalAggregator(aggregation)
            return aggregator.aggregate_grouped(datapoints, group_by)
        else:
            from ..aggregators.temporal import TemporalAggregator
            aggregator = TemporalAggregator(aggregation)
            return aggregator.aggregate([dp.value for dp in datapoints])

    # ==================== Assessment ====================

    def assess(
        self,
        metric_name: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate assessment report for metrics.

        Args:
            metric_name: Specific metric to assess (or all if None)
            session_id: Filter by session
            user_id: Filter by user

        Returns:
            Assessment report dictionary
        """
        from ..assessors.threshold import ThresholdAssessor

        assessments = {}
        metrics_to_assess = [metric_name] if metric_name else list(self._metrics.keys())

        for name in metrics_to_assess:
            metric_def = self._metrics.get(name)
            if not metric_def:
                continue

            values = self.get_values(name, session_id=session_id, user_id=user_id)
            if not values:
                continue

            assessor = ThresholdAssessor(metric_def.thresholds)
            aggregated = self.aggregate(name, session_id=session_id, user_id=user_id)

            assessments[name] = {
                "value": aggregated,
                "count": len(values),
                "level": metric_def.assess(aggregated).value,
                "higher_is_better": metric_def.higher_is_better,
                "min": min(values),
                "max": max(values),
                "target": metric_def.target_value,
            }

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "user_id": user_id,
            "metrics": assessments,
        }

    # ==================== Visualization ====================

    def get_heatmap_data(
        self,
        metric_name: str,
        x_dimension: str,
        y_dimension: str,
        aggregation: Optional[Union[AggregationType, str]] = None,
        **query_params,
    ) -> Dict[str, Any]:
        """
        Generate heatmap data for a metric.

        Args:
            metric_name: Name of the metric
            x_dimension: Dimension for X axis
            y_dimension: Dimension for Y axis
            aggregation: Aggregation method
            **query_params: Additional query parameters

        Returns:
            Heatmap data structure
        """
        from ..visualizers.heatmap import HeatmapVisualizer

        datapoints = self.query(metric_name=metric_name, **query_params)
        metric_def = self._metrics.get(metric_name)
        agg = aggregation or (metric_def.aggregation if metric_def else AggregationType.MEAN)

        visualizer = HeatmapVisualizer(
            x_dimension=x_dimension,
            y_dimension=y_dimension,
            aggregation=agg,
        )
        return visualizer.generate(datapoints, metric_def)

    def get_chart_data(
        self,
        metric_name: str,
        chart_type: str = "bar",
        group_by: Optional[str] = None,
        **query_params,
    ) -> Dict[str, Any]:
        """
        Generate chart data for a metric.

        Args:
            metric_name: Name of the metric
            chart_type: Type of chart (bar, line, pie)
            group_by: Dimension to group by
            **query_params: Additional query parameters

        Returns:
            Chart data structure
        """
        from ..visualizers.chart import ChartVisualizer

        datapoints = self.query(metric_name=metric_name, **query_params)
        metric_def = self._metrics.get(metric_name)

        visualizer = ChartVisualizer(chart_type=chart_type)
        return visualizer.generate(datapoints, group_by, metric_def)

    def get_trend_data(
        self,
        metric_name: str,
        interval: str = "day",
        **query_params,
    ) -> Dict[str, Any]:
        """
        Generate trend data for a metric over time.

        Args:
            metric_name: Name of the metric
            interval: Time interval (hour, day, week, month)
            **query_params: Additional query parameters

        Returns:
            Trend data structure
        """
        from ..visualizers.trend import TrendVisualizer

        datapoints = self.query(metric_name=metric_name, **query_params)
        metric_def = self._metrics.get(metric_name)

        visualizer = TrendVisualizer(interval=interval)
        return visualizer.generate(datapoints, metric_def)

    # ==================== Event Hooks ====================

    def on_record(self, callback: Callable[[DataPoint], None]) -> "MetricsEngine":
        """Register a callback for when metrics are recorded."""
        self._on_record.append(callback)
        return self

    def on_session_start(self, callback: Callable[[MetricSession], None]) -> "MetricsEngine":
        """Register a callback for when sessions start."""
        self._on_session_start.append(callback)
        return self

    def on_session_end(self, callback: Callable[[MetricSession], None]) -> "MetricsEngine":
        """Register a callback for when sessions end."""
        self._on_session_end.append(callback)
        return self

    # ==================== Persistence ====================

    def set_storage(self, backend: "StorageBackend") -> "MetricsEngine":
        """Set the storage backend."""
        self._storage = backend
        return self

    def persist(self) -> None:
        """Persist all data to storage."""
        if self._storage:
            self._storage.flush()

    def clear(self, confirm: bool = False) -> None:
        """Clear all data."""
        if not confirm:
            raise ValueError("Must confirm=True to clear all data")
        with self._lock:
            self._sessions.clear()
            self._active_session = None
            if self._storage:
                self._storage.clear()

    # ==================== Export ====================

    def export(self, format: str = "json") -> Any:
        """Export all data."""
        data = {
            "metrics": {name: {
                "name": m.name,
                "type": m.metric_type.value,
                "description": m.description,
            } for name, m in self._metrics.items()},
            "sessions": {sid: s.to_dict() for sid, s in self._sessions.items()},
            "datapoints": [dp.to_dict() for dp in self._storage.query({}, None)],
        }

        if format == "json":
            import json
            return json.dumps(data, default=str)
        else:
            return data


# Type hint for storage backend (resolved at runtime)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..storage.base import StorageBackend
