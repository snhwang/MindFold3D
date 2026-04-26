"""In-memory storage backend."""

import threading
from typing import Any, Dict, List, Optional

from .base import StorageBackend
from ..core.datapoint import DataPoint


class MemoryStorage(StorageBackend):
    """
    In-memory storage backend.

    Fast but non-persistent storage for development and testing.
    """

    def __init__(self):
        """Initialize the memory storage."""
        self._datapoints: List[DataPoint] = []
        self._lock = threading.RLock()

    def store(self, datapoint: DataPoint) -> None:
        """Store a single datapoint."""
        with self._lock:
            self._datapoints.append(datapoint)

    def store_batch(self, datapoints: List[DataPoint]) -> None:
        """Store multiple datapoints."""
        with self._lock:
            self._datapoints.extend(datapoints)

    def query(
        self,
        filters: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> List[DataPoint]:
        """Query stored datapoints."""
        with self._lock:
            results = [
                dp for dp in self._datapoints
                if dp.matches_filter(filters)
            ]

            # Sort by timestamp (newest first)
            results.sort(key=lambda dp: dp.timestamp, reverse=True)

            if limit:
                results = results[:limit]

            return results

    def delete(self, filters: Dict[str, Any]) -> int:
        """Delete datapoints matching filters."""
        with self._lock:
            original_count = len(self._datapoints)
            self._datapoints = [
                dp for dp in self._datapoints
                if not dp.matches_filter(filters)
            ]
            return original_count - len(self._datapoints)

    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count datapoints matching filters."""
        with self._lock:
            if filters is None:
                return len(self._datapoints)
            return sum(1 for dp in self._datapoints if dp.matches_filter(filters))

    def clear(self) -> None:
        """Clear all stored data."""
        with self._lock:
            self._datapoints.clear()

    def get_all(self) -> List[DataPoint]:
        """Get all stored datapoints."""
        with self._lock:
            return self._datapoints.copy()
