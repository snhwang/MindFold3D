"""JSON file storage backend."""

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import StorageBackend
from ..core.datapoint import DataPoint


class JSONStorage(StorageBackend):
    """
    JSON file storage backend.

    Persists data to JSON files. Suitable for small to medium datasets.
    """

    def __init__(
        self,
        file_path: str = "metrics_data.json",
        auto_save: bool = True,
        pretty_print: bool = True,
    ):
        """
        Initialize JSON storage.

        Args:
            file_path: Path to JSON file
            auto_save: Whether to save after each write
            pretty_print: Whether to format JSON output
        """
        self._file_path = Path(file_path)
        self._auto_save = auto_save
        self._pretty_print = pretty_print
        self._datapoints: List[DataPoint] = []
        self._lock = threading.RLock()
        self._dirty = False

        # Load existing data
        self._load()

    def _load(self) -> None:
        """Load data from file."""
        if self._file_path.exists():
            try:
                with open(self._file_path, "r") as f:
                    data = json.load(f)
                    self._datapoints = [
                        DataPoint.from_dict(dp) for dp in data.get("datapoints", [])
                    ]
            except (json.JSONDecodeError, KeyError):
                self._datapoints = []

    def _save(self) -> None:
        """Save data to file."""
        # Ensure directory exists
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "saved_at": datetime.utcnow().isoformat(),
            "datapoints": [dp.to_dict() for dp in self._datapoints],
        }

        with open(self._file_path, "w") as f:
            if self._pretty_print:
                json.dump(data, f, indent=2, default=str)
            else:
                json.dump(data, f, default=str)

        self._dirty = False

    def store(self, datapoint: DataPoint) -> None:
        """Store a single datapoint."""
        with self._lock:
            self._datapoints.append(datapoint)
            self._dirty = True
            if self._auto_save:
                self._save()

    def store_batch(self, datapoints: List[DataPoint]) -> None:
        """Store multiple datapoints."""
        with self._lock:
            self._datapoints.extend(datapoints)
            self._dirty = True
            if self._auto_save:
                self._save()

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
            deleted = original_count - len(self._datapoints)
            if deleted > 0:
                self._dirty = True
                if self._auto_save:
                    self._save()
            return deleted

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
            self._dirty = True
            if self._auto_save:
                self._save()

    def flush(self) -> None:
        """Flush buffered data to file."""
        with self._lock:
            if self._dirty:
                self._save()

    def close(self) -> None:
        """Close and save any pending data."""
        self.flush()
