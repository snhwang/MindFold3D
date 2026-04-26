"""Base storage backend interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..core.datapoint import DataPoint


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def store(self, datapoint: DataPoint) -> None:
        """
        Store a single datapoint.

        Args:
            datapoint: The datapoint to store
        """
        pass

    @abstractmethod
    def store_batch(self, datapoints: List[DataPoint]) -> None:
        """
        Store multiple datapoints.

        Args:
            datapoints: List of datapoints to store
        """
        pass

    @abstractmethod
    def query(
        self,
        filters: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> List[DataPoint]:
        """
        Query stored datapoints.

        Args:
            filters: Filter criteria
            limit: Maximum number of results

        Returns:
            List of matching datapoints
        """
        pass

    @abstractmethod
    def delete(self, filters: Dict[str, Any]) -> int:
        """
        Delete datapoints matching filters.

        Args:
            filters: Filter criteria

        Returns:
            Number of deleted datapoints
        """
        pass

    @abstractmethod
    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count datapoints matching filters.

        Args:
            filters: Optional filter criteria

        Returns:
            Number of matching datapoints
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored data."""
        pass

    def flush(self) -> None:
        """
        Flush any buffered data to storage.

        Override this method if the backend buffers data.
        """
        pass

    def close(self) -> None:
        """
        Close the storage connection.

        Override this method if the backend needs cleanup.
        """
        pass

    def __enter__(self) -> "StorageBackend":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
