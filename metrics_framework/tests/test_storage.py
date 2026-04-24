"""Tests for storage backends."""

import os
import tempfile
import pytest
from datetime import datetime

from metrics_framework.core.datapoint import DataPoint
from metrics_framework.storage import MemoryStorage, JSONStorage


class TestMemoryStorage:
    """Test MemoryStorage backend."""

    def test_store_and_query(self):
        """Test storing and querying datapoints."""
        storage = MemoryStorage()
        dp = DataPoint(
            metric_name="test",
            value=42.0,
            timestamp=datetime.utcnow(),
        )
        storage.store(dp)

        results = storage.query(metric_name="test")
        assert len(results) == 1
        assert results[0].value == 42.0

    def test_query_with_filters(self):
        """Test querying with dimension filters."""
        storage = MemoryStorage()

        storage.store(DataPoint(
            metric_name="test",
            value=1.0,
            timestamp=datetime.utcnow(),
            dimensions={"level": "easy"},
        ))
        storage.store(DataPoint(
            metric_name="test",
            value=2.0,
            timestamp=datetime.utcnow(),
            dimensions={"level": "hard"},
        ))

        results = storage.query(metric_name="test", filters={"level": "easy"})
        assert len(results) == 1
        assert results[0].value == 1.0

    def test_query_time_range(self):
        """Test querying by time range."""
        storage = MemoryStorage()
        now = datetime.utcnow()

        storage.store(DataPoint(
            metric_name="test",
            value=1.0,
            timestamp=datetime(2024, 1, 1, 12, 0),
        ))
        storage.store(DataPoint(
            metric_name="test",
            value=2.0,
            timestamp=datetime(2024, 6, 1, 12, 0),
        ))
        storage.store(DataPoint(
            metric_name="test",
            value=3.0,
            timestamp=datetime(2024, 12, 1, 12, 0),
        ))

        results = storage.query(
            metric_name="test",
            start_time=datetime(2024, 3, 1),
            end_time=datetime(2024, 9, 1),
        )
        assert len(results) == 1
        assert results[0].value == 2.0

    def test_clear(self):
        """Test clearing storage."""
        storage = MemoryStorage()
        storage.store(DataPoint(
            metric_name="test",
            value=1.0,
            timestamp=datetime.utcnow(),
        ))

        storage.clear()
        results = storage.query()
        assert len(results) == 0

    def test_count(self):
        """Test counting datapoints."""
        storage = MemoryStorage()
        for i in range(5):
            storage.store(DataPoint(
                metric_name="test",
                value=float(i),
                timestamp=datetime.utcnow(),
            ))

        assert storage.count() == 5
        assert storage.count(metric_name="test") == 5
        assert storage.count(metric_name="other") == 0


class TestJSONStorage:
    """Test JSONStorage backend."""

    def test_store_and_load(self):
        """Test storing and loading from JSON file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            storage = JSONStorage(filepath)
            dp = DataPoint(
                metric_name="test",
                value=42.0,
                timestamp=datetime.utcnow(),
                dimensions={"category": "A"},
            )
            storage.store(dp)

            # Create new storage instance to test loading
            storage2 = JSONStorage(filepath)
            results = storage2.query(metric_name="test")

            assert len(results) == 1
            assert results[0].value == 42.0
            assert results[0].dimensions.get("category") == "A"
        finally:
            os.unlink(filepath)

    def test_persistence(self):
        """Test that data persists across instances."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            # Store data
            storage1 = JSONStorage(filepath)
            storage1.store(DataPoint(
                metric_name="test",
                value=1.0,
                timestamp=datetime.utcnow(),
            ))
            storage1.store(DataPoint(
                metric_name="test",
                value=2.0,
                timestamp=datetime.utcnow(),
            ))

            # Load in new instance
            storage2 = JSONStorage(filepath)
            results = storage2.query(metric_name="test")

            assert len(results) == 2
        finally:
            os.unlink(filepath)

    def test_auto_save(self):
        """Test auto-save functionality."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            storage = JSONStorage(filepath, auto_save=True)
            storage.store(DataPoint(
                metric_name="test",
                value=99.0,
                timestamp=datetime.utcnow(),
            ))

            # File should exist and contain data
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
        finally:
            os.unlink(filepath)
