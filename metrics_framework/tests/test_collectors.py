"""Tests for metric collectors."""

import pytest
from metrics_framework.collectors import (
    CounterCollector,
    GaugeCollector,
    HistogramCollector,
    TimerCollector,
)
from metrics_framework.collectors.counter import SuccessFailureCounter
from metrics_framework.collectors.timer import ResponseTimeTracker


class TestCounterCollector:
    """Test CounterCollector functionality."""

    def test_increment(self):
        """Test counter increment."""
        counter = CounterCollector(metric_name="test")
        counter.increment()
        counter.increment()
        counter.increment()

        assert counter.value == 3

    def test_increment_by_amount(self):
        """Test incrementing by specific amount."""
        counter = CounterCollector(metric_name="test")
        counter.increment(5)
        counter.increment(3)

        assert counter.value == 8

    def test_reset(self):
        """Test counter reset."""
        counter = CounterCollector(metric_name="test")
        counter.increment(10)
        counter.reset()

        assert counter.value == 0


class TestSuccessFailureCounter:
    """Test SuccessFailureCounter functionality."""

    def test_record_success(self):
        """Test recording successes."""
        counter = SuccessFailureCounter(metric_name="test")
        counter.record_success()
        counter.record_success()
        counter.record_failure()

        assert counter.successes == 2
        assert counter.failures == 1
        assert counter.total == 3

    def test_success_rate(self):
        """Test success rate calculation."""
        counter = SuccessFailureCounter(metric_name="test")
        counter.record_success()
        counter.record_success()
        counter.record_failure()
        counter.record_failure()

        assert counter.success_rate == 0.5

    def test_success_rate_empty(self):
        """Test success rate when empty."""
        counter = SuccessFailureCounter(metric_name="test")

        assert counter.success_rate == 0.0


class TestGaugeCollector:
    """Test GaugeCollector functionality."""

    def test_set_value(self):
        """Test setting gauge value."""
        gauge = GaugeCollector(metric_name="test")
        gauge.set(42.5)

        assert gauge.value == 42.5

    def test_increment_decrement(self):
        """Test gauge increment and decrement."""
        gauge = GaugeCollector(metric_name="test")
        gauge.set(10)
        gauge.increment(5)
        gauge.decrement(3)

        assert gauge.value == 12

    def test_min_max_tracking(self):
        """Test min/max value tracking."""
        gauge = GaugeCollector(metric_name="test")
        gauge.set(10)
        gauge.set(5)
        gauge.set(15)
        gauge.set(8)

        stats = gauge.get_stats()
        assert stats["min"] == 5
        assert stats["max"] == 15


class TestHistogramCollector:
    """Test HistogramCollector functionality."""

    def test_observe(self):
        """Test histogram observation."""
        histogram = HistogramCollector(
            metric_name="test",
            buckets=[10, 50, 100, 500]
        )
        histogram.observe(5)
        histogram.observe(25)
        histogram.observe(75)
        histogram.observe(200)

        assert histogram.count == 4

    def test_bucket_counts(self):
        """Test bucket counting."""
        histogram = HistogramCollector(
            metric_name="test",
            buckets=[10, 50, 100]
        )
        histogram.observe(5)   # bucket 10
        histogram.observe(25)  # bucket 50
        histogram.observe(25)  # bucket 50
        histogram.observe(75)  # bucket 100
        histogram.observe(200) # overflow

        buckets = histogram.get_buckets()
        assert buckets[10] == 1
        assert buckets[50] == 2
        assert buckets[100] == 1

    def test_percentiles(self):
        """Test percentile calculation."""
        histogram = HistogramCollector(metric_name="test")
        for i in range(1, 101):
            histogram.observe(i)

        percentiles = histogram.get_percentiles([50, 90, 99])
        assert abs(percentiles[50] - 50) < 2
        assert abs(percentiles[90] - 90) < 2


class TestTimerCollector:
    """Test TimerCollector functionality."""

    def test_observe(self):
        """Test timer observation."""
        timer = TimerCollector(metric_name="test")
        timer.observe(1.5)
        timer.observe(2.0)
        timer.observe(2.5)

        assert timer.count == 3

    def test_stats(self):
        """Test timer statistics."""
        timer = TimerCollector(metric_name="test")
        timer.observe(1.0)
        timer.observe(2.0)
        timer.observe(3.0)

        stats = timer.get_stats()
        assert stats["count"] == 3
        assert stats["mean"] == 2.0
        assert stats["min"] == 1.0
        assert stats["max"] == 3.0


class TestResponseTimeTracker:
    """Test ResponseTimeTracker functionality."""

    def test_categorization(self):
        """Test response time categorization."""
        tracker = ResponseTimeTracker(
            metric_name="test",
            fast_threshold=1.0,
            slow_threshold=5.0,
        )
        tracker.observe(0.5)  # fast
        tracker.observe(2.0)  # normal
        tracker.observe(6.0)  # slow

        stats = tracker.get_stats()
        assert stats["fast_count"] == 1
        assert stats["normal_count"] == 1
        assert stats["slow_count"] == 1
