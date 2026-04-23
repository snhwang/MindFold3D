#!/usr/bin/env python
"""
Example 3: Collectors
=====================

This example shows how to use specialized collectors:
- CounterCollector for counting events
- TimerCollector for measuring durations
- SuccessFailureCounter for tracking outcomes
- CompositeCollector for multi-dimensional tracking
"""

import time
from metrics_framework import MetricsEngine
from metrics_framework.collectors import (
    CounterCollector,
    TimerCollector,
    CompositeCollector,
)
from metrics_framework.collectors.counter import SuccessFailureCounter


def main():
    engine = MetricsEngine()

    # --- Counter Collector ---
    print("=== Counter Collector ===")

    counter = CounterCollector("page_views", engine=engine)

    counter.inc()           # +1
    counter.inc()           # +1
    counter.increment(5)    # +5

    print(f"Total page views: {counter.get_value()}")
    print(f"Total increments: {counter.total_increments}")

    # --- Timer Collector ---
    print("\n=== Timer Collector ===")

    timer = TimerCollector("operation_duration", engine=engine)

    # Method 1: Context manager
    with timer.time():
        time.sleep(0.1)  # Simulate work

    # Method 2: Manual start/stop
    timer.start()
    time.sleep(0.05)
    duration = timer.stop()
    print(f"Last operation: {duration:.3f}s")

    # Method 3: Direct observation
    timer.observe(0.25)  # Record a known duration

    print(f"Timer stats:")
    print(f"  Count: {timer.count}")
    print(f"  Mean: {timer.mean:.3f}s")
    print(f"  Min: {timer.min:.3f}s")
    print(f"  Max: {timer.max:.3f}s")

    # --- Success/Failure Counter ---
    print("\n=== Success/Failure Counter ===")

    tracker = SuccessFailureCounter("api_calls", engine=engine)

    # Simulate API calls
    tracker.record_success()
    tracker.record_success()
    tracker.record_failure()
    tracker.record_success()
    tracker.record_failure()

    print(f"Total calls: {tracker.total}")
    print(f"Successes: {tracker.successes}")
    print(f"Failures: {tracker.failures}")
    print(f"Success rate: {tracker.success_rate:.1%}")

    # --- Composite Collector ---
    print("\n=== Composite Collector ===")

    collector = CompositeCollector("quiz_performance", engine=engine)

    # Record attempts with features
    collector.record_attempt(
        correct=True,
        response_time=2.5,
        features={"subject": "math", "difficulty": "easy"}
    )
    collector.record_attempt(
        correct=True,
        response_time=4.2,
        features={"subject": "math", "difficulty": "hard"}
    )
    collector.record_attempt(
        correct=False,
        response_time=8.1,
        features={"subject": "science", "difficulty": "hard"}
    )
    collector.record_attempt(
        correct=True,
        response_time=3.0,
        features={"subject": "science", "difficulty": "easy"}
    )
    collector.record_attempt(
        correct=False,
        response_time=6.5,
        features={"subject": "math", "difficulty": "hard"}
    )

    print(f"Total attempts: {collector.total_attempts}")
    print(f"Correct: {collector.total_correct}")
    print(f"Success rate: {collector.success_rate:.1%}")
    print(f"Avg response time: {collector.avg_response_time:.1f}s")

    # Get feature-level stats
    print("\nFeature stats:")
    stats = collector.get_all_feature_stats()
    for feature_name, values in stats.items():
        print(f"  {feature_name}:")
        for value, data in values.items():
            print(f"    {value}: {data['success_rate']:.0%} ({data['total_attempts']} attempts)")

    # Get weak areas
    print("\nWeak areas:")
    weak = collector.get_weak_areas(min_attempts=1, max_success_rate=0.6)
    for area in weak:
        print(f"  {area['feature_name']}={area['feature_value']}: {area['success_rate']:.0%}")


if __name__ == "__main__":
    main()
