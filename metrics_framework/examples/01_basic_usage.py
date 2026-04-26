#!/usr/bin/env python
"""
Example 1: Basic Usage
======================

This example shows the fundamental operations of the metrics framework:
- Creating an engine
- Defining metrics
- Recording data
- Querying and aggregating
- Assessing performance
"""

from metrics_framework import MetricsEngine, MetricDefinition


def main():
    # 1. Create the metrics engine
    engine = MetricsEngine()

    # 2. Define metrics for your application
    engine.register_metric(MetricDefinition(
        name="score",
        metric_type="percentage",
        description="User's score on tasks",
        higher_is_better=True,
        min_value=0,
        max_value=1,
    ))

    engine.register_metric(MetricDefinition(
        name="completion_time",
        metric_type="timer",
        description="Time to complete a task",
        unit="seconds",
        higher_is_better=False,  # Lower is better
    ))

    # 3. Record some data points
    print("Recording data points...")

    # Scores with dimensions
    engine.record("score", 0.85, dimensions={"task": "quiz", "difficulty": "easy"})
    engine.record("score", 0.72, dimensions={"task": "quiz", "difficulty": "hard"})
    engine.record("score", 0.91, dimensions={"task": "quiz", "difficulty": "easy"})
    engine.record("score", 0.65, dimensions={"task": "puzzle", "difficulty": "hard"})
    engine.record("score", 0.78, dimensions={"task": "puzzle", "difficulty": "medium"})

    # Completion times
    engine.record("completion_time", 45.2, dimensions={"task": "quiz"})
    engine.record("completion_time", 120.5, dimensions={"task": "puzzle"})
    engine.record("completion_time", 38.1, dimensions={"task": "quiz"})

    # 4. Query the data
    print("\n--- Querying Data ---")
    all_scores = engine.query(metric_name="score")
    print(f"Total score records: {len(all_scores)}")

    hard_scores = engine.query(metric_name="score", dimensions={"difficulty": "hard"})
    print(f"Hard difficulty records: {len(hard_scores)}")

    # 5. Aggregate the data
    print("\n--- Aggregation ---")
    avg_score = engine.aggregate("score")
    print(f"Average score: {avg_score:.2%}")

    by_difficulty = engine.aggregate("score", group_by=["difficulty"])
    print(f"By difficulty: {by_difficulty}")

    by_task = engine.aggregate("score", group_by=["task"])
    print(f"By task: {by_task}")

    avg_time = engine.aggregate("completion_time")
    print(f"Average completion time: {avg_time:.1f} seconds")

    # 6. Assess performance
    print("\n--- Assessment ---")
    report = engine.assess()

    for metric_name, data in report["metrics"].items():
        print(f"{metric_name}:")
        print(f"  Value: {data['value']:.2f}")
        print(f"  Level: {data['level']}")
        print(f"  Count: {data['count']}")


if __name__ == "__main__":
    main()
