#!/usr/bin/env python
"""
Example 4: Visualization
========================

This example shows how to generate visualization data:
- Heatmaps for cross-dimensional analysis
- Bar/Line charts
- Trend data over time
- Radar charts for profiles
"""

import json
from datetime import datetime, timedelta
from metrics_framework import MetricsEngine, MetricDefinition
from metrics_framework.visualizers import (
    HeatmapVisualizer,
    ChartVisualizer,
    TrendVisualizer,
    RadarVisualizer,
)


def main():
    engine = MetricsEngine()

    # Define and register metrics
    engine.register_metric(MetricDefinition(
        name="accuracy",
        metric_type="percentage",
        higher_is_better=True,
    ))

    # Generate sample data
    print("Generating sample data...")

    features = ["rotation", "mirror", "assembly", "matching"]
    difficulties = ["easy", "medium", "hard"]

    # Record data with varying accuracy by feature and difficulty
    accuracy_map = {
        ("rotation", "easy"): 0.92,
        ("rotation", "medium"): 0.78,
        ("rotation", "hard"): 0.61,
        ("mirror", "easy"): 0.88,
        ("mirror", "medium"): 0.72,
        ("mirror", "hard"): 0.55,
        ("assembly", "easy"): 0.95,
        ("assembly", "medium"): 0.82,
        ("assembly", "hard"): 0.68,
        ("matching", "easy"): 0.98,
        ("matching", "medium"): 0.85,
        ("matching", "hard"): 0.71,
    }

    # Record multiple data points for each combination
    import random
    for (feature, difficulty), base_accuracy in accuracy_map.items():
        for _ in range(10):
            # Add some variance
            value = base_accuracy + random.uniform(-0.1, 0.1)
            value = max(0, min(1, value))  # Clamp to 0-1
            engine.record("accuracy", value, dimensions={
                "feature": feature,
                "difficulty": difficulty,
            })

    # --- Heatmap ---
    print("\n=== Heatmap Data ===")

    heatmap_viz = HeatmapVisualizer(
        x_dimension="feature",
        y_dimension="difficulty",
        aggregation="mean",
    )
    heatmap_data = heatmap_viz.generate(engine.query(metric_name="accuracy"))

    print(f"X labels: {heatmap_data['x_labels']}")
    print(f"Y labels: {heatmap_data['y_labels']}")
    print(f"Sample cells:")
    for cell in heatmap_data['cells'][:3]:
        print(f"  {cell['x_label']} x {cell['y_label']}: {cell['value']:.2f} (color: {cell['color']})")

    # --- Bar Chart ---
    print("\n=== Bar Chart Data ===")

    chart_viz = ChartVisualizer(chart_type="bar")
    chart_data = chart_viz.generate(
        engine.query(metric_name="accuracy"),
        group_by="feature"
    )

    print(f"Chart type: {chart_data['type']}")
    print(f"Labels: {chart_data['data']['labels']}")
    print(f"Values: {chart_data['data']['datasets'][0]['data']}")

    # --- Trend (simulated time-series) ---
    print("\n=== Trend Data ===")

    # Add time-series data
    base_time = datetime.now() - timedelta(days=10)
    for day in range(10):
        timestamp = base_time + timedelta(days=day)
        # Simulate improving trend
        value = 0.6 + (day * 0.03) + random.uniform(-0.05, 0.05)
        engine.record("accuracy", value, timestamp=timestamp)

    trend_viz = TrendVisualizer(interval="day")
    trend_data = trend_viz.generate(engine.query(metric_name="accuracy"))

    print(f"Trend direction: {trend_data['trend_analysis']['trend']}")
    print(f"Change: {trend_data['trend_analysis']['change_percent']:.1f}%")
    print(f"Data points: {len(trend_data['data']['labels'])}")

    # --- Radar Chart ---
    print("\n=== Radar Chart Data ===")

    # Calculate average accuracy per feature for radar
    by_feature = engine.aggregate("accuracy", group_by=["feature"])

    radar_viz = RadarVisualizer(max_value=1.0)
    radar_data = radar_viz.generate(
        {k.title(): v for k, v in by_feature.items()},
        title="Performance Profile"
    )

    print(f"Radar labels: {radar_data['data']['labels']}")
    print(f"Radar values: {radar_data['data']['datasets'][0]['data']}")

    # --- Export as JSON (for frontend) ---
    print("\n=== JSON Export ===")

    dashboard_data = {
        "heatmap": heatmap_data,
        "chart": chart_data,
        "trend": trend_data,
        "radar": radar_data,
    }

    # Pretty print a sample
    print("Sample JSON structure (heatmap config):")
    print(json.dumps(heatmap_data["config"], indent=2))


if __name__ == "__main__":
    main()
