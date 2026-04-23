"""Heatmap visualization generator."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.definitions import AggregationType, MetricDefinition
from ..core.datapoint import DataPoint
from ..aggregators.dimensional import DimensionalAggregator
from ..utils.stats import calculate_mean


@dataclass
class HeatmapConfig:
    """Configuration for heatmap generation."""
    # Color scheme
    color_scale: str = "diverging"  # diverging, sequential, categorical
    low_color: str = "#ca0020"  # Red for poor
    mid_color: str = "#f7f7f7"  # White for average
    high_color: str = "#0571b0"  # Blue for good

    # Color thresholds (for performance metrics)
    color_thresholds: List[Tuple[float, str]] = field(default_factory=lambda: [
        (0.0, "#ca0020"),   # 0-30%: Very poor (red)
        (0.3, "#f4a582"),   # 30-50%: Poor (light red)
        (0.5, "#f7f7f7"),   # 50-70%: Average (white)
        (0.7, "#92c5de"),   # 70-90%: Good (light blue)
        (0.9, "#0571b0"),   # 90-100%: Excellent (blue)
    ])

    # Display options
    show_values: bool = True
    show_counts: bool = True
    min_count_threshold: int = 1
    value_format: str = ".1%"  # Format for displaying values

    # Size options
    cell_width: int = 40
    cell_height: int = 30

    # Labels
    x_label: str = ""
    y_label: str = ""
    title: str = ""

    def get_color_for_value(self, value: float) -> str:
        """Get color for a value based on thresholds."""
        for threshold, color in reversed(self.color_thresholds):
            if value >= threshold:
                return color
        return self.low_color


class HeatmapVisualizer:
    """
    Visualizer for generating heatmap data structures.

    Generates data suitable for D3.js, Chart.js, or other
    visualization libraries.
    """

    def __init__(
        self,
        x_dimension: str,
        y_dimension: str,
        aggregation: Union[AggregationType, str] = AggregationType.MEAN,
        config: Optional[HeatmapConfig] = None,
    ):
        """
        Initialize the heatmap visualizer.

        Args:
            x_dimension: Dimension for X axis
            y_dimension: Dimension for Y axis
            aggregation: Aggregation method for values
            config: Heatmap configuration
        """
        self._x_dimension = x_dimension
        self._y_dimension = y_dimension
        if isinstance(aggregation, str):
            aggregation = AggregationType(aggregation)
        self._aggregation = aggregation
        self._config = config or HeatmapConfig()
        self._aggregator = DimensionalAggregator(aggregation)

    def generate(
        self,
        datapoints: List[DataPoint],
        metric_definition: Optional[MetricDefinition] = None,
    ) -> Dict[str, Any]:
        """
        Generate heatmap data from datapoints.

        Args:
            datapoints: DataPoints to visualize
            metric_definition: Optional metric definition for context

        Returns:
            Heatmap data structure
        """
        if not datapoints:
            return self._empty_heatmap()

        # Group by both dimensions
        groups: Dict[Tuple[Any, Any], List[float]] = defaultdict(list)
        x_values = set()
        y_values = set()

        for dp in datapoints:
            x_val = dp.dimensions.get(self._x_dimension)
            y_val = dp.dimensions.get(self._y_dimension)
            if x_val is not None and y_val is not None:
                groups[(x_val, y_val)].append(dp.value)
                x_values.add(x_val)
                y_values.add(y_val)

        # Sort labels
        x_labels = sorted(x_values, key=str)
        y_labels = sorted(y_values, key=str)

        # Build cells
        cells = []
        value_range = {"min": float("inf"), "max": float("-inf")}

        for y_idx, y_val in enumerate(y_labels):
            for x_idx, x_val in enumerate(x_labels):
                values = groups.get((x_val, y_val), [])
                count = len(values)

                if count >= self._config.min_count_threshold:
                    agg_value = self._aggregate(values)
                    value_range["min"] = min(value_range["min"], agg_value)
                    value_range["max"] = max(value_range["max"], agg_value)

                    cells.append({
                        "x": x_idx,
                        "y": y_idx,
                        "x_label": str(x_val),
                        "y_label": str(y_val),
                        "value": agg_value,
                        "count": count,
                        "color": self._config.get_color_for_value(agg_value),
                        "display_value": self._format_value(agg_value),
                    })
                else:
                    cells.append({
                        "x": x_idx,
                        "y": y_idx,
                        "x_label": str(x_val),
                        "y_label": str(y_val),
                        "value": None,
                        "count": count,
                        "color": "#cccccc",  # Gray for no data
                        "display_value": "N/A",
                    })

        return {
            "type": "heatmap",
            "x_dimension": self._x_dimension,
            "y_dimension": self._y_dimension,
            "x_labels": [str(x) for x in x_labels],
            "y_labels": [str(y) for y in y_labels],
            "cells": cells,
            "value_range": value_range if value_range["min"] != float("inf") else {"min": 0, "max": 1},
            "aggregation": self._aggregation.value,
            "config": {
                "color_thresholds": self._config.color_thresholds,
                "show_values": self._config.show_values,
                "show_counts": self._config.show_counts,
                "cell_width": self._config.cell_width,
                "cell_height": self._config.cell_height,
                "x_label": self._config.x_label or self._x_dimension,
                "y_label": self._config.y_label or self._y_dimension,
                "title": self._config.title,
            },
            "total_datapoints": len(datapoints),
        }

    def generate_feature_heatmap(
        self,
        feature_stats: Dict[str, Dict[str, Any]],
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate heatmap from feature statistics.

        Args:
            feature_stats: Feature-level statistics dict
            features: List of features to include (or all if None)

        Returns:
            Heatmap data structure
        """
        features_to_use = features or list(feature_stats.keys())

        # Build cells
        cells = []
        x_labels = []
        y_labels_set = set()

        for x_idx, feature_name in enumerate(features_to_use):
            if feature_name not in feature_stats:
                continue

            x_labels.append(feature_name)
            feature_data = feature_stats[feature_name]

            for value_key, value_stats in feature_data.items():
                if not isinstance(value_stats, dict):
                    continue

                y_labels_set.add(value_key)
                correct = value_stats.get("correct", 0)
                incorrect = value_stats.get("incorrect", 0)
                total = correct + incorrect

                if total >= self._config.min_count_threshold:
                    success_rate = correct / total if total > 0 else 0
                    cells.append({
                        "feature": feature_name,
                        "value": value_key,
                        "success_rate": success_rate,
                        "count": total,
                        "color": self._config.get_color_for_value(success_rate),
                        "display_value": f"{success_rate:.1%}",
                    })

        # Sort y_labels
        y_labels = sorted(y_labels_set, key=str)

        # Create structured data for D3.js
        d3_data = []
        for cell in cells:
            d3_data.append({
                "feature": cell["feature"],
                "value": cell["value"],
                "success_rate": cell["success_rate"],
                "count": cell["count"],
                "color": cell["color"],
            })

        return {
            "type": "feature_heatmap",
            "features": x_labels,
            "values": y_labels,
            "cells": cells,
            "d3_data": d3_data,
            "config": {
                "color_thresholds": self._config.color_thresholds,
                "show_values": self._config.show_values,
                "show_counts": self._config.show_counts,
            },
        }

    def _aggregate(self, values: List[float]) -> float:
        """Aggregate values using configured method."""
        if not values:
            return 0.0

        if self._aggregation == AggregationType.MEAN:
            return calculate_mean(values)
        elif self._aggregation == AggregationType.SUM:
            return sum(values)
        elif self._aggregation == AggregationType.MIN:
            return min(values)
        elif self._aggregation == AggregationType.MAX:
            return max(values)
        elif self._aggregation == AggregationType.COUNT:
            return float(len(values))
        else:
            return calculate_mean(values)

    def _format_value(self, value: float) -> str:
        """Format value for display."""
        fmt = self._config.value_format
        if fmt.endswith("%"):
            return f"{value * 100:.{fmt[1:-1]}f}%"
        else:
            return f"{value:{fmt}}"

    def _empty_heatmap(self) -> Dict[str, Any]:
        """Return empty heatmap structure."""
        return {
            "type": "heatmap",
            "x_dimension": self._x_dimension,
            "y_dimension": self._y_dimension,
            "x_labels": [],
            "y_labels": [],
            "cells": [],
            "value_range": {"min": 0, "max": 1},
            "aggregation": self._aggregation.value,
            "config": {},
            "total_datapoints": 0,
        }


class FeatureValueHeatmapGenerator:
    """
    Specialized heatmap generator for feature-value performance analysis.

    Designed for spatial cognition training metrics.
    """

    def __init__(self, config: Optional[HeatmapConfig] = None):
        """Initialize the generator."""
        self._config = config or HeatmapConfig()

    def generate_from_feature_stats(
        self,
        feature_stats: Dict[str, Dict[str, Any]],
        feature_order: Optional[List[str]] = None,
        min_attempts: int = 1,
    ) -> Dict[str, Any]:
        """
        Generate heatmap from feature statistics.

        Args:
            feature_stats: Nested dict of feature -> value -> stats
            feature_order: Optional ordering of features
            min_attempts: Minimum attempts to show

        Returns:
            Heatmap data for D3.js visualization
        """
        features = feature_order or sorted(feature_stats.keys())

        # Collect all data points
        data = []
        all_values = set()

        for feature in features:
            if feature not in feature_stats:
                continue

            for value_key, stats in feature_stats[feature].items():
                if not isinstance(stats, dict):
                    continue

                correct = stats.get("correct", 0)
                incorrect = stats.get("incorrect", 0)
                total = correct + incorrect

                if total >= min_attempts:
                    success_rate = correct / total
                    avg_time = 0.0
                    response_times = stats.get("response_times", [])
                    if response_times:
                        avg_time = sum(response_times) / len(response_times)

                    all_values.add(value_key)
                    data.append({
                        "feature": feature,
                        "value": str(value_key),
                        "success_rate": success_rate,
                        "count": total,
                        "correct": correct,
                        "incorrect": incorrect,
                        "avg_response_time": avg_time,
                        "color": self._config.get_color_for_value(success_rate),
                    })

        return {
            "type": "feature_value_heatmap",
            "data": data,
            "features": features,
            "values": sorted(all_values, key=str),
            "config": {
                "colorScale": [
                    {"threshold": t, "color": c}
                    for t, c in self._config.color_thresholds
                ],
                "minAttempts": min_attempts,
            },
        }
