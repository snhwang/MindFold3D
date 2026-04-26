"""Chart visualization generator."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..core.definitions import AggregationType, MetricDefinition
from ..core.datapoint import DataPoint
from ..utils.stats import calculate_mean


@dataclass
class ChartConfig:
    """Configuration for chart generation."""
    # Colors
    colors: List[str] = field(default_factory=lambda: [
        "#0571b0", "#92c5de", "#f7f7f7", "#f4a582", "#ca0020",
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    ])

    # Chart options
    show_legend: bool = True
    show_grid: bool = True
    responsive: bool = True

    # Labels
    x_label: str = ""
    y_label: str = ""
    title: str = ""

    # Value formatting
    value_format: str = ".2f"
    percentage_format: bool = False


class ChartVisualizer:
    """
    Visualizer for generating chart data structures.

    Supports bar, line, pie, and scatter charts.
    """

    def __init__(
        self,
        chart_type: str = "bar",
        config: Optional[ChartConfig] = None,
    ):
        """
        Initialize the chart visualizer.

        Args:
            chart_type: Type of chart (bar, line, pie, scatter, doughnut)
            config: Chart configuration
        """
        self._chart_type = chart_type
        self._config = config or ChartConfig()

    def generate(
        self,
        datapoints: List[DataPoint],
        group_by: Optional[str] = None,
        metric_definition: Optional[MetricDefinition] = None,
    ) -> Dict[str, Any]:
        """
        Generate chart data from datapoints.

        Args:
            datapoints: DataPoints to visualize
            group_by: Optional dimension to group by
            metric_definition: Optional metric definition

        Returns:
            Chart data structure (compatible with Chart.js)
        """
        if not datapoints:
            return self._empty_chart()

        if group_by:
            return self._generate_grouped(datapoints, group_by)
        else:
            return self._generate_simple(datapoints)

    def _generate_simple(self, datapoints: List[DataPoint]) -> Dict[str, Any]:
        """Generate chart without grouping."""
        values = [dp.value for dp in datapoints]
        labels = [dp.timestamp.strftime("%Y-%m-%d %H:%M") for dp in datapoints]

        return {
            "type": self._chart_type,
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": datapoints[0].metric_name,
                    "data": values,
                    "backgroundColor": self._config.colors[0],
                    "borderColor": self._config.colors[0],
                }],
            },
            "options": self._get_options(),
        }

    def _generate_grouped(
        self,
        datapoints: List[DataPoint],
        group_by: str,
    ) -> Dict[str, Any]:
        """Generate chart with grouping."""
        groups: Dict[Any, List[float]] = defaultdict(list)

        for dp in datapoints:
            group_value = dp.dimensions.get(group_by)
            if group_value is not None:
                groups[group_value].append(dp.value)

        # Aggregate each group
        labels = sorted(groups.keys(), key=str)
        values = [calculate_mean(groups[label]) for label in labels]
        counts = [len(groups[label]) for label in labels]

        # Assign colors
        colors = []
        for i in range(len(labels)):
            colors.append(self._config.colors[i % len(self._config.colors)])

        return {
            "type": self._chart_type,
            "data": {
                "labels": [str(l) for l in labels],
                "datasets": [{
                    "label": datapoints[0].metric_name,
                    "data": values,
                    "backgroundColor": colors,
                    "borderColor": colors,
                    "counts": counts,  # Extra data for tooltips
                }],
            },
            "options": self._get_options(),
        }

    def generate_multi_series(
        self,
        series_data: Dict[str, List[DataPoint]],
        x_accessor: str = "timestamp",
    ) -> Dict[str, Any]:
        """
        Generate multi-series chart.

        Args:
            series_data: Dict of series_name -> datapoints
            x_accessor: How to get X values (timestamp or dimension name)

        Returns:
            Multi-series chart data
        """
        datasets = []
        all_x_values = set()

        for i, (series_name, datapoints) in enumerate(series_data.items()):
            if not datapoints:
                continue

            # Get x values
            x_values = []
            y_values = []

            for dp in sorted(datapoints, key=lambda d: d.timestamp):
                if x_accessor == "timestamp":
                    x = dp.timestamp.strftime("%Y-%m-%d")
                else:
                    x = dp.dimensions.get(x_accessor)

                if x is not None:
                    x_values.append(x)
                    y_values.append(dp.value)
                    all_x_values.add(x)

            color = self._config.colors[i % len(self._config.colors)]
            datasets.append({
                "label": series_name,
                "data": y_values,
                "x_values": x_values,
                "backgroundColor": color,
                "borderColor": color,
                "fill": False,
            })

        return {
            "type": self._chart_type,
            "data": {
                "labels": sorted(all_x_values, key=str),
                "datasets": datasets,
            },
            "options": self._get_options(),
        }

    def generate_comparison_chart(
        self,
        comparisons: List[Dict[str, Any]],
        value_key: str = "value",
        label_key: str = "label",
    ) -> Dict[str, Any]:
        """
        Generate comparison chart from pre-aggregated data.

        Args:
            comparisons: List of comparison dicts
            value_key: Key for values
            label_key: Key for labels

        Returns:
            Comparison chart data
        """
        labels = [c.get(label_key, f"Item {i}") for i, c in enumerate(comparisons)]
        values = [c.get(value_key, 0) for c in comparisons]

        colors = []
        for i, c in enumerate(comparisons):
            # Use custom color if provided
            if "color" in c:
                colors.append(c["color"])
            else:
                colors.append(self._config.colors[i % len(self._config.colors)])

        return {
            "type": self._chart_type,
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": "Comparison",
                    "data": values,
                    "backgroundColor": colors,
                    "borderColor": colors,
                }],
            },
            "options": self._get_options(),
        }

    def generate_stacked_bar(
        self,
        datapoints: List[DataPoint],
        stack_by: str,
        group_by: str,
    ) -> Dict[str, Any]:
        """
        Generate stacked bar chart.

        Args:
            datapoints: DataPoints to visualize
            stack_by: Dimension to stack
            group_by: Dimension to group bars

        Returns:
            Stacked bar chart data
        """
        # Group by both dimensions
        data: Dict[Any, Dict[Any, List[float]]] = defaultdict(lambda: defaultdict(list))
        stack_values = set()
        group_values = set()

        for dp in datapoints:
            stack_val = dp.dimensions.get(stack_by)
            group_val = dp.dimensions.get(group_by)

            if stack_val is not None and group_val is not None:
                data[group_val][stack_val].append(dp.value)
                stack_values.add(stack_val)
                group_values.add(group_val)

        # Create datasets for each stack value
        labels = sorted(group_values, key=str)
        stack_labels = sorted(stack_values, key=str)
        datasets = []

        for i, stack_val in enumerate(stack_labels):
            values = []
            for group_val in labels:
                group_stack_values = data[group_val].get(stack_val, [])
                values.append(calculate_mean(group_stack_values) if group_stack_values else 0)

            color = self._config.colors[i % len(self._config.colors)]
            datasets.append({
                "label": str(stack_val),
                "data": values,
                "backgroundColor": color,
                "borderColor": color,
            })

        options = self._get_options()
        options["scales"] = {
            "x": {"stacked": True},
            "y": {"stacked": True},
        }

        return {
            "type": "bar",
            "data": {
                "labels": [str(l) for l in labels],
                "datasets": datasets,
            },
            "options": options,
        }

    def _get_options(self) -> Dict[str, Any]:
        """Get Chart.js options."""
        options = {
            "responsive": self._config.responsive,
            "plugins": {
                "legend": {
                    "display": self._config.show_legend,
                },
                "title": {
                    "display": bool(self._config.title),
                    "text": self._config.title,
                },
            },
        }

        if self._config.show_grid:
            options["scales"] = {
                "x": {
                    "grid": {"display": True},
                    "title": {
                        "display": bool(self._config.x_label),
                        "text": self._config.x_label,
                    },
                },
                "y": {
                    "grid": {"display": True},
                    "title": {
                        "display": bool(self._config.y_label),
                        "text": self._config.y_label,
                    },
                },
            }

        return options

    def _empty_chart(self) -> Dict[str, Any]:
        """Return empty chart structure."""
        return {
            "type": self._chart_type,
            "data": {
                "labels": [],
                "datasets": [],
            },
            "options": self._get_options(),
        }


class PerformanceChartGenerator:
    """Specialized chart generator for performance metrics."""

    def __init__(self, config: Optional[ChartConfig] = None):
        """Initialize the generator."""
        self._config = config or ChartConfig()

    def generate_success_rate_chart(
        self,
        feature_stats: Dict[str, Dict[str, Any]],
        features: Optional[List[str]] = None,
        min_attempts: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate success rate bar chart by feature.

        Args:
            feature_stats: Feature statistics
            features: Features to include
            min_attempts: Minimum attempts to show

        Returns:
            Chart.js compatible data
        """
        features_to_use = features or sorted(feature_stats.keys())

        labels = []
        success_rates = []
        counts = []
        colors = []

        for feature in features_to_use:
            if feature not in feature_stats:
                continue

            # Aggregate across all values
            total_correct = 0
            total_incorrect = 0

            for value_stats in feature_stats[feature].values():
                if isinstance(value_stats, dict):
                    total_correct += value_stats.get("correct", 0)
                    total_incorrect += value_stats.get("incorrect", 0)

            total = total_correct + total_incorrect
            if total >= min_attempts:
                rate = total_correct / total
                labels.append(feature)
                success_rates.append(rate)
                counts.append(total)

                # Color based on performance
                if rate >= 0.9:
                    colors.append("#0571b0")
                elif rate >= 0.7:
                    colors.append("#92c5de")
                elif rate >= 0.5:
                    colors.append("#f7f7f7")
                elif rate >= 0.3:
                    colors.append("#f4a582")
                else:
                    colors.append("#ca0020")

        return {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": "Success Rate",
                    "data": success_rates,
                    "backgroundColor": colors,
                    "borderColor": colors,
                    "counts": counts,
                }],
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "legend": {"display": False},
                    "title": {"display": True, "text": "Success Rate by Feature"},
                },
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "max": 1,
                        "ticks": {
                            "callback": "percentageCallback",
                        },
                    },
                },
            },
        }

    def generate_response_time_chart(
        self,
        feature_stats: Dict[str, Dict[str, Any]],
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate response time chart by feature.

        Args:
            feature_stats: Feature statistics
            features: Features to include

        Returns:
            Chart.js compatible data
        """
        features_to_use = features or sorted(feature_stats.keys())

        labels = []
        avg_times = []

        for feature in features_to_use:
            if feature not in feature_stats:
                continue

            all_times = []
            for value_stats in feature_stats[feature].values():
                if isinstance(value_stats, dict):
                    times = value_stats.get("response_times", [])
                    all_times.extend(times)

            if all_times:
                labels.append(feature)
                avg_times.append(sum(all_times) / len(all_times))

        return {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": "Avg Response Time (s)",
                    "data": avg_times,
                    "backgroundColor": "#1f77b4",
                    "borderColor": "#1f77b4",
                }],
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "legend": {"display": False},
                    "title": {"display": True, "text": "Average Response Time by Feature"},
                },
                "scales": {
                    "y": {
                        "beginAtZero": True,
                    },
                },
            },
        }
