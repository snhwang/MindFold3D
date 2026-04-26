"""Trend visualization generator."""

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ..core.definitions import AggregationType, MetricDefinition
from ..core.datapoint import DataPoint
from ..aggregators.temporal import TemporalAggregator
from ..utils.stats import calculate_moving_average, calculate_exponential_moving_average


class TrendVisualizer:
    """
    Visualizer for time-series trend data.

    Generates data suitable for line charts with trend analysis.
    """

    def __init__(
        self,
        interval: str = "day",
        aggregation: Union[AggregationType, str] = AggregationType.MEAN,
    ):
        """
        Initialize the trend visualizer.

        Args:
            interval: Time interval for aggregation
            aggregation: Aggregation method
        """
        self._interval = interval
        if isinstance(aggregation, str):
            aggregation = AggregationType(aggregation)
        self._aggregation = aggregation
        self._temporal_aggregator = TemporalAggregator(aggregation)

    def generate(
        self,
        datapoints: List[DataPoint],
        metric_definition: Optional[MetricDefinition] = None,
    ) -> Dict[str, Any]:
        """
        Generate trend visualization data.

        Args:
            datapoints: DataPoints to visualize
            metric_definition: Optional metric definition

        Returns:
            Trend data structure
        """
        if not datapoints:
            return self._empty_trend()

        # Get time series
        time_series = self._temporal_aggregator.get_time_series(
            datapoints,
            self._interval,
            fill_gaps=True,
        )

        # Calculate trend line
        values = [point["value"] for point in time_series]
        trend_data = self._temporal_aggregator.calculate_trend(
            datapoints,
            self._interval,
        )

        # Calculate moving averages
        sma = calculate_moving_average(values, 5) if len(values) >= 5 else values
        ema = calculate_exponential_moving_average(values, 0.3) if len(values) >= 2 else values

        # Build datasets for Chart.js
        labels = [point["timestamp"] for point in time_series]

        return {
            "type": "trend",
            "data": {
                "labels": labels,
                "datasets": [
                    {
                        "label": "Actual",
                        "data": values,
                        "borderColor": "#1f77b4",
                        "backgroundColor": "rgba(31, 119, 180, 0.1)",
                        "fill": True,
                        "tension": 0.1,
                    },
                    {
                        "label": "Moving Avg (5)",
                        "data": sma,
                        "borderColor": "#ff7f0e",
                        "backgroundColor": "transparent",
                        "borderDash": [5, 5],
                        "fill": False,
                    },
                    {
                        "label": "EMA",
                        "data": ema,
                        "borderColor": "#2ca02c",
                        "backgroundColor": "transparent",
                        "borderDash": [2, 2],
                        "fill": False,
                    },
                ],
            },
            "trend_analysis": trend_data,
            "options": {
                "responsive": True,
                "plugins": {
                    "legend": {"display": True},
                    "title": {"display": True, "text": "Performance Trend"},
                },
                "scales": {
                    "x": {"type": "time" if self._interval != "custom" else "category"},
                    "y": {"beginAtZero": True},
                },
            },
        }

    def generate_comparison_trend(
        self,
        series_map: Dict[str, List[DataPoint]],
    ) -> Dict[str, Any]:
        """
        Generate trend comparison for multiple series.

        Args:
            series_map: Dict of series_name -> datapoints

        Returns:
            Comparison trend data
        """
        datasets = []
        all_labels = set()

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for i, (series_name, datapoints) in enumerate(series_map.items()):
            if not datapoints:
                continue

            time_series = self._temporal_aggregator.get_time_series(
                datapoints,
                self._interval,
                fill_gaps=False,
            )

            for point in time_series:
                all_labels.add(point["timestamp"])

            values = [point["value"] for point in time_series]
            labels = [point["timestamp"] for point in time_series]

            color = colors[i % len(colors)]
            datasets.append({
                "label": series_name,
                "data": values,
                "labels": labels,
                "borderColor": color,
                "backgroundColor": f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)",
                "fill": False,
                "tension": 0.1,
            })

        return {
            "type": "trend_comparison",
            "data": {
                "labels": sorted(all_labels),
                "datasets": datasets,
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "legend": {"display": True},
                    "title": {"display": True, "text": "Trend Comparison"},
                },
            },
        }

    def generate_cumulative_trend(
        self,
        datapoints: List[DataPoint],
    ) -> Dict[str, Any]:
        """
        Generate cumulative trend visualization.

        Args:
            datapoints: DataPoints to visualize

        Returns:
            Cumulative trend data
        """
        if not datapoints:
            return self._empty_trend()

        # Sort by timestamp
        sorted_dps = sorted(datapoints, key=lambda dp: dp.timestamp)

        # Calculate cumulative values
        labels = []
        cumulative = []
        running_total = 0

        for dp in sorted_dps:
            running_total += dp.value
            labels.append(dp.timestamp.isoformat())
            cumulative.append(running_total)

        return {
            "type": "cumulative_trend",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": "Cumulative",
                    "data": cumulative,
                    "borderColor": "#1f77b4",
                    "backgroundColor": "rgba(31, 119, 180, 0.2)",
                    "fill": True,
                }],
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "legend": {"display": False},
                    "title": {"display": True, "text": "Cumulative Progress"},
                },
            },
        }

    def generate_progress_trend(
        self,
        session_summaries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate session-over-session progress trend.

        Args:
            session_summaries: List of session summary dicts

        Returns:
            Progress trend data
        """
        labels = []
        accuracy = []
        attempts = []

        for i, summary in enumerate(session_summaries):
            labels.append(f"Session {i + 1}")

            total = summary.get("total_questions", 0)
            correct = summary.get("correct_answers", 0)
            acc = (correct / total * 100) if total > 0 else 0

            accuracy.append(acc)
            attempts.append(total)

        return {
            "type": "progress_trend",
            "data": {
                "labels": labels,
                "datasets": [
                    {
                        "label": "Accuracy (%)",
                        "data": accuracy,
                        "borderColor": "#0571b0",
                        "backgroundColor": "rgba(5, 113, 176, 0.2)",
                        "fill": True,
                        "yAxisID": "y",
                    },
                    {
                        "label": "Questions",
                        "data": attempts,
                        "borderColor": "#ca0020",
                        "backgroundColor": "transparent",
                        "borderDash": [5, 5],
                        "fill": False,
                        "yAxisID": "y1",
                    },
                ],
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "legend": {"display": True},
                    "title": {"display": True, "text": "Session Progress"},
                },
                "scales": {
                    "y": {
                        "type": "linear",
                        "position": "left",
                        "min": 0,
                        "max": 100,
                        "title": {"display": True, "text": "Accuracy (%)"},
                    },
                    "y1": {
                        "type": "linear",
                        "position": "right",
                        "min": 0,
                        "title": {"display": True, "text": "Questions"},
                        "grid": {"drawOnChartArea": False},
                    },
                },
            },
        }

    def _empty_trend(self) -> Dict[str, Any]:
        """Return empty trend structure."""
        return {
            "type": "trend",
            "data": {
                "labels": [],
                "datasets": [],
            },
            "trend_analysis": {
                "trend": "insufficient_data",
                "slope": 0,
                "change_percent": 0,
            },
            "options": {},
        }
