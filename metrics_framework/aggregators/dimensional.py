"""Dimensional aggregator for multi-dimensional analysis."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.definitions import AggregationType
from ..core.datapoint import DataPoint
from ..utils.stats import (
    calculate_mean,
    calculate_median,
    calculate_std,
    calculate_percentile,
    calculate_variance,
    calculate_mode,
)


class DimensionalAggregator:
    """
    Aggregator for dimensional metric analysis.

    Supports aggregating metrics across multiple dimensions
    for cross-cutting analysis.
    """

    def __init__(
        self,
        aggregation: Union[AggregationType, str] = AggregationType.MEAN,
        percentile_value: float = 95,
    ):
        """
        Initialize the dimensional aggregator.

        Args:
            aggregation: Default aggregation method
            percentile_value: Percentile value for PERCENTILE aggregation
        """
        if isinstance(aggregation, str):
            aggregation = AggregationType(aggregation)
        self._aggregation = aggregation
        self._percentile_value = percentile_value

    def _aggregate_values(
        self,
        values: List[float],
        aggregation: Optional[AggregationType] = None,
    ) -> float:
        """Aggregate a list of values."""
        if not values:
            return 0.0

        agg = aggregation or self._aggregation

        if agg == AggregationType.SUM:
            return sum(values)
        elif agg == AggregationType.MEAN:
            return calculate_mean(values)
        elif agg == AggregationType.MEDIAN:
            return calculate_median(values)
        elif agg == AggregationType.MIN:
            return min(values)
        elif agg == AggregationType.MAX:
            return max(values)
        elif agg == AggregationType.COUNT:
            return float(len(values))
        elif agg == AggregationType.FIRST:
            return values[0]
        elif agg == AggregationType.LAST:
            return values[-1]
        elif agg == AggregationType.STD:
            return calculate_std(values)
        elif agg == AggregationType.VARIANCE:
            return calculate_variance(values)
        elif agg == AggregationType.PERCENTILE:
            return calculate_percentile(values, self._percentile_value)
        elif agg == AggregationType.MODE:
            mode = calculate_mode(values)
            return mode if mode is not None else calculate_mean(values)
        else:
            return calculate_mean(values)

    def aggregate_by_dimension(
        self,
        datapoints: List[DataPoint],
        dimension: str,
        aggregation: Optional[Union[AggregationType, str]] = None,
    ) -> Dict[Any, float]:
        """
        Aggregate datapoints by a single dimension.

        Args:
            datapoints: DataPoints to aggregate
            dimension: Dimension name to group by
            aggregation: Aggregation method

        Returns:
            Dict mapping dimension values to aggregated values
        """
        if isinstance(aggregation, str):
            aggregation = AggregationType(aggregation)

        groups: Dict[Any, List[float]] = defaultdict(list)
        for dp in datapoints:
            dim_value = dp.dimensions.get(dimension)
            if dim_value is not None:
                groups[dim_value].append(dp.value)

        return {
            dim_value: self._aggregate_values(values, aggregation)
            for dim_value, values in groups.items()
        }

    def aggregate_grouped(
        self,
        datapoints: List[DataPoint],
        group_by: List[str],
        aggregation: Optional[Union[AggregationType, str]] = None,
    ) -> Dict[Tuple, float]:
        """
        Aggregate datapoints by multiple dimensions.

        Args:
            datapoints: DataPoints to aggregate
            group_by: List of dimension names to group by
            aggregation: Aggregation method

        Returns:
            Dict mapping dimension value tuples to aggregated values
        """
        if isinstance(aggregation, str):
            aggregation = AggregationType(aggregation)

        groups: Dict[Tuple, List[float]] = defaultdict(list)
        for dp in datapoints:
            key = tuple(dp.dimensions.get(dim) for dim in group_by)
            groups[key].append(dp.value)

        return {
            key: self._aggregate_values(values, aggregation)
            for key, values in groups.items()
        }

    def get_dimension_stats(
        self,
        datapoints: List[DataPoint],
        dimension: str,
    ) -> Dict[Any, Dict[str, float]]:
        """
        Get comprehensive stats per dimension value.

        Args:
            datapoints: DataPoints to analyze
            dimension: Dimension to analyze

        Returns:
            Dict mapping dimension values to stat dicts
        """
        groups: Dict[Any, List[float]] = defaultdict(list)
        for dp in datapoints:
            dim_value = dp.dimensions.get(dimension)
            if dim_value is not None:
                groups[dim_value].append(dp.value)

        result = {}
        for dim_value, values in groups.items():
            result[dim_value] = {
                "count": len(values),
                "sum": sum(values),
                "mean": calculate_mean(values),
                "median": calculate_median(values),
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "std": calculate_std(values),
            }

        return result

    def get_cross_dimension_matrix(
        self,
        datapoints: List[DataPoint],
        x_dimension: str,
        y_dimension: str,
        aggregation: Optional[Union[AggregationType, str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a matrix of aggregated values across two dimensions.

        Args:
            datapoints: DataPoints to analyze
            x_dimension: Dimension for X axis
            y_dimension: Dimension for Y axis
            aggregation: Aggregation method

        Returns:
            Matrix data structure
        """
        if isinstance(aggregation, str):
            aggregation = AggregationType(aggregation)

        # Group by both dimensions
        groups: Dict[Tuple[Any, Any], List[float]] = defaultdict(list)
        x_values = set()
        y_values = set()

        for dp in datapoints:
            x_val = dp.dimensions.get(x_dimension)
            y_val = dp.dimensions.get(y_dimension)
            if x_val is not None and y_val is not None:
                groups[(x_val, y_val)].append(dp.value)
                x_values.add(x_val)
                y_values.add(y_val)

        # Build matrix
        x_labels = sorted(x_values, key=str)
        y_labels = sorted(y_values, key=str)

        matrix = []
        for y_val in y_labels:
            row = []
            for x_val in x_labels:
                values = groups.get((x_val, y_val), [])
                agg_value = self._aggregate_values(values, aggregation)
                row.append({
                    "value": agg_value,
                    "count": len(values),
                    "x": x_val,
                    "y": y_val,
                })
            matrix.append(row)

        return {
            "x_dimension": x_dimension,
            "y_dimension": y_dimension,
            "x_labels": x_labels,
            "y_labels": y_labels,
            "matrix": matrix,
            "aggregation": str(aggregation or self._aggregation),
        }

    def get_dimension_ranking(
        self,
        datapoints: List[DataPoint],
        dimension: str,
        aggregation: Optional[Union[AggregationType, str]] = None,
        ascending: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank dimension values by aggregated metric.

        Args:
            datapoints: DataPoints to analyze
            dimension: Dimension to rank
            aggregation: Aggregation method
            ascending: Sort order (True = lowest first)
            limit: Maximum number of results

        Returns:
            Ranked list of dimension values
        """
        stats = self.get_dimension_stats(datapoints, dimension)

        agg = aggregation or self._aggregation
        if isinstance(agg, str):
            agg = AggregationType(agg)

        ranked = [
            {
                "dimension_value": dim_value,
                "aggregated_value": self._aggregate_values(
                    [v for dp in datapoints
                     if dp.dimensions.get(dimension) == dim_value
                     for v in [dp.value]],
                    agg
                ),
                "count": s["count"],
                **s,
            }
            for dim_value, s in stats.items()
        ]

        ranked.sort(key=lambda x: x["aggregated_value"], reverse=not ascending)

        if limit:
            ranked = ranked[:limit]

        return ranked

    def get_dimension_comparison(
        self,
        datapoints: List[DataPoint],
        dimension: str,
        value1: Any,
        value2: Any,
    ) -> Dict[str, Any]:
        """
        Compare two dimension values.

        Args:
            datapoints: DataPoints to analyze
            dimension: Dimension to compare
            value1: First dimension value
            value2: Second dimension value

        Returns:
            Comparison results
        """
        values1 = [dp.value for dp in datapoints
                   if dp.dimensions.get(dimension) == value1]
        values2 = [dp.value for dp in datapoints
                   if dp.dimensions.get(dimension) == value2]

        def get_stats(values):
            if not values:
                return {"count": 0, "mean": 0, "std": 0}
            return {
                "count": len(values),
                "mean": calculate_mean(values),
                "median": calculate_median(values),
                "std": calculate_std(values),
                "min": min(values),
                "max": max(values),
            }

        stats1 = get_stats(values1)
        stats2 = get_stats(values2)

        # Calculate difference
        diff_mean = stats1["mean"] - stats2["mean"]
        diff_percent = 0.0
        if stats2["mean"] != 0:
            diff_percent = (diff_mean / abs(stats2["mean"])) * 100

        return {
            "dimension": dimension,
            "value1": value1,
            "value2": value2,
            "stats1": stats1,
            "stats2": stats2,
            "difference": {
                "mean": diff_mean,
                "percent": diff_percent,
            },
        }
