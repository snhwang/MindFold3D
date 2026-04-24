"""Comparative performance assessor for multi-dimensional analysis."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.datapoint import DataPoint
from ..aggregators.dimensional import DimensionalAggregator
from ..utils.stats import calculate_mean, calculate_std, calculate_percentile


@dataclass
class ComparisonResult:
    """Result of a comparative assessment."""
    dimension: str
    value1: Any
    value2: Any
    metric_name: str
    stats1: Dict[str, float]
    stats2: Dict[str, float]
    difference: float
    percent_difference: float
    is_significant: bool
    effect_size: float
    winner: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimension": self.dimension,
            "value1": self.value1,
            "value2": self.value2,
            "metric_name": self.metric_name,
            "stats1": self.stats1,
            "stats2": self.stats2,
            "difference": self.difference,
            "percent_difference": self.percent_difference,
            "is_significant": self.is_significant,
            "effect_size": self.effect_size,
            "winner": self.winner,
            "details": self.details,
        }


@dataclass
class RankingResult:
    """Result of a ranking assessment."""
    dimension: str
    rankings: List[Dict[str, Any]]
    metric_name: str
    total_items: int
    aggregation: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimension": self.dimension,
            "rankings": self.rankings,
            "metric_name": self.metric_name,
            "total_items": self.total_items,
            "aggregation": self.aggregation,
            "details": self.details,
        }


class ComparativeAssessor:
    """
    Assessor for comparing performance across dimensions.

    Supports pairwise comparisons, rankings, and statistical
    significance testing.
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        effect_size_threshold: float = 0.5,
        higher_is_better: bool = True,
    ):
        """
        Initialize the comparative assessor.

        Args:
            significance_level: Threshold for statistical significance
            effect_size_threshold: Threshold for practical significance
            higher_is_better: Whether higher values are better
        """
        self._significance_level = significance_level
        self._effect_size_threshold = effect_size_threshold
        self._higher_is_better = higher_is_better
        self._aggregator = DimensionalAggregator()

    def compare_dimension_values(
        self,
        datapoints: List[DataPoint],
        dimension: str,
        value1: Any,
        value2: Any,
    ) -> ComparisonResult:
        """
        Compare two values of a dimension.

        Args:
            datapoints: DataPoints to analyze
            dimension: Dimension to compare
            value1: First value
            value2: Second value

        Returns:
            Comparison result
        """
        # Filter by dimension value
        values1 = [
            dp.value for dp in datapoints
            if dp.dimensions.get(dimension) == value1
        ]
        values2 = [
            dp.value for dp in datapoints
            if dp.dimensions.get(dimension) == value2
        ]

        metric_name = datapoints[0].metric_name if datapoints else "unknown"

        # Calculate stats
        stats1 = self._calculate_stats(values1)
        stats2 = self._calculate_stats(values2)

        # Calculate difference
        difference = stats1["mean"] - stats2["mean"]
        percent_diff = 0.0
        if stats2["mean"] != 0:
            percent_diff = (difference / abs(stats2["mean"])) * 100

        # Calculate effect size (Cohen's d)
        pooled_std = (
            (stats1["std"] ** 2 + stats2["std"] ** 2) / 2
        ) ** 0.5
        effect_size = abs(difference) / pooled_std if pooled_std > 0 else 0

        # Determine significance
        is_significant = (
            effect_size > self._effect_size_threshold and
            stats1["count"] >= 3 and
            stats2["count"] >= 3
        )

        # Determine winner
        winner = None
        if is_significant:
            if self._higher_is_better:
                winner = value1 if difference > 0 else value2
            else:
                winner = value1 if difference < 0 else value2

        return ComparisonResult(
            dimension=dimension,
            value1=value1,
            value2=value2,
            metric_name=metric_name,
            stats1=stats1,
            stats2=stats2,
            difference=difference,
            percent_difference=percent_diff,
            is_significant=is_significant,
            effect_size=effect_size,
            winner=winner,
            details={
                "higher_is_better": self._higher_is_better,
                "significance_level": self._significance_level,
                "effect_size_threshold": self._effect_size_threshold,
            },
        )

    def rank_dimension_values(
        self,
        datapoints: List[DataPoint],
        dimension: str,
        limit: Optional[int] = None,
        min_count: int = 1,
    ) -> RankingResult:
        """
        Rank dimension values by metric performance.

        Args:
            datapoints: DataPoints to analyze
            dimension: Dimension to rank
            limit: Maximum number of rankings to return
            min_count: Minimum count to include in ranking

        Returns:
            Ranking result
        """
        metric_name = datapoints[0].metric_name if datapoints else "unknown"

        # Get stats for each dimension value
        stats = self._aggregator.get_dimension_stats(datapoints, dimension)

        # Filter by minimum count
        filtered_stats = {
            k: v for k, v in stats.items()
            if v["count"] >= min_count
        }

        # Create ranking list
        rankings = []
        for dim_value, dim_stats in filtered_stats.items():
            rankings.append({
                "dimension_value": dim_value,
                "mean": dim_stats["mean"],
                "count": dim_stats["count"],
                "std": dim_stats["std"],
                "min": dim_stats["min"],
                "max": dim_stats["max"],
            })

        # Sort by mean (higher is better or lower is better)
        rankings.sort(
            key=lambda x: x["mean"],
            reverse=self._higher_is_better,
        )

        # Add rank numbers
        for i, r in enumerate(rankings):
            r["rank"] = i + 1

        # Apply limit
        if limit:
            rankings = rankings[:limit]

        return RankingResult(
            dimension=dimension,
            rankings=rankings,
            metric_name=metric_name,
            total_items=len(filtered_stats),
            aggregation="mean",
            details={
                "min_count": min_count,
                "higher_is_better": self._higher_is_better,
            },
        )

    def identify_outliers(
        self,
        datapoints: List[DataPoint],
        dimension: str,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Identify outlier dimension values.

        Args:
            datapoints: DataPoints to analyze
            dimension: Dimension to analyze
            method: Outlier detection method (iqr or zscore)
            threshold: Threshold for outlier detection

        Returns:
            Outlier analysis results
        """
        stats = self._aggregator.get_dimension_stats(datapoints, dimension)

        if len(stats) < 4:
            return {
                "error": "Insufficient dimension values for outlier detection",
                "count": len(stats),
            }

        means = [s["mean"] for s in stats.values()]
        dim_values = list(stats.keys())

        outliers = []
        non_outliers = []

        if method == "iqr":
            q1 = calculate_percentile(means, 25)
            q3 = calculate_percentile(means, 75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr

            for dim_value, mean in zip(dim_values, means):
                s = stats[dim_value]
                item = {
                    "dimension_value": dim_value,
                    "mean": mean,
                    "count": s["count"],
                }
                if mean < lower or mean > upper:
                    item["reason"] = "below_lower" if mean < lower else "above_upper"
                    outliers.append(item)
                else:
                    non_outliers.append(item)

        elif method == "zscore":
            overall_mean = calculate_mean(means)
            overall_std = calculate_std(means)

            if overall_std > 0:
                for dim_value, mean in zip(dim_values, means):
                    s = stats[dim_value]
                    z = abs((mean - overall_mean) / overall_std)
                    item = {
                        "dimension_value": dim_value,
                        "mean": mean,
                        "count": s["count"],
                        "z_score": z,
                    }
                    if z > threshold:
                        item["reason"] = "high_z_score"
                        outliers.append(item)
                    else:
                        non_outliers.append(item)

        return {
            "dimension": dimension,
            "method": method,
            "threshold": threshold,
            "outliers": outliers,
            "outlier_count": len(outliers),
            "total_values": len(stats),
        }

    def cross_dimension_analysis(
        self,
        datapoints: List[DataPoint],
        dimension1: str,
        dimension2: str,
    ) -> Dict[str, Any]:
        """
        Analyze performance across two dimensions.

        Args:
            datapoints: DataPoints to analyze
            dimension1: First dimension
            dimension2: Second dimension

        Returns:
            Cross-dimension analysis
        """
        matrix = self._aggregator.get_cross_dimension_matrix(
            datapoints, dimension1, dimension2
        )

        # Find best and worst combinations
        all_cells = []
        for row in matrix["matrix"]:
            for cell in row:
                if cell["count"] > 0:
                    all_cells.append(cell)

        if not all_cells:
            return {"error": "No data for cross-dimension analysis"}

        # Sort to find extremes
        sorted_cells = sorted(
            all_cells,
            key=lambda c: c["value"],
            reverse=self._higher_is_better,
        )

        best_combinations = sorted_cells[:5] if len(sorted_cells) >= 5 else sorted_cells
        worst_combinations = sorted_cells[-5:] if len(sorted_cells) >= 5 else []

        return {
            "dimension1": dimension1,
            "dimension2": dimension2,
            "matrix": matrix,
            "best_combinations": best_combinations,
            "worst_combinations": worst_combinations,
            "total_combinations": len(all_cells),
        }

    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values."""
        if not values:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        return {
            "count": len(values),
            "mean": calculate_mean(values),
            "std": calculate_std(values),
            "min": min(values),
            "max": max(values),
        }
