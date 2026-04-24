"""Radar chart visualization generator."""

from typing import Any, Dict, List, Optional


class RadarVisualizer:
    """
    Visualizer for radar/spider chart data.

    Ideal for visualizing cognitive profiles and multi-dimensional assessments.
    """

    def __init__(
        self,
        max_value: float = 1.0,
        show_legend: bool = True,
    ):
        """
        Initialize the radar visualizer.

        Args:
            max_value: Maximum value for the radar scale
            show_legend: Whether to show legend
        """
        self._max_value = max_value
        self._show_legend = show_legend

    def generate(
        self,
        dimensions: Dict[str, float],
        title: str = "Profile",
    ) -> Dict[str, Any]:
        """
        Generate radar chart data.

        Args:
            dimensions: Dict of dimension_name -> value
            title: Chart title

        Returns:
            Radar chart data structure
        """
        labels = list(dimensions.keys())
        values = list(dimensions.values())

        return {
            "type": "radar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": title,
                    "data": values,
                    "backgroundColor": "rgba(5, 113, 176, 0.2)",
                    "borderColor": "#0571b0",
                    "pointBackgroundColor": "#0571b0",
                    "pointBorderColor": "#fff",
                    "pointHoverBackgroundColor": "#fff",
                    "pointHoverBorderColor": "#0571b0",
                }],
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "legend": {"display": self._show_legend},
                    "title": {"display": True, "text": title},
                },
                "scales": {
                    "r": {
                        "angleLines": {"display": True},
                        "suggestedMin": 0,
                        "suggestedMax": self._max_value,
                        "ticks": {
                            "stepSize": self._max_value / 5,
                        },
                    },
                },
            },
        }

    def generate_comparison(
        self,
        profiles: Dict[str, Dict[str, float]],
        title: str = "Profile Comparison",
    ) -> Dict[str, Any]:
        """
        Generate radar chart comparing multiple profiles.

        Args:
            profiles: Dict of profile_name -> {dimension: value}
            title: Chart title

        Returns:
            Comparison radar chart data
        """
        # Get all dimensions
        all_dimensions = set()
        for dims in profiles.values():
            all_dimensions.update(dims.keys())
        labels = sorted(all_dimensions)

        # Colors for different profiles
        colors = [
            ("rgba(5, 113, 176, 0.2)", "#0571b0"),
            ("rgba(202, 0, 32, 0.2)", "#ca0020"),
            ("rgba(46, 204, 113, 0.2)", "#2ecc71"),
            ("rgba(155, 89, 182, 0.2)", "#9b59b6"),
        ]

        datasets = []
        for i, (profile_name, dimensions) in enumerate(profiles.items()):
            values = [dimensions.get(dim, 0) for dim in labels]
            bg_color, border_color = colors[i % len(colors)]

            datasets.append({
                "label": profile_name,
                "data": values,
                "backgroundColor": bg_color,
                "borderColor": border_color,
                "pointBackgroundColor": border_color,
            })

        return {
            "type": "radar",
            "data": {
                "labels": labels,
                "datasets": datasets,
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "legend": {"display": True},
                    "title": {"display": True, "text": title},
                },
                "scales": {
                    "r": {
                        "angleLines": {"display": True},
                        "suggestedMin": 0,
                        "suggestedMax": self._max_value,
                    },
                },
            },
        }

    def generate_cognitive_profile(
        self,
        cognitive_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate radar chart from cognitive assessment results.

        Args:
            cognitive_results: Dict of dimension_name -> CognitiveDimensionResult

        Returns:
            Cognitive profile radar chart
        """
        dimensions = {}

        for dim_name, result in cognitive_results.items():
            if hasattr(result, "score"):
                dimensions[dim_name.replace("_", " ").title()] = result.score
            elif isinstance(result, dict):
                dimensions[dim_name.replace("_", " ").title()] = result.get("score", 0)

        return self.generate(dimensions, "Cognitive Profile")

    def generate_strength_weakness(
        self,
        scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Generate radar chart highlighting strengths and weaknesses.

        Args:
            scores: Dict of dimension -> score

        Returns:
            Strength/weakness radar chart
        """
        labels = list(scores.keys())
        values = list(scores.values())

        # Calculate average
        avg = sum(values) / len(values) if values else 0

        # Create average line dataset
        avg_values = [avg] * len(labels)

        return {
            "type": "radar",
            "data": {
                "labels": labels,
                "datasets": [
                    {
                        "label": "Your Performance",
                        "data": values,
                        "backgroundColor": "rgba(5, 113, 176, 0.2)",
                        "borderColor": "#0571b0",
                        "pointBackgroundColor": [
                            "#0571b0" if v >= avg else "#ca0020"
                            for v in values
                        ],
                    },
                    {
                        "label": "Average",
                        "data": avg_values,
                        "backgroundColor": "transparent",
                        "borderColor": "#999999",
                        "borderDash": [5, 5],
                        "pointRadius": 0,
                    },
                ],
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "legend": {"display": True},
                    "title": {"display": True, "text": "Strengths & Weaknesses"},
                },
                "scales": {
                    "r": {
                        "suggestedMin": 0,
                        "suggestedMax": self._max_value,
                    },
                },
            },
        }


class CognitiveProfileVisualizer:
    """Specialized visualizer for cognitive profiles."""

    def __init__(self):
        """Initialize the visualizer."""
        self._radar = RadarVisualizer()

    def generate_full_profile(
        self,
        profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate complete cognitive profile visualization.

        Args:
            profile: CognitiveProfile dict or object

        Returns:
            Full profile visualization data
        """
        # Extract dimension scores
        dimensions = {}
        if isinstance(profile, dict):
            dim_data = profile.get("dimensions", {})
        else:
            dim_data = getattr(profile, "dimensions", {})

        for dim_name, dim_result in dim_data.items():
            if isinstance(dim_result, dict):
                score = dim_result.get("score", 0)
            else:
                score = getattr(dim_result, "score", 0)

            # Format dimension name
            formatted_name = dim_name.replace("_", " ").title()
            dimensions[formatted_name] = score

        # Generate radar chart
        radar_data = self._radar.generate(
            dimensions,
            title="Cognitive Profile",
        )

        # Get strengths and weaknesses
        strengths = profile.get("strengths", []) if isinstance(profile, dict) else getattr(profile, "strengths", [])
        weaknesses = profile.get("weaknesses", []) if isinstance(profile, dict) else getattr(profile, "weaknesses", [])
        recommendations = profile.get("recommendations", []) if isinstance(profile, dict) else getattr(profile, "recommendations", [])

        return {
            "radar_chart": radar_data,
            "dimensions": dimensions,
            "summary": {
                "overall_score": profile.get("overall_score", 0) if isinstance(profile, dict) else getattr(profile, "overall_score", 0),
                "overall_level": profile.get("overall_level", "unknown") if isinstance(profile, dict) else getattr(profile, "overall_level", "unknown"),
                "strengths": [s.replace("_", " ").title() for s in strengths],
                "weaknesses": [w.replace("_", " ").title() for w in weaknesses],
                "recommendations": recommendations,
            },
        }

    def generate_progress_comparison(
        self,
        profiles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate progress comparison across multiple profiles.

        Args:
            profiles: List of profiles (chronologically ordered)

        Returns:
            Progress comparison visualization
        """
        if len(profiles) < 2:
            return {"error": "Need at least 2 profiles for comparison"}

        # Compare first and last
        first_profile = profiles[0]
        last_profile = profiles[-1]

        comparison = {}

        # Get dimension scores from first profile
        first_dims = first_profile.get("dimensions", {})
        last_dims = last_profile.get("dimensions", {})

        profile_map = {}

        for dim_name in first_dims:
            first_score = first_dims[dim_name].get("score", 0) if isinstance(first_dims[dim_name], dict) else 0
            last_score = last_dims.get(dim_name, {}).get("score", 0) if isinstance(last_dims.get(dim_name, {}), dict) else 0

            formatted_name = dim_name.replace("_", " ").title()
            comparison[formatted_name] = {
                "first": first_score,
                "last": last_score,
                "change": last_score - first_score,
            }

            if "First Session" not in profile_map:
                profile_map["First Session"] = {}
            if "Latest Session" not in profile_map:
                profile_map["Latest Session"] = {}

            profile_map["First Session"][formatted_name] = first_score
            profile_map["Latest Session"][formatted_name] = last_score

        return {
            "comparison_radar": self._radar.generate_comparison(
                profile_map,
                title="Progress Over Time",
            ),
            "dimension_changes": comparison,
        }
