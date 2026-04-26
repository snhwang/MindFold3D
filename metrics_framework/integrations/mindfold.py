"""MindFold3D integration for the metrics framework."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.engine import MetricsEngine
from ..core.definitions import MetricDefinition, MetricType, AggregationType, ThresholdConfig
from ..core.session import MetricSession
from ..collectors.composite import CompositeCollector
from ..collectors.timer import ResponseTimeTracker
from ..aggregators.temporal import TemporalAggregator
from ..assessors.threshold import ThresholdAssessor
from ..assessors.cognitive import CognitiveDimensionAssessor, CognitiveProfile
from ..visualizers.heatmap import HeatmapVisualizer, FeatureValueHeatmapGenerator, HeatmapConfig
from ..visualizers.chart import PerformanceChartGenerator
from ..visualizers.radar import CognitiveProfileVisualizer
from ..visualizers.trend import TrendVisualizer
from ..storage.memory import MemoryStorage


@dataclass
class MindFoldMetricsConfig:
    """Configuration for MindFold3D metrics tracking."""
    min_attempts_for_assessment: int = 3
    weak_area_threshold: float = 0.5
    strong_area_threshold: float = 0.8
    response_time_fast_threshold: float = 2.0
    response_time_slow_threshold: float = 10.0
    track_response_times: bool = True
    track_cognitive_dimensions: bool = True


class MindFoldMetrics:
    """
    Integrated metrics system for MindFold3D spatial cognition training.

    Provides a complete metrics tracking solution including:
    - Performance tracking by feature
    - Response time analysis
    - Cognitive dimension assessment
    - Visualization data generation
    """

    def __init__(
        self,
        config: Optional[MindFoldMetricsConfig] = None,
        storage_backend: Optional[Any] = None,
    ):
        """
        Initialize MindFold3D metrics.

        Args:
            config: Configuration options
            storage_backend: Optional storage backend
        """
        self._config = config or MindFoldMetricsConfig()
        self._engine = MetricsEngine(storage_backend=storage_backend or MemoryStorage())

        # Register MindFold-specific metrics
        self._register_metrics()

        # Initialize collectors
        self._performance_collector = CompositeCollector(
            metric_name="performance",
            engine=self._engine,
        )
        self._response_tracker = ResponseTimeTracker(
            metric_name="response_time",
            fast_threshold=self._config.response_time_fast_threshold,
            slow_threshold=self._config.response_time_slow_threshold,
            engine=self._engine,
        )

        # Initialize assessors
        self._threshold_assessor = ThresholdAssessor()
        self._cognitive_assessor = CognitiveDimensionAssessor(
            min_attempts_per_feature=self._config.min_attempts_for_assessment,
        )

        # Initialize visualizers
        self._heatmap_generator = FeatureValueHeatmapGenerator()
        self._chart_generator = PerformanceChartGenerator()
        self._cognitive_visualizer = CognitiveProfileVisualizer()
        self._trend_visualizer = TrendVisualizer()

        # Session tracking
        self._current_session: Optional[MetricSession] = None

    def _register_metrics(self) -> None:
        """Register MindFold-specific metric definitions."""
        metrics = [
            MetricDefinition(
                name="accuracy",
                metric_type=MetricType.PERCENTAGE,
                description="Percentage of correct responses",
                unit="%",
                aggregation=AggregationType.MEAN,
                higher_is_better=True,
                min_value=0,
                max_value=1,
                thresholds=ThresholdConfig(
                    excellent=0.9, good=0.7, average=0.5,
                    below_average=0.3, poor=0.1
                ),
            ),
            MetricDefinition(
                name="response_time",
                metric_type=MetricType.TIMER,
                description="Time to respond to a question",
                unit="seconds",
                aggregation=AggregationType.MEAN,
                higher_is_better=False,
                min_value=0,
            ),
            MetricDefinition(
                name="performance",
                metric_type=MetricType.GAUGE,
                description="Overall performance score",
                aggregation=AggregationType.MEAN,
                higher_is_better=True,
                min_value=0,
                max_value=1,
            ),
        ]

        for metric in metrics:
            self._engine.register_metric(metric)

    # ==================== Session Management ====================

    def start_session(
        self,
        user_id: Optional[str] = None,
        session_name: str = "",
    ) -> MetricSession:
        """Start a new training session."""
        self._current_session = self._engine.start_session(
            name=session_name,
            user_id=user_id,
        )
        return self._current_session

    def end_session(self) -> Optional[Dict[str, Any]]:
        """End the current session and return summary."""
        if self._current_session:
            session = self._engine.end_session()
            return self.get_session_summary()
        return None

    @property
    def current_session(self) -> Optional[MetricSession]:
        """Get the current active session."""
        return self._current_session

    # ==================== Recording ====================

    def record_attempt(
        self,
        correct: bool,
        response_time: float,
        target_features: Dict[str, Any],
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record a training attempt.

        Args:
            correct: Whether the response was correct
            response_time: Time taken to respond (seconds)
            target_features: Features of the target shape
            user_id: Optional user identifier
            metadata: Additional metadata

        Returns:
            Recording result with updated stats
        """
        # Record to composite collector
        result = self._performance_collector.record_attempt(
            correct=correct,
            response_time=response_time,
            features=target_features,
        )

        # Record response time
        if self._config.track_response_times:
            self._response_tracker.observe(response_time)

        # Record to engine for persistence
        self._engine.record(
            metric_name="accuracy",
            value=1.0 if correct else 0.0,
            dimensions=target_features,
            metadata={
                "response_time": response_time,
                **(metadata or {}),
            },
            user_id=user_id,
        )

        return {
            "recorded": True,
            "correct": correct,
            "response_time": response_time,
            "current_accuracy": self._performance_collector.success_rate,
            "total_attempts": self._performance_collector.total_attempts,
        }

    # ==================== Statistics ====================

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        return {
            "total_questions": self._performance_collector.total_attempts,
            "correct_answers": self._performance_collector.total_correct,
            "accuracy": self._performance_collector.success_rate,
            "avg_response_time": self._performance_collector.avg_response_time,
            "response_time_stats": self._response_tracker.get_stats(),
        }

    def get_feature_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics by feature."""
        return self._performance_collector.get_all_feature_stats()

    def get_weak_areas(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get areas needing improvement."""
        return self._performance_collector.get_weak_areas(
            min_attempts=self._config.min_attempts_for_assessment,
            max_success_rate=self._config.weak_area_threshold,
            limit=limit,
        )

    def get_strong_areas(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get areas of strength."""
        return self._performance_collector.get_strong_areas(
            min_attempts=self._config.min_attempts_for_assessment,
            min_success_rate=self._config.strong_area_threshold,
            limit=limit,
        )

    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary."""
        return {
            **self.get_session_stats(),
            "weak_areas": self.get_weak_areas(),
            "strong_areas": self.get_strong_areas(),
            "feature_stats": self.get_feature_stats(),
        }

    # ==================== Cognitive Assessment ====================

    def get_cognitive_profile(
        self,
        user_id: Optional[str] = None,
    ) -> CognitiveProfile:
        """Generate cognitive profile from performance data."""
        feature_stats = self.get_feature_stats()
        return self._cognitive_assessor.create_cognitive_profile(
            feature_stats=feature_stats,
            user_id=user_id,
        )

    def get_cognitive_dimension_scores(self) -> Dict[str, float]:
        """Get scores for each cognitive dimension."""
        profile = self.get_cognitive_profile()
        return {
            dim_name: result.score
            for dim_name, result in profile.dimensions.items()
        }

    # ==================== Visualization ====================

    def get_heatmap_data(
        self,
        features: Optional[List[str]] = None,
        min_attempts: int = 1,
    ) -> Dict[str, Any]:
        """Generate heatmap data for feature performance."""
        feature_stats = self.get_feature_stats()
        return self._heatmap_generator.generate_from_feature_stats(
            feature_stats=feature_stats,
            feature_order=features,
            min_attempts=min_attempts,
        )

    def get_success_rate_chart(
        self,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate success rate bar chart data."""
        feature_stats = self.get_feature_stats()
        return self._chart_generator.generate_success_rate_chart(
            feature_stats=feature_stats,
            features=features,
            min_attempts=self._config.min_attempts_for_assessment,
        )

    def get_response_time_chart(
        self,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate response time chart data."""
        feature_stats = self.get_feature_stats()
        return self._chart_generator.generate_response_time_chart(
            feature_stats=feature_stats,
            features=features,
        )

    def get_cognitive_radar_chart(self) -> Dict[str, Any]:
        """Generate cognitive profile radar chart."""
        profile = self.get_cognitive_profile()
        return self._cognitive_visualizer.generate_full_profile(profile.to_dict())

    def get_trend_data(self, interval: str = "day") -> Dict[str, Any]:
        """Generate trend visualization data."""
        datapoints = self._engine.query(metric_name="accuracy")
        return self._trend_visualizer.generate(datapoints)

    # ==================== Scorecard ====================

    def get_scorecard(self, mode: str = "session") -> Dict[str, Any]:
        """
        Generate comprehensive scorecard data.

        Args:
            mode: "session" for current session, "cumulative" for all time

        Returns:
            Complete scorecard data structure
        """
        stats = self.get_session_stats()
        feature_stats = self.get_feature_stats()

        # Build feature scorecard
        feature_scorecard = {}
        for feature_name, values in feature_stats.items():
            feature_scorecard[feature_name] = {}
            for value_key, value_stats in values.items():
                if isinstance(value_stats, dict):
                    correct = value_stats.get("correct", 0)
                    incorrect = value_stats.get("incorrect", 0)
                    total = correct + incorrect
                    response_times = value_stats.get("response_times", [])

                    feature_scorecard[feature_name][value_key] = {
                        "correct": correct,
                        "incorrect": incorrect,
                        "total_attempts": total,
                        "success_rate": correct / total if total > 0 else 0,
                        "avg_response_time": (
                            sum(response_times) / len(response_times)
                            if response_times else 0
                        ),
                    }

        return {
            "mode": mode,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_questions": stats["total_questions"],
                "correct_answers": stats["correct_answers"],
                "accuracy": stats["accuracy"],
                "avg_response_time": stats["avg_response_time"],
            },
            "feature_scorecard": feature_scorecard,
            "weak_areas": self.get_weak_areas(),
            "strong_areas": self.get_strong_areas(),
            "cognitive_profile": self.get_cognitive_profile().to_dict(),
            "visualizations": {
                "heatmap": self.get_heatmap_data(),
                "success_chart": self.get_success_rate_chart(),
                "response_time_chart": self.get_response_time_chart(),
                "cognitive_radar": self.get_cognitive_radar_chart(),
            },
        }

    # ==================== Data Management ====================

    def reset_session(self) -> None:
        """Reset current session data."""
        self._performance_collector.reset()
        self._response_tracker.reset()
        if self._current_session:
            self._engine.end_session()
            self._current_session = None

    def reset_all(self) -> None:
        """Reset all data including persistent storage."""
        self.reset_session()
        self._engine.clear(confirm=True)

    def export_data(self, format: str = "json") -> Any:
        """Export all metrics data."""
        return self._engine.export(format)

    def import_feature_stats(
        self,
        feature_stats: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Import existing feature stats into the collector.

        Useful for restoring state from database.
        """
        self._performance_collector = CompositeCollector.from_dict(
            data={
                "total_correct": sum(
                    v.get("correct", 0)
                    for f in feature_stats.values()
                    for v in f.values()
                    if isinstance(v, dict)
                ),
                "total_incorrect": sum(
                    v.get("incorrect", 0)
                    for f in feature_stats.values()
                    for v in f.values()
                    if isinstance(v, dict)
                ),
                "response_times": [],
                "performance": feature_stats,
            },
            metric_name="performance",
            engine=self._engine,
        )
