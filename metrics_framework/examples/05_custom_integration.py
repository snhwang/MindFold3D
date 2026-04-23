#!/usr/bin/env python
"""
Example 5: Custom Integration
=============================

This example shows how to build a domain-specific integration
for your own application. We'll create an e-commerce metrics tracker.
"""

from metrics_framework import MetricsEngine, MetricDefinition
from metrics_framework.core.definitions import ThresholdConfig
from metrics_framework.storage import JSONStorage


class EcommerceMetrics:
    """
    E-commerce specific metrics integration.

    Tracks conversions, cart abandonment, order values, and page performance.
    """

    def __init__(self, storage_path: str = None):
        """
        Initialize e-commerce metrics.

        Args:
            storage_path: Optional path for JSON persistence
        """
        storage = JSONStorage(storage_path) if storage_path else None
        self._engine = MetricsEngine(storage_backend=storage)
        self._setup_metrics()

    def _setup_metrics(self):
        """Define all e-commerce specific metrics."""
        self._engine.register_metrics([
            MetricDefinition(
                name="conversion",
                metric_type="boolean",
                description="Whether a visit resulted in a purchase",
                aggregation="mean",  # Mean of 0/1 = conversion rate
                higher_is_better=True,
                thresholds=ThresholdConfig(
                    excellent=0.05,   # 5% conversion is excellent
                    good=0.03,
                    average=0.02,
                    below_average=0.01,
                    poor=0.005,
                ),
            ),
            MetricDefinition(
                name="cart_abandonment",
                metric_type="boolean",
                description="Whether a cart was abandoned",
                aggregation="mean",
                higher_is_better=False,  # Lower is better
                thresholds=ThresholdConfig(
                    excellent=0.3,   # 30% abandonment is excellent
                    good=0.5,
                    average=0.65,
                    below_average=0.75,
                    poor=0.85,
                ),
            ),
            MetricDefinition(
                name="order_value",
                metric_type="gauge",
                description="Value of completed orders",
                unit="USD",
                aggregation="mean",
                higher_is_better=True,
            ),
            MetricDefinition(
                name="page_load_time",
                metric_type="timer",
                description="Page load time",
                unit="seconds",
                aggregation="median",
                higher_is_better=False,
            ),
        ])

    # --- Recording Methods ---

    def record_visit(
        self,
        converted: bool,
        traffic_source: str,
        device: str,
        country: str = "US",
    ):
        """Record a site visit."""
        self._engine.record(
            "conversion",
            1.0 if converted else 0.0,
            dimensions={
                "traffic_source": traffic_source,
                "device": device,
                "country": country,
            }
        )

    def record_cart_event(
        self,
        abandoned: bool,
        cart_value: float,
        traffic_source: str,
        device: str,
    ):
        """Record a cart event (checkout or abandonment)."""
        self._engine.record(
            "cart_abandonment",
            1.0 if abandoned else 0.0,
            dimensions={
                "traffic_source": traffic_source,
                "device": device,
                "value_bucket": self._get_value_bucket(cart_value),
            }
        )

    def record_order(
        self,
        order_value: float,
        traffic_source: str,
        device: str,
        product_category: str,
    ):
        """Record a completed order."""
        self._engine.record(
            "order_value",
            order_value,
            dimensions={
                "traffic_source": traffic_source,
                "device": device,
                "product_category": product_category,
            }
        )

    def record_page_load(
        self,
        load_time: float,
        page_type: str,
        device: str,
    ):
        """Record a page load time."""
        self._engine.record(
            "page_load_time",
            load_time,
            dimensions={
                "page_type": page_type,
                "device": device,
            }
        )

    # --- Analysis Methods ---

    def get_conversion_rate(self, **filters) -> float:
        """Get overall conversion rate."""
        return self._engine.aggregate("conversion", **filters)

    def get_conversion_by_source(self) -> dict:
        """Get conversion rate by traffic source."""
        return self._engine.aggregate("conversion", group_by=["traffic_source"])

    def get_conversion_by_device(self) -> dict:
        """Get conversion rate by device type."""
        return self._engine.aggregate("conversion", group_by=["device"])

    def get_abandonment_rate(self) -> float:
        """Get cart abandonment rate."""
        return self._engine.aggregate("cart_abandonment")

    def get_avg_order_value(self, **filters) -> float:
        """Get average order value."""
        return self._engine.aggregate("order_value", **filters)

    def get_aov_by_category(self) -> dict:
        """Get average order value by product category."""
        return self._engine.aggregate("order_value", group_by=["product_category"])

    def get_dashboard(self) -> dict:
        """Get complete dashboard data."""
        return {
            "summary": {
                "conversion_rate": self.get_conversion_rate(),
                "abandonment_rate": self.get_abandonment_rate(),
                "avg_order_value": self.get_avg_order_value(),
                "avg_page_load": self._engine.aggregate("page_load_time"),
            },
            "by_source": self.get_conversion_by_source(),
            "by_device": self.get_conversion_by_device(),
            "by_category": self.get_aov_by_category(),
            "assessment": self._engine.assess(),
            "heatmap": self._engine.get_heatmap_data(
                "conversion",
                x_dimension="traffic_source",
                y_dimension="device",
            ),
        }

    # --- Helpers ---

    def _get_value_bucket(self, value: float) -> str:
        """Categorize cart value into buckets."""
        if value < 50:
            return "under_50"
        elif value < 100:
            return "50_to_100"
        elif value < 200:
            return "100_to_200"
        else:
            return "over_200"


def main():
    # Create the e-commerce metrics tracker
    metrics = EcommerceMetrics()

    print("=== Recording Sample Data ===")

    # Simulate visits
    visits = [
        (True, "google_ads", "mobile", "US"),
        (False, "google_ads", "mobile", "US"),
        (False, "google_ads", "desktop", "US"),
        (True, "organic", "desktop", "US"),
        (True, "organic", "desktop", "UK"),
        (False, "facebook", "mobile", "US"),
        (False, "facebook", "mobile", "US"),
        (True, "email", "desktop", "US"),
        (False, "direct", "mobile", "US"),
        (True, "organic", "mobile", "UK"),
    ]

    for converted, source, device, country in visits:
        metrics.record_visit(converted, source, device, country)

    # Simulate orders
    orders = [
        (89.99, "google_ads", "mobile", "electronics"),
        (156.50, "organic", "desktop", "clothing"),
        (42.00, "organic", "desktop", "books"),
        (220.00, "email", "desktop", "electronics"),
        (78.50, "organic", "mobile", "clothing"),
    ]

    for value, source, device, category in orders:
        metrics.record_order(value, source, device, category)

    # Simulate cart events
    cart_events = [
        (True, 75.00, "google_ads", "mobile"),   # Abandoned
        (False, 89.99, "google_ads", "mobile"),  # Completed
        (True, 120.00, "facebook", "mobile"),    # Abandoned
        (True, 45.00, "facebook", "mobile"),     # Abandoned
        (False, 156.50, "organic", "desktop"),   # Completed
    ]

    for abandoned, value, source, device in cart_events:
        metrics.record_cart_event(abandoned, value, source, device)

    # Simulate page loads
    page_loads = [
        (1.2, "home", "desktop"),
        (2.5, "home", "mobile"),
        (0.8, "product", "desktop"),
        (1.8, "product", "mobile"),
        (0.5, "checkout", "desktop"),
        (1.5, "checkout", "mobile"),
    ]

    for load_time, page_type, device in page_loads:
        metrics.record_page_load(load_time, page_type, device)

    # Get dashboard
    print("\n=== Dashboard ===")
    dashboard = metrics.get_dashboard()

    print(f"\nSummary:")
    print(f"  Conversion Rate: {dashboard['summary']['conversion_rate']:.1%}")
    print(f"  Abandonment Rate: {dashboard['summary']['abandonment_rate']:.1%}")
    print(f"  Avg Order Value: ${dashboard['summary']['avg_order_value']:.2f}")
    print(f"  Avg Page Load: {dashboard['summary']['avg_page_load']:.2f}s")

    print(f"\nConversion by Source:")
    for source, rate in dashboard['by_source'].items():
        print(f"  {source}: {rate:.1%}")

    print(f"\nConversion by Device:")
    for device, rate in dashboard['by_device'].items():
        print(f"  {device}: {rate:.1%}")

    print(f"\nAOV by Category:")
    for category, aov in dashboard['by_category'].items():
        print(f"  {category}: ${aov:.2f}")

    print(f"\nAssessment:")
    for metric, data in dashboard['assessment']['metrics'].items():
        print(f"  {metric}: {data['level']}")


if __name__ == "__main__":
    main()
