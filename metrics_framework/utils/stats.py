"""Statistical utility functions."""

from typing import List, Optional
import math


def calculate_mean(values: List[float]) -> float:
    """Calculate arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_median(values: List[float]) -> float:
    """Calculate median."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2
    return sorted_values[mid]


def calculate_std(values: List[float], sample: bool = True) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = calculate_mean(values)
    variance = sum((x - mean) ** 2 for x in values)
    divisor = len(values) - 1 if sample else len(values)
    return math.sqrt(variance / divisor)


def calculate_variance(values: List[float], sample: bool = True) -> float:
    """Calculate variance."""
    if len(values) < 2:
        return 0.0
    mean = calculate_mean(values)
    variance = sum((x - mean) ** 2 for x in values)
    divisor = len(values) - 1 if sample else len(values)
    return variance / divisor


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate percentile (0-100)."""
    if not values:
        return 0.0
    if percentile < 0 or percentile > 100:
        raise ValueError("Percentile must be between 0 and 100")

    sorted_values = sorted(values)
    n = len(sorted_values)

    if percentile == 0:
        return sorted_values[0]
    if percentile == 100:
        return sorted_values[-1]

    k = (n - 1) * (percentile / 100)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return sorted_values[int(k)]

    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return d0 + d1


def calculate_mode(values: List[float]) -> Optional[float]:
    """Calculate mode (most frequent value)."""
    if not values:
        return None
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    max_count = max(counts.values())
    modes = [v for v, c in counts.items() if c == max_count]
    return modes[0] if len(modes) == 1 else None


def calculate_rate(values: List[float], time_delta_seconds: float) -> float:
    """Calculate rate (sum per time unit)."""
    if not values or time_delta_seconds <= 0:
        return 0.0
    return sum(values) / time_delta_seconds


def calculate_success_rate(
    correct: int,
    total: int,
    as_percentage: bool = False
) -> float:
    """Calculate success rate."""
    if total == 0:
        return 0.0
    rate = correct / total
    return rate * 100 if as_percentage else rate


def calculate_weighted_mean(
    values: List[float],
    weights: List[float]
) -> float:
    """Calculate weighted mean."""
    if not values or not weights or len(values) != len(weights):
        return 0.0
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def calculate_moving_average(
    values: List[float],
    window: int = 5
) -> List[float]:
    """Calculate moving average."""
    if not values or window <= 0:
        return []
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_values = values[start:i + 1]
        result.append(calculate_mean(window_values))
    return result


def calculate_exponential_moving_average(
    values: List[float],
    alpha: float = 0.3
) -> List[float]:
    """Calculate exponential moving average."""
    if not values:
        return []
    result = [values[0]]
    for i in range(1, len(values)):
        ema = alpha * values[i] + (1 - alpha) * result[-1]
        result.append(ema)
    return result


def detect_outliers(
    values: List[float],
    method: str = "iqr",
    threshold: float = 1.5
) -> List[int]:
    """Detect outlier indices using IQR or Z-score method."""
    if len(values) < 4:
        return []

    outliers = []

    if method == "iqr":
        q1 = calculate_percentile(values, 25)
        q3 = calculate_percentile(values, 75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        for i, v in enumerate(values):
            if v < lower or v > upper:
                outliers.append(i)
    elif method == "zscore":
        mean = calculate_mean(values)
        std = calculate_std(values)
        if std > 0:
            for i, v in enumerate(values):
                z = abs((v - mean) / std)
                if z > threshold:
                    outliers.append(i)

    return outliers
