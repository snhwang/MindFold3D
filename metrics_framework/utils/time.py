"""Time utility functions."""

from datetime import datetime, timedelta
from typing import Optional, Tuple


def get_time_bucket(
    timestamp: datetime,
    interval: str = "hour"
) -> datetime:
    """
    Get the time bucket for a timestamp.

    Args:
        timestamp: The timestamp to bucket
        interval: The interval (minute, hour, day, week, month)

    Returns:
        The start of the time bucket
    """
    if interval == "minute":
        return timestamp.replace(second=0, microsecond=0)
    elif interval == "hour":
        return timestamp.replace(minute=0, second=0, microsecond=0)
    elif interval == "day":
        return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    elif interval == "week":
        # Start of week (Monday)
        days_since_monday = timestamp.weekday()
        start_of_week = timestamp - timedelta(days=days_since_monday)
        return start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
    elif interval == "month":
        return timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif interval == "year":
        return timestamp.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError(f"Unknown interval: {interval}")


def parse_interval(interval: str) -> timedelta:
    """
    Parse an interval string to timedelta.

    Args:
        interval: Interval string (e.g., "1h", "30m", "7d")

    Returns:
        timedelta object
    """
    if not interval:
        raise ValueError("Empty interval")

    unit = interval[-1].lower()
    try:
        value = int(interval[:-1]) if len(interval) > 1 else 1
    except ValueError:
        # Try parsing named intervals
        named = {
            "minute": timedelta(minutes=1),
            "hour": timedelta(hours=1),
            "day": timedelta(days=1),
            "week": timedelta(weeks=1),
            "month": timedelta(days=30),
            "year": timedelta(days=365),
        }
        if interval.lower() in named:
            return named[interval.lower()]
        raise ValueError(f"Invalid interval: {interval}")

    units = {
        "s": timedelta(seconds=1),
        "m": timedelta(minutes=1),
        "h": timedelta(hours=1),
        "d": timedelta(days=1),
        "w": timedelta(weeks=1),
    }

    if unit not in units:
        raise ValueError(f"Unknown interval unit: {unit}")

    return units[unit] * value


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable string (e.g., "2h 30m", "45s")
    """
    if seconds < 0:
        return "0s"

    if seconds < 1:
        return f"{int(seconds * 1000)}ms"

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    if minutes < 60:
        if remaining_seconds > 0:
            return f"{minutes}m {remaining_seconds}s"
        return f"{minutes}m"

    hours = int(minutes // 60)
    remaining_minutes = minutes % 60

    if hours < 24:
        if remaining_minutes > 0:
            return f"{hours}h {remaining_minutes}m"
        return f"{hours}h"

    days = int(hours // 24)
    remaining_hours = hours % 24

    if remaining_hours > 0:
        return f"{days}d {remaining_hours}h"
    return f"{days}d"


def get_time_range(
    period: str,
    end_time: Optional[datetime] = None
) -> Tuple[datetime, datetime]:
    """
    Get time range for a named period.

    Args:
        period: Named period (today, yesterday, this_week, last_week, etc.)
        end_time: End time (defaults to now)

    Returns:
        Tuple of (start_time, end_time)
    """
    end = end_time or datetime.utcnow()

    if period == "today":
        start = end.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "yesterday":
        start = (end - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = end.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "this_week":
        days_since_monday = end.weekday()
        start = (end - timedelta(days=days_since_monday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    elif period == "last_week":
        days_since_monday = end.weekday()
        end = (end - timedelta(days=days_since_monday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        start = end - timedelta(days=7)
    elif period == "this_month":
        start = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif period == "last_month":
        first_of_this_month = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = first_of_this_month
        last_month = first_of_this_month - timedelta(days=1)
        start = last_month.replace(day=1)
    elif period == "last_7_days":
        start = end - timedelta(days=7)
    elif period == "last_30_days":
        start = end - timedelta(days=30)
    elif period == "last_90_days":
        start = end - timedelta(days=90)
    else:
        raise ValueError(f"Unknown period: {period}")

    return start, end


def datetime_to_iso(dt: datetime) -> str:
    """Convert datetime to ISO format string."""
    return dt.isoformat()


def iso_to_datetime(iso_str: str) -> datetime:
    """Convert ISO format string to datetime."""
    return datetime.fromisoformat(iso_str)
