"""Module for managing time interval calculations for train/test data partitioning.

This module provides utilities to determine which training or testing interval
a given date falls into, supporting the multi-part training/testing split strategy
used in the quantitative trading model.
"""

from datetime import datetime

import src.config as config

# Global variables to cache parsed date boundaries, avoiding repeated string-to-datetime conversions.
train_start = None
train_start_str = None
train_end = None
train_end_str = None
bench_end = None
bench_end_str = None


def get_interval_type(
    query_date: str | datetime, interval_days: int = 90, end_limit: bool = True
) -> str | None:
    """Determine which training or testing interval a given date falls into.

    Maps dates to interval labels (part1A, part1B, part2A, part2B, part3A, part3B)
    based on the time elapsed from TRAIN_START_DATE. Labels are modified to C/D
    variants during the test period (TRAIN_END_DATE to TEST_END_DATE).

    Args:
        query_date: Target date as string ("YYYY-MM-DD") or datetime object.
        interval_days: Length of each interval in days (default: 90).
        end_limit: If True, returns None for dates beyond TEST_END_DATE.
                   If False, allows dates beyond TEST_END_DATE (used by robot trading).

    Returns:
        Interval label string (e.g., "part1A", "part2C"), or None if date is
        outside valid range or before TRAIN_START_DATE.
    """

    global train_start
    global train_end
    global bench_end
    global train_start_str
    global train_end_str
    global bench_end_str

    # Initialize or refresh cached date boundaries when configuration changes.
    if train_start is None or train_start_str != config.TRAIN_START_DATE:
        train_start = datetime.strptime(config.TRAIN_START_DATE, "%Y-%m-%d")
        train_start_str = config.TRAIN_START_DATE

    if train_end is None or train_end_str != config.TRAIN_END_DATE:
        train_end = datetime.strptime(config.TRAIN_END_DATE, "%Y-%m-%d")
        train_end_str = config.TRAIN_END_DATE

    if bench_end is None or bench_end_str != config.BENCHMARK_END_DATE:
        bench_end = datetime.strptime(config.BENCHMARK_END_DATE, "%Y-%m-%d")
        bench_end_str = config.BENCHMARK_END_DATE

    # Define the repeating cycle of interval labels for the training phase.
    interval_types = ["part1A", "part1B", "part2A", "part2B", "part3A", "part3B"]

    # Convert string dates to datetime objects for comparison.
    if isinstance(query_date, str):
        query_date = datetime.strptime(query_date, "%Y-%m-%d")

    # Reject dates before training period starts.
    if query_date < train_start:
        return None

    # Enforce upper bound on dates when end_limit is enabled.
    if end_limit and query_date > bench_end:
        return None

    # When end_limit is False, dates beyond TEST_END_DATE are allowed for live robot trading.

    # Calculate which interval the query_date falls into based on elapsed days.
    days_diff = (query_date - train_start).days
    interval_index = days_diff // interval_days
    interval = interval_types[interval_index % len(interval_types)]

    # During the test period, modify interval labels: A→C, B→D for test data identification.
    if query_date >= train_end and query_date <= bench_end:
        if "A" in interval:
            return interval.replace("A", "C")
        elif "B" in interval:
            return interval.replace("B", "D")

    return interval
