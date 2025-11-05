from datetime import datetime

import src.config as config


# Define a function to assign query dates to intervals (full, partial, test) based on training date ranges.
def get_interval_type(
    query_date_str: str, interval_days: int = 90, end_limit: bool = True
) -> str | None:
    """
    Assigns the query date to one of the intervals: 'full', 'partial', 'test'.
    Each interval lasts 3 months, chained starting from train_start_date.
    Returns None if the date is out of range.
    """
    interval_types = ["part1A", "part1B", "part2A", "part2B", "part3A", "part3B"]

    train_start = datetime.strptime(config.TRAIN_START_DATE, "%Y-%m-%d")
    test_end = datetime.strptime(config.TEST_END_DATE, "%Y-%m-%d")
    query_date = datetime.strptime(query_date_str, "%Y-%m-%d")

    if query_date < train_start:
        return None

    if end_limit and query_date > test_end:
        return None

    # Number of months between train_start and query_date
    days_diff = (query_date - train_start).days
    interval_index = days_diff // interval_days
    interval = interval_types[interval_index % len(interval_types)]

    if query_date >= datetime.strptime(
        config.TRAIN_END_DATE, "%Y-%m-%d"
    ) and query_date <= datetime.strptime(config.TEST_END_DATE, "%Y-%m-%d"):
        if "A" in interval:
            return interval.replace("A", "C")
        elif "B" in interval:
            return interval.replace("B", "D")

    return interval
