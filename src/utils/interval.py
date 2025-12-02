from datetime import datetime

import src.config as config

train_start = None
train_start_str = None
train_end = None
train_end_str = None
test_end = None
test_end_str = None


def get_interval_type(
    query_date: str | datetime, interval_days: int = 90, end_limit: bool = True
) -> str | None:

    global train_start
    global train_end
    global test_end
    global train_start_str
    global train_end_str
    global test_end_str

    if train_start is None or train_start_str != config.TRAIN_START_DATE:
        # store results as global to avoid converting at each call
        train_start = datetime.strptime(config.TRAIN_START_DATE, "%Y-%m-%d")
        train_start_str = config.TRAIN_START_DATE

    if train_end is None or train_end_str != config.TRAIN_END_DATE:
        # store results as global to avoid converting at each call
        train_end = datetime.strptime(config.TRAIN_END_DATE, "%Y-%m-%d")
        train_end_str = config.TRAIN_END_DATE

    if test_end is None or test_end_str != config.TEST_END_DATE:
        # store results as global to avoid converting at each call
        test_end = datetime.strptime(config.TEST_END_DATE, "%Y-%m-%d")
        test_end_str = config.TEST_END_DATE

    interval_types = ["part1A", "part1B", "part2A", "part2B", "part3A", "part3B"]

    if isinstance(query_date, str):
        query_date = datetime.strptime(query_date, "%Y-%m-%d")

    if query_date < train_start:
        return None

    if end_limit and query_date > test_end:
        return None  # stop after TEST_END_DATE if end_limit is set

    # if end_limit is False, allow dates beyond TEST_END_DATE for robot.

    days_diff = (query_date - train_start).days
    interval_index = days_diff // interval_days
    interval = interval_types[interval_index % len(interval_types)]

    if query_date >= train_end and query_date <= test_end:
        if "A" in interval:
            return interval.replace("A", "C")
        elif "B" in interval:
            return interval.replace("B", "D")

    return interval
