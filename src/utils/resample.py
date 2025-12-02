"""
resample.py

This module provides utility functions for data manipulation and processing.

Functions:
- round_floats(): Recursively rounds all floating-point numbers in a dictionary or list to a specified precision.

"""


def round_floats(obj, precision=2):
    """
    Recursively rounds all floating-point numbers in a dictionary or list.

    Args:
        obj (dict or list): The input object containing floats to round.
        precision (int, optional): The number of decimal places to round to. Defaults to 2.

    Returns:
        dict or list: The input object with all floats rounded to the specified precision.
    """
    if isinstance(obj, dict):
        return {k: round_floats(v, precision) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(i, precision) for i in obj]
    elif isinstance(obj, float):
        return round(obj, precision)
    else:
        return obj
