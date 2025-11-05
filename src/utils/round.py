def round_floats(obj, precision=2):
    """Recursively rounds all floats in a dictionary or list."""
    if isinstance(obj, dict):
        return {k: round_floats(v, precision) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(i, precision) for i in obj]
    elif isinstance(obj, float):
        return round(obj, precision)
    else:
        return obj
