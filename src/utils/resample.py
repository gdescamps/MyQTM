def round_floats(obj, precision=2):
    """Arrondit récursivement tous les floatants dans un dictionnaire ou une liste."""
    if isinstance(obj, dict):
        return {k: round_floats(v, precision) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(i, precision) for i in obj]
    elif isinstance(obj, float):
        return round(obj, precision)
    else:
        return obj
