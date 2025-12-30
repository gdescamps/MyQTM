import json

import src.config as config


def _get_param_names(param_names):
    return param_names or config.CMA_PARAM_NAMES


def _get_default_values(default_values):
    return default_values or config.INIT_X0


def save_cma_params(path, params, param_names=None):
    param_names = _get_param_names(param_names)
    if isinstance(params, dict):
        payload = {"params": params}
    else:
        payload = {"params": dict(zip(param_names, params))}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_cma_params(path, param_names=None, default_values=None):
    param_names = _get_param_names(param_names)
    default_values = _get_default_values(default_values)
    with open(path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload

    params = payload.get("params") if isinstance(payload, dict) else None
    if params is None and isinstance(payload, dict):
        params = payload

    if not isinstance(params, dict):
        return default_values

    values = []
    for name, default in zip(param_names, default_values):
        values.append(params.get(name, default))
    return values
