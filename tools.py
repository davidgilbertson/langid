from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import json
import gzip
import numpy as np
from sklearn.linear_model import LogisticRegression

JSON_DECIMALS_UNSET = object()


@dataclass
class stopwatch(AbstractContextManager):
    name: str
    start: int = field(default_factory=perf_counter, init=False)
    log: bool = True

    def get_time_ms(self):
        return (perf_counter() - self.start) * 1000

    def get_print_string(self):
        duration = timedelta(milliseconds=self.get_time_ms())

        return f"⏱ {duration} ⮜ {self.name}"

    def done(self):
        if self.log:
            print(self.get_print_string())

    def __enter__(self):
        if self.log:
            print(f"⏱ {self.name}...", end="\r")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.done()
        return False


def compact_value(value: float) -> int | float:
    if value == 0:
        return 0
    if float(value).is_integer():
        return int(value)
    return float(value)


def smart_round(values: np.ndarray, decimals: int) -> list:
    """
    Round values, and coerce to an int if possible without loss.
    """
    rounded = np.round(values, decimals)
    if rounded.ndim == 1:
        return [compact_value(value) for value in rounded]
    return [[compact_value(value) for value in row] for row in rounded]


def model_to_dict(
    model,
    X,
    json_decimals: int | None | object = JSON_DECIMALS_UNSET,
) -> dict:
    # Sort features by importance and align coefficients accordingly.
    importance = np.mean(np.abs(model.coef_), axis=0)
    feature_order = np.argsort(-importance)
    ordered_features = [X.columns[i] for i in feature_order]
    ordered_coef = model.coef_[:, feature_order]

    if json_decimals is JSON_DECIMALS_UNSET:
        json_decimals = 1

    if json_decimals is None:
        coef = ordered_coef.tolist()
        intercept = model.intercept_.tolist()
    else:
        coef = smart_round(ordered_coef, json_decimals)
        intercept = smart_round(model.intercept_, json_decimals)

    return {
        "features": ordered_features,
        "classes": model.classes_.tolist(),
        "coef": coef,
        "intercept": intercept,
    }


def trim_model_features(model: dict, n_features: int) -> dict:
    if not n_features:
        return model
    trimmed = dict(model)
    trimmed["features"] = model["features"][:n_features]
    trimmed["coef"] = [row[:n_features] for row in model["coef"]]
    return trimmed


def get_gzipped_size_kb(
    model: Path | str | dict | LogisticRegression,
    X=None,
    json_decimals: int | None | object = JSON_DECIMALS_UNSET,
    n_features: int | None = None,
) -> float:
    if isinstance(model, LogisticRegression):
        if X is None:
            raise ValueError("X is required when passing a LogisticRegression model.")
        model_dict = model_to_dict(model, X, json_decimals)
        if n_features is not None:
            model_dict = trim_model_features(model_dict, n_features)
        payload = json.dumps(model_dict)
    elif isinstance(model, dict):
        model_dict = trim_model_features(model, n_features) if n_features else model
        payload = json.dumps(model_dict)
    else:
        path = Path(model)
        if n_features is None:
            payload = path.read_text(encoding="utf-8")
        else:
            model_dict = json.loads(path.read_text(encoding="utf-8"))
            model_dict = trim_model_features(model_dict, n_features)
            payload = json.dumps(model_dict)

    return len(gzip.compress(payload.encode("utf-8"))) / 1024
