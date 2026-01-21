from contextlib import AbstractContextManager
from copy import copy
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import json
import gzip
import numpy as np
from sklearn.linear_model import LogisticRegression

# The number of decimals used when saving the model to JSON
ROUNDING_DECIMALS: int | None = 1


def set_rounding(decimals: int | None) -> None:
    global ROUNDING_DECIMALS
    ROUNDING_DECIMALS = decimals


def get_rounding() -> int | None:
    return ROUNDING_DECIMALS


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


def round_model(model: LogisticRegression) -> LogisticRegression:
    decimals = get_rounding()
    if decimals is None:
        return model

    rounded_model = copy(model)
    rounded_model.coef_ = np.array(smart_round(model.coef_, decimals), dtype=float)
    rounded_model.intercept_ = np.array(
        smart_round(model.intercept_, decimals),
        dtype=float,
    )

    return rounded_model


def model_to_dict(model: LogisticRegression) -> dict:
    return {
        "features": model.feature_names_in_.tolist(),
        "classes": model.classes_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
    }


def get_gzipped_size_kb(
    model: LogisticRegression,
    n_features: int | None = None,
) -> float:
    model_dict = model_to_dict(round_model(model))
    if n_features is not None:
        model_dict["features"] = model_dict["features"][:n_features]
        model_dict["coef"] = [row[:n_features] for row in model_dict["coef"]]
    payload = json.dumps(model_dict, separators=(",", ":"))

    return len(gzip.compress(payload.encode("utf-8"))) / 1024


def save_for_web(
    model: LogisticRegression,
    n_features: int | None = None,
) -> None:
    model_dict = model_to_dict(round_model(model))
    if n_features is not None:
        model_dict["features"] = model_dict["features"][:n_features]
        model_dict["coef"] = [row[:n_features] for row in model_dict["coef"]]

    payload = json.dumps(model_dict, separators=(",", ":"))
    model_path = Path("web/public/model.json")
    model_path.write_text(payload, encoding="utf-8")
