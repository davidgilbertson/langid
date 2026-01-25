from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from tools import (
    get_gzipped_size_kb,
    get_rounding,
    round_model,
    set_rounding,
)


def score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return f1_score(y_true, y_pred, average="macro")


def score_model(
    model: LogisticRegression,
    X: pd.DataFrame,
    y: pd.Series,
    coef: ArrayLike | None = None,
    bias: ArrayLike | None = None,
    languages: ArrayLike | None = None,
) -> float:
    coef = coef if coef is not None else model.coef_
    bias = bias if bias is not None else model.intercept_
    languages = languages if languages is not None else model.classes_
    scores = X @ coef.T + bias
    preds = np.take(languages, np.argmax(scores, axis=1))
    return score(y, preds)


def generate_f1_curve(
    model: LogisticRegression,
    X: pd.DataFrame,
    y: pd.Series,
    steps=50,
) -> pd.DataFrame:
    """Compute F1 as the number of top features increases."""
    assert steps >= 2

    rounded_model = round_model(model)
    coef = rounded_model.coef_
    bias = rounded_model.intercept_
    languages = rounded_model.classes_

    total_features = coef.shape[1]
    ns = np.rint(np.linspace(1, total_features, num=steps)).astype(int).tolist()
    ns[0] = 1
    ns[-1] = total_features

    results = []

    for n in ns:
        n_coef = coef[:, :n]
        f1 = score_model(
            rounded_model,
            X.iloc[:, :n],
            y,
            coef=n_coef,
            bias=bias,
            languages=languages,
        )
        size_kb = get_gzipped_size_kb(rounded_model, n_features=n)
        results.append(dict(N=n, F1=f1, SizeKb=size_kb))
        print(f"F1 @ {n=}: {f1:.2%}", end="\r")

    return pd.DataFrame(results)


def generate_rounding_curve(
    model: LogisticRegression,
    X: pd.DataFrame,
    y: pd.Series,
    max_decimals: int = 6,
) -> pd.DataFrame:
    """Compute F1 after rounding weights to different decimal places."""

    languages = model.classes_

    results = []
    original_rounding = get_rounding()
    for places in list(range(max_decimals + 1)) + [None]:
        set_rounding(places)
        rounded_model = round_model(model)
        coef = rounded_model.coef_
        bias = rounded_model.intercept_
        f1 = score_model(
            rounded_model,
            X,
            y,
            coef=coef,
            bias=bias,
            languages=languages,
        )

        size_kb = get_gzipped_size_kb(rounded_model)
        results.append(dict(Decimals=places, F1=f1, SizeKb=size_kb))
        print(f"F1 @ rounding={places}: {f1:.2%}", end="\r")

    set_rounding(original_rounding)

    return pd.DataFrame(results)


class IdealSize(NamedTuple):
    n: int
    f1: float


def find_ideal_size(
    model: LogisticRegression,
    X: pd.DataFrame,
    y: pd.Series,
    f1_delta=0.01,
) -> IdealSize:
    """
    Scan backward to keep dropping features until F1 falls by `f1_delta`.
    """

    rounded_model = round_model(model)
    coef = rounded_model.coef_
    bias = rounded_model.intercept_
    languages = rounded_model.classes_
    total_features = coef.shape[1]

    f1_full = score_model(
        rounded_model,
        X,
        y,
        coef=coef,
        bias=bias,
        languages=languages,
    )
    f1_target = f1_full - f1_delta

    if f1_delta == 0:
        return IdealSize(n=total_features, f1=f1_full)

    print(f"Scanning for n features where F1 >= {f1_target:.1%}")

    best_n = total_features
    f1 = f1_full
    for n in range(total_features, 0, -1):
        # The right-most columns are often all zeros, skip them.
        if np.all(coef[:, n - 1] == 0):
            if f1 >= f1_target:
                best_n = n
                continue
            break
        n_coef = coef[:, :n]
        f1 = score_model(
            rounded_model,
            X.iloc[:, :n],
            y,
            coef=n_coef,
            bias=bias,
            languages=languages,
        )
        print(f"F1 @ {n=}: {f1:.1%}", end="\r")
        if f1 >= f1_target:
            best_n = n
        else:
            break

    return IdealSize(
        n=best_n,
        f1=f1,
    )


if __name__ == "__main__":
    from train_model import train_model

    df = pd.read_parquet("E:/Datasets/the_stack_whole_files.parquet")
    results = train_model(df)

    f1_df = generate_f1_curve(
        model=results.model,
        X=results.X_val,
        y=results.y_val,
        steps=10,
    )
