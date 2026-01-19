from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from tools import get_gzipped_size_kb, model_to_dict, smart_round


def score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return f1_score(y_true, y_pred, average="macro")


def generate_f1_curve(
    model: LogisticRegression,
    X: pd.DataFrame,
    y: pd.Series,
    steps=50,
) -> pd.DataFrame:
    """Compute F1 as the number of top features increases."""
    assert steps >= 2

    coef = model.coef_
    bias = model.intercept_
    languages = model.classes_

    total_features = coef.shape[1]
    ns = np.rint(np.linspace(1, total_features, num=steps)).astype(int).tolist()
    ns[0] = 1
    ns[-1] = total_features

    results = []

    for n in ns:
        n_coef = coef[:, :n]
        scores = X.iloc[:, :n] @ n_coef.T + bias
        preds = np.take(languages, np.argmax(scores, axis=1))
        f1 = score(y, preds)
        results.append(dict(N=n, F1=f1))
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
    for places in range(max_decimals + 1):
        coef = np.array(smart_round(model.coef_, places), dtype=float)
        bias = np.array(smart_round(model.intercept_, places), dtype=float)
        scores = X @ coef.T + bias
        preds = np.take(languages, np.argmax(scores, axis=1))
        f1 = score(y, preds)
        model_dict = model_to_dict(
            model=model,
            json_decimals=places,
        )
        size_kb = get_gzipped_size_kb(model_dict)
        results.append(dict(Decimals=places, F1=f1, SizeKb=size_kb))
        print(f"F1 @ rounding={places}: {f1:.2%}", end="\r")

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

    coef = model.coef_
    bias = model.intercept_
    languages = model.classes_
    total_features = coef.shape[1]

    scores_full = X @ coef.T + bias
    preds_full = np.take(languages, np.argmax(scores_full, axis=1))
    f1_full = score(y, preds_full)
    f1_target = f1_full - f1_delta

    if f1_delta == 0:
        return IdealSize(n=total_features, f1=f1_full)

    print(f"Scanning for n features where F1 >= {f1_target:.1%}")

    best_n = total_features
    f1 = 0
    for n in range(total_features, 0, -1):
        n_coef = coef[:, :n]
        scores = X.iloc[:, :n] @ n_coef.T + bias
        preds = np.take(languages, np.argmax(scores, axis=1))
        f1 = score(y, preds)
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

    # rounding_df = generate_rounding_curve(
    #     model=results.model,
    #     X=results.X_val,
    #     y=results.y_val,
    # )

    # ideal_n, ideal_f1 = find_ideal_size(
    #     model=results.model,
    #     X=results.X_val,
    #     y=results.y_val,
    # )

    # # %% - basic inference
    # scores = features_matrix @ coef.T + bias
    # preds = np.take(languages, np.argmax(scores, axis=1))
    # f1 = score(labels, preds)
    # print(f"F1: {f1:.2%}")
