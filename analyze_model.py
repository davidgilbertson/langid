from functools import cache
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average="macro")


class ModelMeta(NamedTuple):
    model: LogisticRegression
    languages: list[str]
    tokens: list[str]
    coef: np.ndarray
    bias: np.ndarray


@cache
def load_model(path: Path = None) -> ModelMeta:
    path = path or Path("model.json")
    model = json.loads(path.read_text())
    languages = model["classes"]
    tokens = model["features"]
    coef = np.array(model["coef"])
    bias = np.array(model["intercept"])

    assert len(coef) == len(bias) == len(languages)
    assert coef.shape[1] == len(tokens)

    return ModelMeta(
        model=model,
        languages=languages,
        tokens=tokens,
        coef=coef,
        bias=bias,
    )


def generate_features(snippets: list[str], tokens: list[str]) -> np.ndarray:
    features_matrix = []
    for snippet in snippets:
        snippet = snippet.replace("\r\n", "\n").replace("\r", "\n")
        features_matrix.append([token in snippet for token in tokens])

    features_matrix = np.array(features_matrix, dtype=bool)
    return features_matrix


def generate_f1_curve(
    df: pd.DataFrame,
    model_path: Path = None,
    number_of_n=100,
) -> pd.DataFrame:
    model, languages, tokens, coef, bias = load_model(path=model_path)
    features_matrix = generate_features(df.Snippet.to_list(), tokens=tokens)
    labels = df.Language.to_numpy()

    total_tokens = coef.shape[1]
    if number_of_n <= 1:
        ns = [total_tokens]
    else:
        ns = np.rint(np.linspace(1, total_tokens, num=number_of_n)).astype(int).tolist()
        ns[0] = 1
        ns[-1] = total_tokens

    results = []

    for n in ns:
        n_coef = coef[:, :n]
        scores = features_matrix[:, :n] @ n_coef.T + bias
        preds = np.take(languages, np.argmax(scores, axis=1))
        f1 = score(labels, preds)
        results.append(dict(N=n, F1=f1))
        print(f"F1 @ {n=}: {f1:.2%}", end="\r")

    return pd.DataFrame(results)


class IdealSize(NamedTuple):
    n: int
    f1: float


def find_ideal_size(
    df: pd.DataFrame,
    model_path: Path = None,
    f1_delta=0.01,
) -> IdealSize:
    """
    Does a backward scan to keep dropping features until F1 falls by `f1_delta`
    """

    model, languages, tokens, coef, bias = load_model(path=model_path)
    features_matrix = generate_features(df.Snippet.to_list(), tokens=tokens)
    labels = df.Language.to_numpy()
    total_tokens = coef.shape[1]

    scores_full = features_matrix @ coef.T + bias
    preds_full = np.take(languages, np.argmax(scores_full, axis=1))
    f1_full = score(labels, preds_full)
    f1_target = f1_full - f1_delta
    print(f"Scanning for n features where F1 >= {f1_target:.1%})")

    best_n = total_tokens
    f1 = 0
    for n in range(total_tokens, 0, -1):
        n_coef = coef[:, :n]
        scores = features_matrix[:, :n] @ n_coef.T + bias
        preds = np.take(languages, np.argmax(scores, axis=1))
        f1 = score(labels, preds)
        print(f"F1 @ {n=}: {f1:.2%}", end="\r")
        if f1 >= f1_target:
            best_n = n
        else:
            break

    return IdealSize(
        n=best_n,
        f1=f1,
    )


if __name__ == "__main__":
    # model, languages, tokens, coef, bias = load_model()
    from data.utils import get_stack_data

    # df = get_stack_data(snippet_limit=10, subset=0.1)
    # labels = df.Language.to_numpy()
    # features_matrix = generate_features(df.Snippet.to_list(), tokens=tokens)
    # total_tokens = len(tokens)

    df = get_stack_data(snippet_limit=10, subset=0.1)
    # f1_df = generate_f1_curve(df=df)

    ideal_n, ideal_f1 = find_ideal_size(df)

    # # %% - basic inference
    # scores = features_matrix @ coef.T + bias
    # preds = np.take(languages, np.argmax(scores, axis=1))
    # f1 = score(labels, preds)
    # print(f"F1: {f1:.2%}")
