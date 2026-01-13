import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from tools import stopwatch


def save_model(model, features: list, rounding=2) -> Path:
    # When saving the model, put the features in order of importance.
    # We save the ordered features and also rearrange the weights
    # You can then shrink the model just by truncating weights and features to top-n
    importance = np.mean(np.abs(model.coef_), axis=0)
    feature_order = np.argsort(-importance)
    ordered_features = [features[i] for i in feature_order]
    ordered_coef = model.coef_[:, feature_order]

    payload = {
        "features": ordered_features,
        "classes": model.classes_.tolist(),
        "coef": np.round(ordered_coef, rounding).tolist(),
        "intercept": np.round(model.intercept_, rounding).tolist(),
    }

    path = Path(__file__).parent / "model.json"
    path.write_text(json.dumps(payload, indent=2))

    # imp_df = (
    #     pd.DataFrame(dict(Feature=features, Importance=importance))
    #     .sort_values("Importance", ascending=False)
    #     .reset_index(drop=True)
    # )

    return path


# We go crazy with the return value for easier exploration
class TrainResult(NamedTuple):
    f1: float
    model: LogisticRegression
    model_path: Path | None
    features: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    X_trn: pd.DataFrame
    X_val: pd.DataFrame
    y_trn: pd.Series
    y_val: pd.Series
    preds: np.ndarray
    probs: np.ndarray


def train_model(
    features: pd.DataFrame | None = None,
    save=True,
    frac: float = 1.0,
) -> TrainResult:
    df = pd.read_parquet("features/features.parquet") if features is None else features

    if 0 < frac < 1:
        # Subset of rows
        df = df.groupby("Target", group_keys=False).sample(frac=frac)

    X = df.drop(columns=["Target"])
    y = df["Target"]

    X_trn, X_val, y_trn, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    with stopwatch(
        f"Training on {len(df):,} rows, {len(X.columns)} features, and {y.nunique()} languages"
    ):
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )

        model.fit(X_trn, y_trn)

        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)

        # Accuracy is misleading for imbalanced classes (per-snippet counts skew).
        f1 = f1_score(y_val, preds, average="macro")
    print(f"F1 (macro): {f1:.1%}")

    model_path = None
    if save:
        model_path = save_model(model, features=X.columns.to_list())

    return TrainResult(
        f1=f1,
        model=model,
        model_path=model_path,
        features=df,
        X=X,
        y=y,
        X_trn=X_trn,
        X_val=X_val,
        y_trn=y_trn,
        y_val=y_val,
        preds=preds,
        probs=probs,
    )


if __name__ == "__main__":
    result = train_model(frac=0.1, save=False)

    # # %% - Inspect the wrong answers
    # from data.utils import get_stack_data
    # model, features, X, y, X_trn, X_val, y_trn, y_val, preds, probs = result
    # result_df = y_val.to_frame()
    # result_df["Pred"] = preds
    # result_df["Confidence"] = np.max(probs, axis=1)
    # # result_df["Snippet"] = snippets
    # wrong_df = (
    #     result_df[y_val.ne(preds)].copy().sort_values("Confidence", ascending=False)
    # )
    #
    # snippets = get_stack_data(snippet_limit=10).Snippet
    # # Below is dodgy, but snippets and features came from the same place so share an index.
    # wrong_df["Snippet"] = snippets.loc[wrong_df.index]
    #
    # wrong_counts_df = (
    #     wrong_df.groupby(["Target", "Pred"])
    #     .size()
    #     .reset_index(name="Errors")
    #     .sort_values("Errors", ascending=False)
    # )

    # print("Most common error types")
    # print(wrong_counts_df.head(10))
