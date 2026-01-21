import json
import hashlib
from pathlib import Path
import pickle
from typing import NamedTuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from features import generate_features
from tools import model_to_dict, round_model, stopwatch


def save_model_json(
    model,
    y: pd.Series,
) -> Path:
    model_dict = model_to_dict(round_model(model))

    # We give this a human-readable name rather than a hash since it must be
    # manually selected. N=Number, F=Features, L=Languages
    suffix = f"N={len(y)}_F={model.n_features_in_}_L={y.nunique()}"
    path = Path(f"models/model__{suffix}.json")
    print(f"Model JSON saved to {path}")
    path.write_text(json.dumps(model_dict, indent=2))

    return path


def reorder_model_and_data(
    model: LogisticRegression,
    X: pd.DataFrame,
    X_trn: pd.DataFrame,
    X_val: pd.DataFrame,
) -> tuple[LogisticRegression, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    importance = np.mean(np.abs(model.coef_), axis=0)
    feature_order = np.argsort(-importance)
    ordered_features = [model.feature_names_in_[i] for i in feature_order]

    model.coef_ = model.coef_[:, feature_order]
    model.feature_names_in_ = np.array(ordered_features)

    X = X[ordered_features]
    X_trn = X_trn[ordered_features]
    X_val = X_val[ordered_features]

    return model, X, X_trn, X_val


# We go crazy with the return value for easier exploration
class TrainResult(NamedTuple):
    f1: float
    model: LogisticRegression
    model_pickle_file: Path
    model_json_file: Path | None
    features: pd.DataFrame
    X_trn: pd.DataFrame
    X_val: pd.DataFrame
    y_trn: pd.Series
    y_val: pd.Series
    preds: np.ndarray
    probs: np.ndarray
    wrong_df: pd.DataFrame


def train_model(
    df: pd.DataFrame,
    frac: float = 1.0,
    use_cache=True,
) -> TrainResult:
    if 0 < frac < 1:
        # Subset of rows
        df = df.groupby("Language").sample(frac=frac, random_state=0)

    df = df.reset_index(drop=True)
    features_df = generate_features(df, use_cache=use_cache)

    X = features_df.drop(columns=["Target"])
    y = features_df["Target"]

    X_trn, X_val, y_trn, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Cache keyed by dataset stats + feature set + model name (no hyperparams).
    target_counts = features_df.Target.value_counts().sort_index()
    fingerprint = (
        "model=LogisticRegression\n"
        f"rows={len(features_df)}\n"
        f"mean_len={df.Snippet.str.len().mean()}\n"
        f"features={len(X.columns)}\n"
        f"{target_counts.to_string()}\n"
        f"columns:\n" + "\n".join(features_df.columns)
    )
    model_hash = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12]
    model_pickle_file = Path(f"models/model_{model_hash}.pkl")
    model_pickle_file.parent.mkdir(exist_ok=True)
    if model_pickle_file.exists() and use_cache:
        print(f"⚡ Using cached model from {model_pickle_file}")
        model: LogisticRegression = pickle.loads(model_pickle_file.read_bytes())
    else:
        with stopwatch(
            f"Training on {len(features_df):,} rows, {len(X.columns)} features, and {y.nunique()} languages"
        ):
            model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            )
            model.fit(X_trn, y_trn)

    # We reorder model weights by importance, and reorder data to match
    model, X, X_trn, X_val = reorder_model_and_data(
        model=model,
        X=X,
        X_trn=X_trn,
        X_val=X_val,
    )

    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)

    # Accuracy is misleading for imbalanced classes (per-snippet counts skew).
    f1 = f1_score(y_val, preds, average="macro")
    print(f"F1 (macro): {f1:.1%}")

    # Save the model object (for Python inference)
    model_pickle_file.write_bytes(pickle.dumps(model))

    # Save the model as JSON (for JS inference)
    model_json_file = save_model_json(
        model,
        y=y,
    )

    # Collect just the incorrect predictions
    wrong_df = pd.DataFrame(
        dict(
            Snippet=df.Snippet.loc[y_val.index],
            Language=y_val.to_list(),
            Pred=preds,
            Prob=np.max(probs, axis=1),
        )
    )
    wrong_df = wrong_df[wrong_df.Language.ne(wrong_df.Pred)]

    return TrainResult(
        f1=f1,
        model=model,
        model_pickle_file=model_pickle_file,
        model_json_file=model_json_file,
        features=features_df,
        X_trn=X_trn,
        X_val=X_val,
        y_trn=y_trn,
        y_val=y_val,
        preds=preds,
        probs=probs,
        wrong_df=wrong_df,
    )


if __name__ == "__main__":
    # df = pd.read_parquet("E:/Datasets/the_stack_whole_files.parquet")
    # df = pd.read_parquet("E:/Datasets/the_stack_20_line_snippets.parquet")
    df = pd.read_parquet("E:/Datasets/the_stack_10_line_snippets.parquet")

    with stopwatch("train_model"):
        result = train_model(
            df=df,
            frac=0.1,
            # use_cache=False,
        )
