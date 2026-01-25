import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tools import stopwatch

df = pd.read_parquet("features.parquet")

X = df.drop(columns=["Target"])
y = df["Target"]

X_trn, X_val, y_trn, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()
X_trn = scaler.fit_transform(X_trn)
X_val = scaler.transform(X_val)
model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    class_weight="balanced",
)

with stopwatch(f"Training with {len(X.columns)} features"):
    model.fit(X_trn, y_trn)

    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)

acc = accuracy_score(y_val, preds)
print(f"Accuracy: {acc:.3%}")

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "logreg.json"

payload = {
    "model_type": "logistic_regression",
    "classes": model.classes_.tolist(),
    "coef": np.round(model.coef_, 4).tolist(),
    "intercept": np.round(model.intercept_, 4).tolist(),
    "scaler": {
        "mean": np.round(scaler.mean_, 6).tolist(),
        "scale": np.round(scaler.scale_, 6).tolist(),
        "var": np.round(scaler.var_, 6).tolist(),
    },
}
model_path.write_text(json.dumps(payload, indent=2))

# %%
result_df = y_val.to_frame()
result_df["Pred"] = preds
result_df["Confidence"] = np.max(probs, axis=1)
wrong_df = result_df[y_val.ne(preds)]

print("Mistakes (actual -> predicted: count):")
for (target, pred), group_df in wrong_df.groupby(["Target", "Pred"]):
    if len(group_df) > 1:
        print(f"{target} -> {pred}: {len(group_df)}")
