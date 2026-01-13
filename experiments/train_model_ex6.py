import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from utils import stopwatch

df = pd.read_parquet("features/features.parquet")

X = df.drop(columns=["Target"])
y = df["Target"]

X_trn, X_val, y_trn, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    class_weight="balanced",
)
with stopwatch("train model"):
    model.fit(X_trn, y_trn)

    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)

acc = accuracy_score(y_val, preds)
f1 = f1_score(y_val, preds, average="macro")
print(f"Accuracy: {acc:.3%}")
print(f"F1 (macro): {f1:.3%}")

# %%
result_df = y_val.to_frame()
result_df["Pred"] = preds
result_df["Confidence"] = np.max(probs, axis=1)
wrong_df = result_df[y_val.ne(preds)]

print("Mistakes (actual -> predicted: count):")
for (target, pred), group_df in wrong_df.groupby(["Target", "Pred"]):
    if len(group_df) > 1:
        print(f"{target} -> {pred}: {len(group_df)}")
