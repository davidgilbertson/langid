import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from data.utils import get_csn_data
from utils import stopwatch

train_df = pd.read_parquet("features/features_train.parquet")
valid_df = pd.read_parquet("features/features_valid.parquet")

X_trn = train_df.drop(columns=["Target"])
y_trn = train_df["Target"]

X_val = valid_df.drop(columns=["Target"])
y_val = valid_df["Target"]

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

df = get_csn_data()["valid"].to_pandas()
df["Pred"] = preds
df["Confidence"] = np.max(probs, axis=1)
df_wrong = df[df.Language.ne(df.Pred)]
