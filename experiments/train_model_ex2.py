import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils import stopwatch

df = pd.read_csv("features.csv")

X = df.drop(columns=["Target"])
y = df["Target"]

X_trn, X_tst, y_trn, y_tst = train_test_split(
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

    preds = model.predict(X_tst)
acc = accuracy_score(y_tst, preds)
print(f"Accuracy: {acc:.0%}")
