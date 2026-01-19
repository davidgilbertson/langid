import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC

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

models = [
    LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
    ),
    BernoulliNB(),
    MultinomialNB(),
    LinearSVC(),
    Perceptron(max_iter=1000),
    SGDClassifier(loss="log_loss", max_iter=1000),
    RandomForestClassifier(n_estimators=200, random_state=42),
]
for model in models:
    model_name = model.__class__.__name__
    with stopwatch(f"train {model_name}"):
        model.fit(X_trn, y_trn)

        preds = model.predict(X_tst)

    acc = accuracy_score(y_tst, preds)
    print(f"Accuracy: {acc:.0%}")
