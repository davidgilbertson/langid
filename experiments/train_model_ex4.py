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

# labels = list(model.classes_)
# cm = confusion_matrix(y_tst, preds, labels=labels)
# cm_df = pd.DataFrame(cm, index=labels, columns=labels)
# cm_df.to_csv("confusion_matrix.csv", index=True)
#
# row_totals = cm.sum(axis=1)
# top_confusions = []
# for i, actual in enumerate(labels):
#     total = row_totals[i]
#     if total == 0:
#         continue
#     for j, predicted in enumerate(labels):
#         if i == j:
#             continue
#         count = cm[i, j]
#         if count == 0:
#             continue
#         top_confusions.append(
#             {
#                 "Actual": actual,
#                 "Predicted": predicted,
#                 "Count": int(count),
#                 "ShareOfActual": count / total,
#             }
#         )
#
# top_confusions = sorted(
#     top_confusions,
#     key=lambda row: (row["Count"], row["ShareOfActual"]),
#     reverse=True,
# )
#
# top_df = pd.DataFrame(top_confusions[:25])
# top_df.to_csv("confusion_top.csv", index=False)
#
# if not top_df.empty:
#     pretty = top_df.copy()
#     pretty["ShareOfActual"] = (pretty["ShareOfActual"] * 100).map(lambda v: f"{v:.1f}%")
#     print("\nTop confusions (Actual -> Predicted):")
#     for row in pretty.itertuples(index=False):
#         print(f"- {row.Actual} -> {row.Predicted}: {row.Count} ({row.ShareOfActual})")
#
#     header = "| Actual | Predicted | Count | Share of Actual |\n| --- | --- | --- | --- |\n"
#     rows = [
#         f"| {row.Actual} | {row.Predicted} | {row.Count} | {row.ShareOfActual} |"
#         for row in pretty.itertuples(index=False)
#     ]
#     with open("confusion_top.md", "w", encoding="utf-8") as handle:
#         handle.write(header + "\n".join(rows) + "\n")
