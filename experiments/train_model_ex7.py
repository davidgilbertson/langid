import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

with stopwatch("do it"):
    X_trn["Target"] = y_trn.values
    centroids = X_trn.groupby("Target").mean()

    class_labels = centroids.index.to_numpy()
    centroid_matrix = centroids.to_numpy()
    X_val_matrix = X_val.to_numpy()

    x_sq = np.sum(X_val_matrix**2, axis=1, keepdims=True)
    c_sq = np.sum(centroid_matrix**2, axis=1)
    dist_sq = x_sq + c_sq - 2 * X_val_matrix @ centroid_matrix.T
    pred_indices = np.argmin(dist_sq, axis=1)
    preds = class_labels[pred_indices]

accuracy = (preds == y_val).mean()
print(f"Accuracy: {accuracy:.0%}")
