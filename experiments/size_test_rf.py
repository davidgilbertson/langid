
import pickle
import json
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree
from sklearn.model_selection import train_test_split
from tools import stopwatch

# Load data
print("Loading data...")
df = pd.read_parquet("features/features.parquet")
X = df.drop(columns=["Target"])
y = df["Target"]

X_trn, X_val, y_trn, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
print("Training Random Forest...")
# Use relatively small settings to give it a fair chance at 100KB
clf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
clf.fit(X_trn, y_trn)

# 1. Pickle Size
pickle_path = "rf_model.pkl"
with open(pickle_path, "wb") as f:
    pickle.dump(clf, f)
pickle_size_kb = os.path.getsize(pickle_path) / 1024
print(f"Pickle Size (n=10, depth=10): {pickle_size_kb:.2f} KB")

# 2. JSON Size (Simulating a highly optimized JS format)
# A tree is basically list of (feature_index, threshold, left_child, right_child, value)
def tree_to_dict(tree):
    # This is a rough approximation of the data needed
    return {
        "nodes": tree.node_count,
        "children_left": tree.children_left.tolist(),
        "children_right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        # We don't even strictly need values for leaf nodes if we just care about majority class, 
        # but let's assume we do for probabilities
    }

forest_data = []
for estimator in clf.estimators_:
    forest_data.append(tree_to_dict(estimator.tree_))

json_path = "rf_model.json"
with open(json_path, "w") as f:
    json.dump(forest_data, f)
json_size_kb = os.path.getsize(json_path) / 1024
print(f"Rough JSON Size (n=10, depth=10): {json_size_kb:.2f} KB")

# Train a bigger one to compare
print("\nTraining Larger Random Forest (n=100)...")
clf_large = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
clf_large.fit(X_trn, y_trn)
with open("rf_large.pkl", "wb") as f:
    pickle.dump(clf_large, f)
print(f"Pickle Size (n=100, depth=None): {os.path.getsize('rf_large.pkl') / 1024:.2f} KB")
