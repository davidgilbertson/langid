from pathlib import Path
import pickle

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from features import generate_features


# df = pd.read_parquet("E:/Datasets/the_stack_10_line_snippets.parquet")
# df = pd.read_parquet("E:/Datasets/the_stack_20_line_snippets.parquet")
df = pd.read_parquet("E:/Datasets/the_stack_whole_files.parquet")
df = df[df.Language.isin(["Go", "Java", "JavaScript", "PHP", "Python", "Ruby"])]

features_df = generate_features(df=df)
assert len(df) == len(features_df)

model: LogisticRegression = pickle.loads(
    # Path("model__whole_files__31_langs.pkl").read_bytes()
    Path("model__whole_files__6_langs.pkl").read_bytes()
)
# %%

X = features_df.drop(columns=["Target"])
predictions = model.predict(X)
f1 = f1_score(df.Language, predictions, average="macro")
print(f"F1: {f1:.1%}")
