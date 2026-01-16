from tools import stopwatch
from transformers import pipeline
from sklearn.metrics import f1_score
import pandas as pd

###################################
############  RESULTS  ############
###################################
# the_stack_10_line_snippets: 88.6%
# the_stack_20_line_snippets: 91.4%
# the_stack_files: 94.3%

# df = pd.read_parquet("E:/Datasets/the_stack_10_line_snippets.parquet")
# df = pd.read_parquet("E:/Datasets/the_stack_20_line_snippets.parquet")
df = pd.read_parquet("E:/Datasets/the_stack_whole_files.parquet")

# This model only does six languages
df = df[df.Language.isin(["Go", "Java", "JavaScript", "PHP", "Python", "Ruby"])]

classifier = pipeline(
    "text-classification",
    model="huggingface/CodeBERTa-language-id",
    truncation=True,
)

with stopwatch(f"CodeBERTA inference over {len(df)} items"):
    outputs: list[dict] = classifier(df.Snippet.tolist(), top_k=1, truncation=True)

predictions = [output[0]["label"] for output in outputs]
targets = df.Language.replace(
    dict(
        Go="go",
        Java="java",
        JavaScript="javascript",
        PHP="php",
        Python="python",
        Ruby="ruby",
    )
)
f1 = f1_score(targets, predictions, average="macro")
print(f"F1: {f1:.1%}")
