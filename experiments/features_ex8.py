from __future__ import annotations

from collections import Counter, defaultdict
import re

import numpy as np
import pandas as pd

from data.utils import get_stack_data
from tools import stopwatch

WORD_RE = re.compile(r"[A-Za-z_]+")

SUBSET = 1
MIN_NGRAM = 2
MAX_NGRAM = 4
TOP_K = 186
MIN_OCCURRENCES = 3


def get_ngram_counts(text: str, min_ngram: int, max_ngram: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not text:
        return counts
    length = len(text)
    for n in range(min_ngram, max_ngram + 1):
        if length < n:
            break
        for idx in range(length - n + 1):
            chunk = text[idx : idx + n]
            if len(chunk.strip()) != len(chunk):
                continue
            if not chunk.isascii():
                continue
            counts[chunk] += 1
    return counts


def get_substring_counts(snippet: str, min_ngram: int, max_ngram: int) -> Counter[str]:
    # counts = get_ngram_counts(snippet, min_ngram, max_ngram)
    counts = Counter()
    # tokens = snippet.split()
    tokens = WORD_RE.findall(snippet)
    for token in tokens:
        if len(token) <= max_ngram:
            continue
        if not token.isascii():
            continue
        counts[token] += 1
    return counts


def gini(values: np.ndarray) -> float:
    total = values.sum()
    if total == 0:
        return 0.0
    sorted_values = np.sort(values)
    cumulative = np.cumsum(sorted_values)
    n = len(values)
    return (n + 1 - 2 * np.sum(cumulative) / total) / n


dataset = get_stack_data(subset=SUBSET)
df = dataset.to_pandas()[["Language", "Snippet"]]
if df["Language"].isna().any() or df["Snippet"].isna().any():
    raise ValueError("Dataset contains null Language or Snippet values.")

language_substring_counts: dict[str, Counter[str]] = defaultdict(Counter)
language_tf_sums: dict[str, Counter[str]] = defaultdict(Counter)
language_char_totals: Counter[str] = Counter()
language_doc_totals: Counter[str] = Counter()
global_counts: Counter[str] = Counter()
global_doc_counts: Counter[str] = Counter()

with stopwatch("Count substring frequencies"):
    for language, group_df in df.groupby("Language"):
        for snippet in group_df["Snippet"]:
            if not snippet:
                continue
            counts = get_substring_counts(snippet, MIN_NGRAM, MAX_NGRAM)
            if not counts:
                continue
            token_total = counts.total()
            if token_total == 0:
                continue
            language_substring_counts[language].update(counts)
            language_tf_sums[language].update(
                {substring: count / token_total for substring, count in counts.items()}
            )
            language_char_totals[language] += len(snippet)
            language_doc_totals[language] += 1
            global_counts.update(counts)
            global_doc_counts.update(set(counts))

languages = sorted(language_char_totals)
counts_df = pd.DataFrame.from_dict(language_substring_counts, orient="index").fillna(
    0.0
)
counts_df = counts_df.loc[languages]

valid_substrings = [s for s, c in global_counts.items() if c >= MIN_OCCURRENCES]
counts_df = counts_df[valid_substrings]

denoms = pd.Series(language_char_totals).loc[languages]
freq_df = counts_df.div(denoms, axis=0)

tf_sums_df = pd.DataFrame.from_dict(language_tf_sums, orient="index").fillna(0.0)
tf_sums_df = tf_sums_df.loc[languages]
tf_sums_df = tf_sums_df[valid_substrings]

doc_counts = pd.Series(language_doc_totals).loc[languages]
total_docs = int(doc_counts.sum())
idf = pd.Series(global_doc_counts).reindex(valid_substrings).fillna(0.0)
idf = np.log((1 + total_docs) / (1 + idf)) + 1.0
tfidf_df = tf_sums_df.div(doc_counts, axis=0).mul(idf, axis=1)

with stopwatch("Score substrings"):
    records: list[dict[str, object]] = []
    freq_values = freq_df.to_numpy()
    count_values = counts_df.to_numpy()
    tfidf_values = tfidf_df.to_numpy()
    num_languages = len(languages)
    for substring, vector, count_vector, tfidf_vector in zip(
        freq_df.columns, freq_values.T, count_values.T, tfidf_values.T, strict=True
    ):
        best_idx = int(np.argmax(vector))
        best_lang = languages[best_idx]
        best_value = float(vector[best_idx])
        rest_total = float(vector.sum() - best_value)
        rest_mean = rest_total / (num_languages - 1) if num_languages > 1 else 0.0
        ratio = best_value / (rest_mean + 1e-12)
        gini_score = gini(vector) * np.log1p(vector.sum())
        best_count = int(count_vector[best_idx])
        rest_count = int(count_vector.sum() - best_count)
        best_tfidf = float(tfidf_vector[best_idx])
        rest_tfidf_mean = float(
            (tfidf_vector.sum() - best_tfidf) / (num_languages - 1)
            if num_languages > 1
            else 0.0
        )
        tfidf_ratio = best_tfidf / (rest_tfidf_mean + 1e-12)
        records.append(
            {
                "Substring": substring,
                "BestLanguage": best_lang,
                "BestFreq": best_value,
                "RestMean": rest_mean,
                "Ratio": ratio,
                "Gini": gini_score,
                "LangCount": best_count,
                "RestCount": rest_count,
                "TfIdf": best_tfidf,
                "TfIdfRatio": tfidf_ratio,
            }
        )

result_df = pd.DataFrame.from_records(records)
result_df = result_df.sort_values(["TfIdfRatio", "Ratio", "Gini"], ascending=False)
top_df = result_df.head(TOP_K)

print(
    top_df[["Substring", "BestLanguage", "TfIdfRatio"]].head(20).to_string(index=False)
)

feature_list = top_df["Substring"].to_list()
token_features = {feature for feature in feature_list if len(feature) > MAX_NGRAM}
ngram_features = [feature for feature in feature_list if len(feature) <= MAX_NGRAM]

with stopwatch("Build feature matrix"):
    rows: list[dict[str, object]] = []
    for snippet, language in zip(df["Snippet"], df["Language"], strict=True):
        tokens = set(WORD_RE.findall(snippet))
        feature_row = {"Target": language}
        for feature in ngram_features:
            feature_row[feature] = feature in snippet
        for token in token_features:
            feature_row[token] = token in tokens
        rows.append(feature_row)

feature_df = pd.DataFrame(rows)
feature_df.to_parquet("features.parquet", index=False)
