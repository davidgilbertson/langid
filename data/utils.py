import sqlite3
from pathlib import Path


from datasets import Dataset, DatasetDict, load_from_disk
import pandas as pd


def get_smola_data() -> pd.DataFrame:
    resolved_path = Path(__file__).with_name("local").joinpath("dataset.sqlite")
    conn = sqlite3.connect(resolved_path)
    try:
        df = pd.read_sql_query(
            "select snippet as Snippet, language as Language from snippets",
            conn,
        )
    finally:
        conn.close()
    return df


def get_csn_data(subset: float = 1.0) -> DatasetDict:
    """
    Dataset has train/test/valid splits and columns: `Snippet` and `Language`
    """
    dataset = load_from_disk(r"E:\Datasets\code_search_net")
    if subset >= 1:
        return dataset
    if subset <= 0:
        raise ValueError("subset must be > 0")

    sampled_splits: dict[str, Dataset] = {}
    for split_name, split in dataset.items():
        sample_size = max(1, int(len(split) * subset))
        sampled_splits[split_name] = split.shuffle(seed=42).select(range(sample_size))
    return DatasetDict(sampled_splits)


def _split_snippets(df: pd.DataFrame, snippet_limit: int) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in df.iterrows():
        snippet = row.Snippet
        lines = snippet.splitlines()
        if not lines:
            chunks = [""]
        else:
            chunks = []
            current_lines: list[str] = []
            non_empty = 0
            for line in lines:
                current_lines.append(line)
                if line.strip():
                    non_empty += 1
                if non_empty >= snippet_limit:
                    chunks.append("\n".join(current_lines))
                    current_lines = []
                    non_empty = 0
            if current_lines:
                chunks.append("\n".join(current_lines))
        if len(chunks) >= 2:
            chunks[-2] = "\n".join([chunks[-2], chunks[-1]])
            chunks.pop()
        for chunk in chunks:
            digit_ratio = sum(c.isdigit() for c in chunk) / max(
                1, sum(c.isalnum() for c in chunk)
            )
            if digit_ratio > 0.9:
                continue
            item = row.to_dict()
            item["Snippet"] = chunk
            rows.append(item)
    return pd.DataFrame(rows)


def get_stack_data(
    subset: float = 1.0,
    snippet_limit: int | None = None,
    languages: list[str] | None = None,
) -> pd.DataFrame:
    """
    Returned DataFrame has columns: `Id`, `Snippet` and `Language`.
    If snippet_limit is provided, each snippet is split into chunks of at most
    snippet_limit non-empty lines, expanding the number of rows.
    """
    df: pd.DataFrame = load_from_disk("E:/Datasets/the_stack_select").to_pandas()
    if languages:
        df = df[df.Language.isin(languages)].reset_index(drop=True)
    else:
        # By default, we drop a few weird ones
        # Cheating? Maybe, but differentiating CSS, Less, and SCSS
        #  from a 10-line snippet mid-file is not a goal I have.
        drop = ["Text", "Less", "SCSS", "Diff"]
        df = df[~df.Language.isin(drop)].reset_index(drop=True)

    if snippet_limit is not None and snippet_limit <= 0:
        raise ValueError("snippet_limit must be > 0")

    if subset >= 1:
        if snippet_limit is None:
            return df
        return _split_snippets(df, snippet_limit)
    if subset <= 0:
        raise ValueError("subset must be > 0")

    items = len(df)
    sample_size = max(1, int(items * subset))
    step = items / sample_size
    indices = [int(i * step) for i in range(sample_size)]
    df = df.iloc[indices].reset_index(drop=True)
    if snippet_limit is None:
        return df

    return _split_snippets(df, snippet_limit)


if __name__ == "__main__":
    df = get_stack_data(
        subset=0.01,
        # snippet_limit=10,
    )
