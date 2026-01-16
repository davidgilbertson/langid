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


def get_stack_data() -> pd.DataFrame:
    # return pd.read_parquet("E:/Datasets/the_stack_whole_files.parquet")
    return pd.read_parquet("E:/Datasets/the_stack_10_line_snippets.parquet")
    # return pd.read_parquet("E:/Datasets/the_stack_20_line_snippets.parquet")


if __name__ == "__main__":
    df = get_stack_data()
