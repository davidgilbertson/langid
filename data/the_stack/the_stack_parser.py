"""
This file extends an existing copy of The Stack V2 with more records, and saves as a new DB
"""

import os
import boto3
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from dotenv import load_dotenv
from smart_open import open as s3open

from data.the_stack.the_stack_meta import stack_langs
from tools import stopwatch

load_dotenv()

s3 = boto3.Session(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
).client("s3")

OLD_DB_PATH = "E:/Datasets/the_stack_select"
NEW_DB_PATH = "E:/Datasets/the_stack_select_html"
SKIP = 300
TAKE = 100


def fetch_from_s3(row):
    id = row["blob_id"]
    url = f"s3://softwareheritage/content/{id}"
    with s3open(
        url,
        "rb",
        compression=".gz",
        # Reusing the client makes it ~4x faster
        transport_params={"client": s3},
    ) as f:
        data = f.read()

    encoding = row.get("src_encoding", "utf-8")
    return data.decode(encoding, errors="replace")


def get_data(langs):
    items = []
    for lang in langs:
        ds = load_dataset(
            "bigcode/the-stack-v2-dedup",
            lang,
            split="train",
            streaming=True,
        )
        ds = ds.skip(SKIP).take(TAKE)
        for row in ds:
            items.append(
                {
                    "Id": row["blob_id"],
                    "Language": lang,
                    "Snippet": fetch_from_s3(row),
                }
            )
    return items


with stopwatch("get code"):
    old_dataset = load_from_disk(OLD_DB_PATH)
    assert len(old_dataset) == len(stack_langs) * SKIP

    new_dataset = Dataset.from_list(get_data(stack_langs))

dataset = concatenate_datasets([old_dataset, new_dataset]).sort("Language")

# We assume that there is enough data to take `TAKE` more rows of every language, warn if that's not true.
if len(new_dataset) != len(stack_langs) * TAKE:
    print(f"⚠️ The new dataset has {len(new_dataset)} rows, that doesn't look right")

dataset.save_to_disk(NEW_DB_PATH)

loc = 0
for row in dataset:
    loc += len(row["Snippet"].splitlines())

print(f"You now have {loc:,.0f} lines of code")
