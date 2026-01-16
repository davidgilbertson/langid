"""
This takes The Stack V2 data, where each row is a file, and does the following:
1. Drops dual-language files (manually identified cases of JS in PHP, CSS-in-HTML etc.
2. Removes sub-languages. E.g. codefences from Markdown, <script> tags from HTML
3. With these pure files, it then splits them up into 10-line snippets
"""

import re

from datasets import Dataset, load_from_disk
import pandas as pd

from tools import stopwatch


def _find_tag_end(text: str, start: int) -> int | None:
    quote = None
    i = start
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = None
        else:
            if ch in ('"', "'"):
                quote = ch
            elif ch == ">":
                return i
        i += 1
    return None


def _strip_rawtext_tag(
    text: str, tag_name: str, language: str, file_id: str
) -> tuple[str, bool]:
    lower = text.lower()
    open_token = f"<{tag_name}"
    close_token = f"</{tag_name}>"

    out = []
    i = 0
    removed_any = False

    while True:
        idx = lower.find(open_token, i)
        if idx == -1:
            out.append(text[i:])
            break

        after = idx + len(open_token)
        if after < len(lower) and lower[after].isalnum():
            i = after
            continue

        out.append(text[i:idx])
        tag_end = _find_tag_end(text, after)
        if tag_end is None:
            print(f"ERROR: malformed <{tag_name}> blocks in {language} file {file_id}")
            raise ValueError(f"Unclosed <{tag_name}> tag in file {file_id}")

        out.append(text[idx : tag_end + 1])
        close_idx = lower.find(close_token, tag_end + 1)
        if close_idx == -1:
            print(f"ERROR: malformed <{tag_name}> blocks in {language} file {file_id}")
            raise ValueError(f"Missing </{tag_name}> tag in file {file_id}")

        out.append(text[close_idx : close_idx + len(close_token)])
        i = close_idx + len(close_token)
        removed_any = True

    return "".join(out), removed_any


def remove_sub_language(row) -> dict:
    language = row.get("Language")
    snippet = row.get("Snippet")
    file_id = row.get("Id", row.get("id", "unknown"))

    if snippet is None:
        return {"Snippet": snippet}

    if language == "Markdown":
        fence_count = snippet.count("```")
        if fence_count % 2 != 0:
            print(f"ERROR: malformed code fences in Markdown file {file_id}")
            raise ValueError(f"Uneven code fences in file {file_id}")

        def _strip_fenced_block(match: re.Match) -> str:
            text = match.group(0)
            fence_open = text.find("```")
            fence_close = text.rfind("```")
            if fence_open == -1 or fence_close == -1 or fence_close == fence_open:
                return text
            return text[: fence_open + 3] + text[fence_close:]

        cleaned = re.sub(r"(?s)```.*?```", _strip_fenced_block, snippet)
        if cleaned != snippet:
            print(f"removed fenced code contents from Markdown file {file_id}")
        snippet = cleaned

    if language == "SQL":
        if re.search(r"(?is)<html\b.*?>.*?</html>", snippet):
            snippet = re.sub(r"(?is)<html\b.*?>.*?</html>", "", snippet)
            print(f"removed <html> block from SQL file {file_id}")

    if language in ["PHP", "HTML"]:
        snippet, removed_script = _strip_rawtext_tag(
            snippet, "script", language, file_id
        )
        if removed_script:
            print(f"removed <script> contents from {language} file {file_id}")

        snippet, removed_style = _strip_rawtext_tag(snippet, "style", language, file_id)
        if removed_style:
            print(f"removed <style> contents from {language} file {file_id}")

    return {"Snippet": snippet}


def _split_snippet(snippet: str, snippet_limit: int) -> list[str]:
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
    return chunks


def _chunk_metrics(chunk: str) -> dict[str, float | int]:
    total_chars = len(chunk)
    if total_chars == 0:
        return {
            "TotalChars": 0,
            "LineCount": 0,
            "NonEmptyLineCount": 0,
            "LetterRatio": 0.0,
            "DigitRatio": 0.0,
            "WhitespaceRatio": 0.0,
            "PunctRatio": 0.0,
            "AlnumRatio": 0.0,
        }

    letters = sum(c.isalpha() for c in chunk)
    digits = sum(c.isdigit() for c in chunk)
    alnum = sum(c.isalnum() for c in chunk)
    whitespace = sum(c.isspace() for c in chunk)
    punct = total_chars - letters - digits - whitespace
    lines = chunk.splitlines()
    non_empty_lines = sum(1 for line in lines if line.strip())

    return {
        "DigitRatio": digits / total_chars,
        "WhitespaceRatio": whitespace / total_chars,
        "PunctRatio": punct / total_chars,
        "AlnumRatio": alnum / total_chars,
        "TotalChars": total_chars,
        "LineCount": len(lines),
        "NonEmptyLineCount": non_empty_lines,
        "LetterRatio": letters / total_chars,
    }


def split_snippets(snippet_rows, snippet_limit: int = 10):
    rows = []
    drop_count = 0
    for i, row in enumerate(snippet_rows):
        print(f"{i+1}/{len(snippet_rows)}", end="\r")
        snippet = row.get("Snippet")
        chunks = _split_snippet(snippet, snippet_limit)
        for chunk in chunks:
            metrics = _chunk_metrics(chunk)

            # Filter conditions that indicate junk
            if any(
                [
                    metrics["TotalChars"] < 75,
                    metrics["TotalChars"] > 50_000,
                    metrics["LineCount"] > 100,
                    metrics["LetterRatio"] < 0.01,
                    metrics["LetterRatio"] > 0.9,
                    metrics["PunctRatio"] < 0.01,
                    metrics["PunctRatio"] > 0.4,
                    metrics["DigitRatio"] > 0.5,
                ]
            ):
                drop_count += 1
                continue

            rows.append(
                {
                    "FileId": row.get("Id"),
                    "Language": row.get("Language"),
                    "Snippet": chunk,
                    # **metrics,
                }
            )
    print(f"Dropped {drop_count} junky snippets")
    return rows


if __name__ == "__main__":
    n = 20
    ds = load_from_disk("E:/Datasets/the_stack_select")
    ds = ds.filter(
        lambda row: row.get("Language") not in ["Text", "Less", "SCSS", "Diff"]
    )

    cleaned_ds = ds.map(remove_sub_language)

    with stopwatch("split snippets"):
        snippets_rows = split_snippets(cleaned_ds, snippet_limit=n)
    snippets_ds = Dataset.from_list(snippets_rows)
    snippet_df = snippets_ds.to_pandas()

    lang_counts = snippet_df.Language.value_counts()
    outlier_ratio = lang_counts.iloc[0] / lang_counts.iloc[1]
    if outlier_ratio > 2:
        lang_df = snippet_df[snippet_df.Language == lang_counts.index[0]]
        rest_df = snippet_df[snippet_df.Language != lang_counts.index[0]]
        snippet_df = pd.concat(
            [
                lang_df.sample(frac=1 / outlier_ratio),
                rest_df,
            ]
        )

    snippet_df.to_parquet(
        f"E:/Datasets/the_stack_{n}_line_snippets.parquet",
        index=False,
    )
