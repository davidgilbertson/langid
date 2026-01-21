import re
import pandas as pd
from sklearn.metrics import f1_score as f1
from data.utils import get_csn_data

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = get_csn_data()["valid"].to_pandas()


def predict(text: str) -> str:
    text = (
        text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\r\n", "\n").lstrip()
    )

    # Must be Go.
    if re.search(r"(?m)^\s*package\s+\w+", text) and re.search(
        r"(?m)^\s*func\b\s*(?:\([^)]+\)\s*)?\w+\s*\(", text
    ):
        return "go"
    if re.search(r"(?m)^\s*func\b\s*(?:\([^)]+\)\s*)?\w+\s*\(", text):
        return "go"
    if re.search(r"(?m)\b:=\b", text) and re.search(
        r"(?m)^\s*func\b\s*(?:\([^)]+\)\s*)?\w+\s*\(", text
    ):
        return "go"

    # Must be Python or Ruby.
    if re.search(r"(?m)^\s*(?:async\s+def|def)\s+\S+", text):
        lines = text.splitlines()
        def_line_index = 0
        for idx, raw in enumerate(lines):
            if re.match(r"^\s*(?:async\s+def|def)\s+\S+", raw):
                def_line_index = idx
                break
        sig_lines = lines[def_line_index : def_line_index + 8]
        sig_text = "\n".join(sig_lines)

        if re.search(r"(?m):\s*$", sig_text):
            return "python"
        if re.search(r"(?m)\)\s*:\s*$", sig_text):
            return "python"
        if re.search(r"(?m)^\s*end\b", text):
            return "ruby"
        if re.search(r"(?m)^\s*(elif|except|with|async|await)\b", text):
            return "python"
        if re.search(r"(?m)^\s*(elsif|unless|rescue|ensure)\b", text):
            return "ruby"
        if re.search(r"(?m)\bNone\b|\bTrue\b|\bFalse\b", text):
            return "python"
        if re.search(r"(?m)\bnil\b", text):
            return "ruby"
        if re.search(r"(?m)@[A-Za-z_]\w*", text):
            return "ruby"
        if re.search(r"(?m)=>", text):
            return "ruby"
        return "python"

    # Must be PHP.
    if re.search(
        r"(?m)^\s*(?:(?:final|public|private|protected|static)\s+){1,3}function\b",
        text,
    ):
        return "php"
    if re.search(r"(?m)^\s*(namespace|use)\s+\\", text):
        return "php"
    if re.search(r"(?m)^\s*<\?php", text):
        return "php"

    # Must be JavaScript or PHP.
    if re.search(r"(?m)^\s*(?:async\s+)?function\b", text):
        if re.search(r"(?m)^\s*function\s*\(", text):
            if re.search(r"(?m)\b(module|exports|require|__webpack_require__)\b", text):
                return "javascript"
            if re.search(r"(?m)^\s*(?:var|let|const)\s+", text):
                return "javascript"
            if re.search(r"(?m)\b(this|window|document|console)\b", text):
                return "javascript"
            if re.search(r"\$[A-Za-z_]\w*", text) or re.search(
                r"(->|::|\$_[A-Za-z_]\w*|\$this|<\?php)", text
            ):
                return "php"
            return "javascript"
        if re.search(r"(->|::|\$_[A-Za-z_]\w*|\$this|<\?php)", text):
            return "php"
        if re.search(r"(?m)^\s*(?:var|let|const)\s+", text):
            return "javascript"
        if re.search(r"(?m)\b(this|window|document|console)\b", text):
            return "javascript"
        if re.search(r"\$[A-Za-z_]\w*", text):
            return "php"
        return "javascript"

    # Must be Java.
    if re.search(r"(?m)^\s*(public|private|protected)\s+class\b", text):
        return "java"
    if re.search(r"(?m)^\s*import\s+java\.", text):
        return "java"
    if re.search(r"(?m)^\s*@\w+.*\b(public|private|protected|static)\b", text):
        return "java"
    if re.search(
        r"(?m)^\s*(public|private|protected|static|final|abstract|synchronized|native|strictfp|default)\s+",
        text,
    ):
        return "java"
    if re.search(
        r"(?m)^\s*(?:static\s+)?(?:<[^>]+>\s+)?(?:void|boolean|byte|short|int|long|float|double|char|[A-Z][\w<>,\s\[\]\.]*)\s+\w+\s*\(",
        text,
    ):
        return "java"
    if re.search(r"(?m)^[a-z_]\w*(?:\.[A-Z][\w$]*)+\s+\w+\s*\(", text):
        return "java"
    if re.search(r"(?m)^\s*@\w+", text) and re.search(
        r"(?m)^\s*(?:<[^>]+>\s+)?[A-Z][\w<>\[\]\s?&.,]*\s+\w+\s*\(",
        text,
    ):
        return "java"

    return "unknown"


df["Pred"] = df["Snippet"].apply(predict)
score = f1(
    df.Language,
    df.Pred,
    average="macro",
)
print(f"F1: {score:.3f}")

wrong_df = df[df.Language.ne(df.Pred)]


def preview(text: str) -> str:
    clean = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\r\n", "\n")
    return "\n".join(clean.splitlines()[:2]) if clean else ""


wrong_df_sample = wrong_df.groupby("Language").head().copy()
wrong_df_sample["Preview"] = wrong_df_sample["Snippet"].apply(preview)
print(wrong_df_sample[["Language", "Pred", "Preview"]].to_string(index=True))
