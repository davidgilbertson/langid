from pathlib import Path
import re

from data.utils import get_csn_data
from utils import stopwatch


def extract_features(snippet: str) -> dict[str, bool]:
    # PascalCase keys make later Pandas attribute access like df.HasConst possible.
    text = snippet or ""
    lower = text.lower()

    def is_word_char(ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    def has_word_in(haystack: str, word: str) -> bool:
        if not word:
            return False
        start = 0
        while True:
            idx = haystack.find(word, start)
            if idx == -1:
                return False
            before_ok = idx == 0 or not is_word_char(haystack[idx - 1])
            after_idx = idx + len(word)
            after_ok = after_idx == len(haystack) or not is_word_char(
                haystack[after_idx]
            )
            if before_ok and after_ok:
                return True
            start = idx + 1

    def has_word(word: str) -> bool:
        return has_word_in(text, word)

    def has_symbol(sym: str) -> bool:
        return sym in text

    def has_regex(pattern: str) -> bool:
        return re.search(pattern, text, re.MULTILINE) is not None

    def has_line_starting_with(ch: str) -> bool:
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(ch):
                return True
        return False

    def has_indent_colons() -> bool:
        for line in text.splitlines():
            stripped = line.rstrip()
            if not stripped:
                continue
            if "#" in stripped:
                stripped = stripped.split("#", 1)[0].rstrip()
            if stripped.endswith(":"):
                return True
        return False

    def has_leading_indent(width: int) -> bool:
        if width <= 0:
            return False
        for line in text.splitlines():
            if not line.strip():
                continue
            if line.startswith(" " * width):
                return True
        return False

    def has_dollar_var() -> bool:
        for idx, ch in enumerate(text):
            if ch != "$":
                continue
            if idx + 1 < len(text) and (
                text[idx + 1].isalpha() or text[idx + 1] == "_"
            ):
                return True
        return False

    return {
        # Python-only-ish keywords
        "HasNoneTC": has_word("None"),
        "HasSelf": has_word("self"),
        "HasTrueTC": has_word("True"),
        "HasFalseTC": has_word("False"),
        "HasLambda": has_word("lambda"),
        "HasIndentColon": has_indent_colons(),
        "HasTripleQuotes": has_symbol('"""') or has_symbol("'''"),
        # Java-only-ish keywords
        "HasPublic": has_word("public"),
        "HasStatic": has_word("static"),
        "HasVoid": has_word("void"),
        "HasNew": has_word("new"),
        "HasPackage": has_word("package"),
        "HasExtends": has_word("extends"),
        "HasImplements": has_word("implements"),
        "HasAtSymbol": "@" in text,
        # JavaScript-only-ish keywords
        "HasLet": has_word("let"),
        "HasConst": has_word("const"),
        "HasVar": has_word("var"),
        "HasFatArrow": has_symbol("=>"),
        "HasExport": has_word("export"),
        "HasDefault": has_word("default"),
        # Go-only-ish keywords
        "HasFunc": has_word("func"),
        "HasColonEquals": has_symbol(":="),
        "HasDefer": has_word("defer"),
        "HasGoKeyword": has_word("go"),
        "HasChan": has_word("chan"),
        "HasLeftArrow": has_symbol("<-"),
        # Ruby-only-ish keywords
        "HasEnd": has_word("end"),
        "HasModule": has_word("module"),
        "HasRequire": has_word("require"),
        "HasDo": has_word("do"),
        "HasPuts": has_word("puts"),
        # PHP-only-ish markers
        "HasPhpTag": has_symbol("<?php"),
        "HasDollarVar": has_dollar_var(),
        "HasNamespace": has_word("namespace"),
        "HasUse": has_word("use"),
        "HasEcho": has_word("echo"),
        "HasThinArrow": has_symbol("->"),
        "HasDoubleColon": has_symbol("::"),
        # Shared keywords: Python + JavaScript
        "HasAsync": has_word("async"),
        "HasAwait": has_word("await"),
        # Shared keywords: Python + Ruby (+ others)
        "HasDef": has_word("def"),
        # Shared keywords: JavaScript + PHP
        "HasFunction": has_word("function"),
        # Shared structure / punctuation
        "HasImport": has_word("import"),
        "HasFromImport": has_word("from") and has_word("import"),
        "HasClass": has_word("class"),
        "HasSemicolon": ";" in text,
        "HasHashComment": "#" in text,
        "HasDecoratorLine": has_line_starting_with("@"),
        "HasThisDot": "this." in text,
        "HasThisArrow": "$this->" in text,
        "HasDollarThis": "$this" in text,
        "HasKwargs": "**kwargs" in text,
        "HasArgs": "*args" in text,
        "HasAngleQuestion": "<?" in text,
        "HasAngleBrackets": "<" in text and ">" in text,
        "HasDollarParen": "($" in text,
        "HasDotPlus": ".+" in text,
        "HasPlusEquals": "+=" in text,
        "HasAmpDollar": "&$" in text,
        "HasSpaceColonSpace": " : " in text,
        "HasSlashStarHash": "/*#" in text,
        "HasPublic": has_word("public"),
        "HasPrivate": has_word("private"),
        "HasProtected": has_word("protected"),
        "HasBool": has_word("bool"),
        "HasBoolean": has_word("boolean"),
        "HasStr": has_word("str"),
        "HasString": has_word("string"),
        "HasStringTC": has_word("String"),
        "HasStdString": has_word("std") and has_symbol("::") and has_word("string"),
        "HasCString": has_word("cstring"),
        "HasChar": has_word("char"),
        "HasByte": has_word("byte"),
        "HasBytes": has_word("bytes"),
        "HasStrlen": has_word("strlen"),
        "HasTabs": "\t" in text,
        "HasIndentTwoSpaces": has_leading_indent(2),
        "HasIndentFourSpaces": has_leading_indent(4),
        # Higher-signal regex hints
        "HasDefLineColon": has_regex(r"^\s*def\s+\w+\s*\(.*\)\s*:"),
        "HasGenericWildcard": has_regex(r"<\?\s*(extends|super)\b"),
    }


def extract_features_batch(snippets: list[str]) -> dict[str, list[bool]]:
    features: dict[str, list[bool]] = {}
    for snippet in snippets:
        row = extract_features(snippet)
        for key, value in row.items():
            features.setdefault(key, []).append(value)
    return features


if __name__ == "__main__":
    dataset = get_csn_data()
    output_dir = Path("features")
    output_dir.mkdir(exist_ok=True)

    feature_keys = list(extract_features("").keys())
    for split in ["train", "test", "valid"]:
        split_data = dataset[split]
        with stopwatch(f"extract {split} features") as sw:
            split_features = split_data.map(
                lambda batch: extract_features_batch(batch["Snippet"]),
                batched=True,
            )
            run_time = sw.get_time_ms()

        split_features = split_features.rename_column("Language", "Target")
        keep_columns = feature_keys + ["Target"]
        drop_columns = [
            col for col in split_features.column_names if col not in keep_columns
        ]
        if drop_columns:
            split_features = split_features.remove_columns(drop_columns)

        items_per_second = len(split_features) / (run_time / 1000)
        print(f"{split}: {items_per_second:,.0f} items/second")

        output_path = output_dir / f"features_{split}.parquet"
        split_features.to_parquet(output_path)
