from typing import Dict

import pandas as pd

from data.utils import get_smola_data
from utils import stopwatch


def extract_features(snippet: str) -> Dict[str, bool]:
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

    def has_word_ci(word: str) -> bool:
        return has_word_in(lower, word.lower())

    def has_any_word(words: list[str]) -> bool:
        for word in words:
            if has_word(word):
                return True
        return False

    def has_symbol(sym: str) -> bool:
        return sym in text

    def has_closing_tag() -> bool:
        start = 0
        while True:
            idx = text.find("</", start)
            if idx == -1:
                return False
            after_idx = idx + 2
            if after_idx < len(text) and text[after_idx].isalpha():
                return True
            start = idx + 2

    def has_yaml_key_value() -> bool:
        for line in text.splitlines():
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                continue
            key, rest = stripped.split(":", 1)
            if (
                key
                and key.replace("-", "").replace("_", "").isalnum()
                and rest.startswith(" ")
            ):
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

    def has_matlab_percent_comment() -> bool:
        for line in text.splitlines():
            if line.lstrip().startswith("%"):
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

    def has_rust_macro() -> bool:
        for idx, ch in enumerate(text):
            if ch != "!":
                continue
            if idx > 0 and is_word_char(text[idx - 1]):
                return True
        return False

    return {
        # Common reserved words / tokens
        "HasConst": "const" in text,
        "HasLet": "let" in text,
        "HasVar": "var" in text,
        "HasDef": "def" in text,
        "HasClass": "class" in text,
        "HasFunction": "function" in text,
        "HasFn": "fn" in text,
        "HasStruct": "struct" in text,
        "HasModule": "module" in text,
        "HasNamespace": "namespace" in text,
        "HasInterface": "interface" in text,
        "HasImplements": "implements" in text,
        "HasExtends": "extends" in text,
        "HasImport": "import" in text,
        "HasUsing": "using" in text,
        "HasInclude": "include" in text,
        "HasRequire": "require" in text,
        "HasPackage": "package" in text,
        "HasPublic": "public" in text,
        "HasPrivate": "private" in text,
        "HasProtected": "protected" in text,
        "HasStatic": "static" in text,
        "HasThis": "this" in text,
        "HasSelf": "self" in text,
        "HasNull": "null" in text,
        "HasNil": "nil" in text,
        "HasNoneTC": "None" in text,
        "HasTrueLC": "true" in text,
        "HasFalseLC": "false" in text,
        "HasTrueTC": "True" in text,
        "HasFalseTC": "False" in text,
        "HasFromImport": has_word("from") and has_word("import"),
        # Symbols / operators often tied to specific languages
        "HasColonEquals": has_symbol(":="),
        "HasFatArrow": has_symbol("=>"),
        "HasThinArrow": has_symbol("->"),
        "HasDoubleColon": has_symbol("::"),
        "HasElif": has_word("elif"),
        "HasEnd": has_word("end"),
        "HasThen": has_word("then"),
        "HasFi": has_word("fi"),
        "HasBegin": has_word("begin"),
        "HasHashBang": text.startswith("#!"),
        "HasTripleQuotes": has_symbol('"""') or has_symbol("'''"),
        #
        # # XML / markup hints
        "HasXmlDeclaration": has_symbol("<?xml"),
        "HasClosingTag": has_closing_tag(),
        "HasDoctype": has_word_ci("doctype"),
        #
        # # YAML hints
        "HasYamlDocStart": has_symbol("---"),
        "HasYamlDocEnd": has_symbol("..."),
        "HasYamlKeyValue": has_yaml_key_value(),
        # SQL hints
        "HasSelect": has_word_ci("select"),
        "HasFrom": has_word_ci("from"),
        "HasWhere": has_word_ci("where"),
        "HasJoin": has_word_ci("join"),
        #
        # # Python hints
        "HasIndentColons": has_indent_colons(),
        "HasPrintFunc": has_word("print"),
        "HasLambda": has_word("lambda"),
        #
        # # Java / C / Go hints
        "HasSemicolons": ";" in text,
        "HasDoubleSlashComment": "//" in text,
        "HasCBlockComment": "/*" in text,
        "HasGoFunc": has_word("func"),
        "HasGoPackage": has_word("package"),
        #
        # # JavaScript / TypeScript hints
        "HasArrowFunction": "=>" in text,
        "HasAsync": has_word("async"),
        "HasAwait": has_word("await"),
        "HasExport": has_word("export"),
        "HasDefault": has_word("default"),
        #
        # # PHP hints
        "HasPhpTag": has_symbol("<?php"),
        "HasDollarVar": has_dollar_var(),
        #
        # # Shell hints
        "HasShebang": text.startswith("#!"),
        "HasEcho": has_word("echo"),
        "HasFiElse": has_word("fi") and has_word("then"),
        #
        # # R hints
        "HasRAssign": has_symbol("<-"),
        "HasRFunction": has_word("function"),
        #
        # # MATLAB / Octave hints
        "HasMatlabEnd": has_word("end"),
        "HasMatlabFunction": has_word("function"),
        "HasMatlabPercentComment": has_matlab_percent_comment(),
        #
        # # Rust hints
        "HasRustUse": has_word("use"),
        "HasRustImpl": has_word("impl"),
        "HasRustMacro": has_rust_macro(),
        #
        # # Lua hints
        "HasLuaLocal": has_word("local"),
        "HasLuaThen": has_word("then"),
        #
        # # ABAP / Coq / Eiffel / ASP.NET hints
        "HasAbapData": has_word_ci("data") and has_symbol("."),
        "HasCoqLemma": has_word("Lemma"),
        "HasEiffelDo": has_word("do"),
        # # Round 3 targeted hints
        "HasAspectJAspect": has_word("aspect"),
        "HasAspectJPointcut": has_word("pointcut"),
        "HasNemerleMacro": has_word("macro"),
        "HasNemerleSyntax": has_word("syntax"),
        "HasSystemVerilogLogic": has_word("logic"),
        "HasSystemVerilogAlwaysFf": has_word("always_ff"),
        "HasSystemVerilogAlwaysComb": has_word("always_comb"),
        "HasSystemVerilogModport": has_word("modport"),
        "HasEndModule": has_word("endmodule"),
        "HasRubyPuts": has_word("puts"),
        "HasRubyAttrAccessor": has_any_word(
            ["attr_accessor", "attr_reader", "attr_writer"]
        ),
        "HasCrystalPointerof": has_word("pointerof"),
        "HasCrystalMacro": has_word("macro"),
        "HasHaxeTypedef": has_word("typedef"),
        "HasHaxeInline": has_word("inline"),
        "HasHaxeMacro": has_word("macro"),
        "HasIdrisTotal": has_word("total"),
        "HasIdrisMutual": has_word("mutual"),
    }


if __name__ == "__main__":
    df = get_smola_data()
    # df = df.sample(10)
    with stopwatch("extract features") as sw:
        features = [extract_features(row.Snippet) for i, row in df.iterrows()]
        run_time = sw.get_time_ms()

    print(f"{len(df) / (run_time / 1000):,.0f} items/second")

    feature_df = pd.DataFrame(features)
    feature_df["Target"] = df.Language
    feature_df.to_csv("features.csv", index=False)
