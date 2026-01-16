from pathlib import Path
import re

from data.utils import get_stack_data
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

    def has_word_ci(word: str) -> bool:
        return has_word_in(lower, word.lower())

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
        "HasPythonFString": has_regex(r"(?<!\w)f['\"]"),
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
        "HasFromImportOrder": has_regex(r"^\s*from\s+\S+\s+import\b"),
        "HasImportFromOrder": has_regex(r"^\s*import\b.+\bfrom\s+['\"]"),
        "HasImportAs": has_regex(r"^\s*(from\s+\S+\s+import|import)\s+.+\s+as\s+"),
        "HasImportStarAs": has_regex(r"^\s*import\s+\*\s+as\s+"),
        "HasImportQuoteLine": has_regex(r"^\s*import\s+['\"]"),
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
        # Diff / patch
        "HasDiffHeader": has_line_starting_with("diff --git"),
        "HasDiffHunk": has_line_starting_with("@@"),
        "HasDiffFilePlus": has_line_starting_with("+++"),
        "HasDiffFileMinus": has_line_starting_with("---"),
        # Makefile
        "HasMakeTarget": has_regex(r"^[A-Za-z0-9_.-]+:\s"),
        "HasMakeVar": has_symbol("$("),
        "HasPhony": has_word(".PHONY"),
        "HasMakeAssign": has_regex(r"^\s*[A-Za-z0-9_.-]+\s*[:?+]?="),
        "HasMakeShell": has_symbol("$(shell "),
        "HasMakeRecipeTab": has_regex(r"^\t"),
        # C / C++
        "HasIncludeStdIO": has_symbol("#include <stdio"),
        "HasPrintf": has_word("printf"),
        "HasScanf": has_word("scanf"),
        "HasIostream": has_symbol("#include <iostream"),
        "HasStdNamespace": has_symbol("std::"),
        "HasTemplate": has_word("template"),
        # Rust
        "HasRustFnArrow": has_regex(r"\bfn\s+\w+\s*\(.*\)\s*->"),
        "HasRustUse": has_word("use") and has_symbol("::"),
        "HasRustMacroRules": has_symbol("macro_rules!"),
        "HasRustDerive": has_symbol("#[derive"),
        "HasLetMut": has_word("let") and has_word("mut"),
        "HasRustMatch": has_word("match"),
        # Go
        "HasGoPackage": has_word("package"),
        "HasGoImportBlock": has_symbol("import ("),
        # CSS / Less / SCSS
        "HasCssProperty": has_regex(
            r"\b(color|display|font|margin|padding|border)\s*:"
        ),
        "HasScssVar": has_regex(r"\$[A-Za-z_-][A-Za-z0-9_-]*\s*:"),
        "HasScssMixin": has_word("@mixin"),
        "HasScssInclude": has_word("@include"),
        "HasScssExtend": has_word("@extend"),
        "HasScssInterp": has_symbol("#{"),
        "HasScssNestedAmp": has_regex(r"&[.:#\[]"),
        "HasLessVar": has_regex(r"@[A-Za-z_-][A-Za-z0-9_-]*\s*:"),
        "HasLessInterp": has_symbol("@{"),
        "HasLessGuard": has_word("when") and has_symbol("("),
        "HasLessImportOpts": has_symbol("@import ("),
        "HasLessMixinCall": has_regex(r"\.[A-Za-z_-][A-Za-z0-9_-]*\s*\("),
        "HasLessVarUse": has_regex(r"@[A-Za-z_-][A-Za-z0-9_-]*\b"),
        # SQL
        "HasSqlSelect": has_word_ci("select"),
        "HasSqlFrom": has_word_ci("from"),
        "HasSqlWhere": has_word_ci("where"),
        "HasSqlCreateTable": has_word_ci("create") and has_word_ci("table"),
        "HasSqlInsertInto": has_word_ci("insert") and has_word_ci("into"),
        # JSON / INI / YAML / Markdown / XML
        "HasJsonObject": has_symbol("{")
        and has_symbol("}")
        and has_regex(r'"[^"]+"\s*:'),
        "HasIniSection": has_regex(r"^\s*\[[^\]]+\]\s*$"),
        "HasIniKeyValue": has_regex(r"^\s*[A-Za-z0-9_.-]+\s*=\s*.+$"),
        "HasIniComment": has_regex(r"^\s*[#;]"),
        "HasYamlList": has_line_starting_with("- "),
        "HasYamlDocMarker": has_line_starting_with("---")
        or has_line_starting_with("..."),
        "HasYamlKeyValue": has_regex(r"^\s*[A-Za-z0-9_.-]+\s*:\s+.+$"),
        "HasMarkdownHeading": has_line_starting_with("#"),
        "HasMarkdownList": has_regex(r"^\s*[-*+]\s+"),
        "HasMarkdownOrdered": has_regex(r"^\s*\d+\.\s+"),
        "HasMarkdownQuote": has_regex(r"^\s*>"),
        "HasMarkdownUnderlineHeading": has_regex(r"^\s*={2,}\s*$")
        or has_regex(r"^\s*-{2,}\s*$"),
        "HasMarkdownFence": has_symbol("```"),
        "HasMarkdownLink": has_symbol("]("),
        "HasXmlDecl": has_symbol("<?xml"),
        "HasXmlNs": has_symbol("xmlns"),
        # GraphQL
        "HasGraphQlQuery": has_word_ci("query"),
        "HasGraphQlMutation": has_word_ci("mutation"),
        "HasGraphQlFragment": has_word_ci("fragment"),
        "HasGraphQlSpread": has_symbol("... on"),
        # TypeScript
        "HasTsInterface": has_word("interface"),
        "HasTsTypeAlias": has_word("type"),
        "HasTsImplements": has_word("implements"),
        "HasTsEnum": has_word("enum"),
        "HasTsAs": has_word("as"),
        "HasTsReadonly": has_word("readonly"),
        "HasTsAccessMod": has_word("public")
        or has_word("private")
        or has_word("protected"),
        # Shell
        "HasShebang": has_line_starting_with("#!"),
        "HasShellExport": has_word("export"),
        "HasShellIf": has_symbol("if [") or has_symbol("if test"),
        "HasShellFi": has_word("fi"),
        "HasShellThen": has_word("then"),
        "HasShellDo": has_word("do"),
        "HasShellDone": has_word("done"),
        "HasShellVar": has_regex(r"\$[A-Za-z_][A-Za-z0-9_]*"),
        # Perl
        "HasPerlStrict": has_symbol("use strict"),
        "HasPerlWarnings": has_symbol("use warnings"),
        "HasPerlMySigil": has_symbol("my $"),
        "HasPerlArgv": has_symbol("@ARGV"),
        "HasPerlRegexSub": has_regex(r"\bs/.+/.*?/"),
        # Lua
        "HasLuaLocal": has_word("local"),
        "HasLuaThen": has_word("then"),
        "HasLuaElseIf": has_word("elseif"),
        "HasLuaConcat": has_symbol(".."),
        # R
        "HasRAssign": has_regex(r"\s<-\s"),
        "HasRLibrary": has_word("library"),
        "HasRDataFrame": has_symbol("data.frame("),
        "HasRGgplot": has_symbol("ggplot("),
        "HasRSetSeed": has_symbol("set.seed("),
        "HasRTrueFalse": has_word("TRUE") or has_word("FALSE"),
        # C main / includes
        "HasMainFunc": has_regex(r"\bint\s+main\s*\("),
        "HasReturnZero": has_regex(r"\breturn\s+0\s*;"),
        # VB.NET
        "HasVbImports": has_word("Imports"),
        "HasVbModule": has_word("Module"),
        "HasVbSub": has_word("Sub"),
        "HasVbEndSub": has_symbol("End Sub"),
        "HasVbDim": has_word("Dim"),
        "HasVbAs": has_word("As"),
    }


def extract_features_batch(snippets: list[str]) -> dict[str, list[bool]]:
    features: dict[str, list[bool]] = {}
    for snippet in snippets:
        row = extract_features(snippet)
        for key, value in row.items():
            features.setdefault(key, []).append(value)
    return features


if __name__ == "__main__":
    dataset = get_stack_data()

    with stopwatch("extract features") as sw:
        features = dataset.map(
            lambda batch: extract_features_batch(batch["Snippet"]),
            batched=True,
        )
        run_time = sw.get_time_ms()

    drop_cols = list(set(dataset.column_names) - {"Language"})
    features = features.remove_columns(drop_cols)
    features = features.rename_column("Language", "Target")

    items_per_second = len(features) / (run_time / 1000)
    print(f"items/second: {items_per_second:,.0f}")

    features.to_parquet("features.parquet")
