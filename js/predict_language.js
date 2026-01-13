import fs from "node:fs";
import path from "node:path";

const modelPath = path.resolve("models", "logreg.json");
const model = JSON.parse(fs.readFileSync(modelPath, "utf8"));

const {classes, coef, intercept} = model;

const isWordChar = (ch) => /[A-Za-z0-9_]/.test(ch);

const hasWordIn = (haystack, word) => {
  if (!word) {
    return false;
  }
  let start = 0;
  while (true) {
    const idx = haystack.indexOf(word, start);
    if (idx === -1) {
      return false;
    }
    const beforeOk = idx === 0 || !isWordChar(haystack[idx - 1]);
    const afterIdx = idx + word.length;
    const afterOk = afterIdx === haystack.length || !isWordChar(haystack[afterIdx]);
    if (beforeOk && afterOk) {
      return true;
    }
    start = idx + 1;
  }
};

const hasLineStartingWith = (lines, ch) => {
  for (const line of lines) {
    const stripped = line.replace(/^\s+/, "");
    if (stripped.startsWith(ch)) {
      return true;
    }
  }
  return false;
};

const hasIndentColons = (lines) => {
  for (const line of lines) {
    let stripped = line.replace(/\s+$/, "");
    if (!stripped) {
      continue;
    }
    const hashIdx = stripped.indexOf("#");
    if (hashIdx !== -1) {
      stripped = stripped.slice(0, hashIdx).replace(/\s+$/, "");
    }
    if (stripped.endsWith(":")) {
      return true;
    }
  }
  return false;
};

const hasLeadingIndent = (lines, width) => {
  if (width <= 0) {
    return false;
  }
  const pad = " ".repeat(width);
  for (const line of lines) {
    if (!line.trim()) {
      continue;
    }
    if (line.startsWith(pad)) {
      return true;
    }
  }
  return false;
};

const hasDollarVar = (text) => {
  for (let idx = 0; idx < text.length; idx += 1) {
    if (text[idx] !== "$") {
      continue;
    }
    const next = text[idx + 1];
    if (next && (/[A-Za-z_]/.test(next))) {
      return true;
    }
  }
  return false;
};

const hasPythonFString = (text) => {
  for (let idx = 0; idx < text.length - 1; idx += 1) {
    if (text[idx] !== "f") {
      continue;
    }
    const next = text[idx + 1];
    if (next !== "'" && next !== "\"") {
      continue;
    }
    if (idx === 0 || !isWordChar(text[idx - 1])) {
      return true;
    }
  }
  return false;
};

const softmax = (logits) => {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - maxLogit));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => value / sum);
};

const extractFeatures = (snippet) => {
  const text = snippet ?? "";
  const lower = text.toLowerCase();
  const lines = text.split(/\r\n|\n|\r/);

  const hasWord = (word) => hasWordIn(text, word);
  const hasWordCi = (word) => hasWordIn(lower, word.toLowerCase());
  const hasSymbol = (sym) => text.includes(sym);
  const hasRegex = (re) => re.test(text);

  return {
    HasNoneTC: hasWord("None"),
    HasSelf: hasWord("self"),
    HasTrueTC: hasWord("True"),
    HasFalseTC: hasWord("False"),
    HasLambda: hasWord("lambda"),
    HasIndentColon: hasIndentColons(lines),
    HasTripleQuotes: hasSymbol("\"\"\"") || hasSymbol("'''"),
    HasPythonFString: hasPythonFString(text),
    HasPublic: hasWord("public"),
    HasStatic: hasWord("static"),
    HasVoid: hasWord("void"),
    HasNew: hasWord("new"),
    HasPackage: hasWord("package"),
    HasExtends: hasWord("extends"),
    HasImplements: hasWord("implements"),
    HasAtSymbol: text.includes("@"),
    HasLet: hasWord("let"),
    HasConst: hasWord("const"),
    HasVar: hasWord("var"),
    HasFatArrow: hasSymbol("=>"),
    HasExport: hasWord("export"),
    HasDefault: hasWord("default"),
    HasFunc: hasWord("func"),
    HasColonEquals: hasSymbol(":="),
    HasDefer: hasWord("defer"),
    HasGoKeyword: hasWord("go"),
    HasChan: hasWord("chan"),
    HasLeftArrow: hasSymbol("<-"),
    HasEnd: hasWord("end"),
    HasModule: hasWord("module"),
    HasRequire: hasWord("require"),
    HasDo: hasWord("do"),
    HasPuts: hasWord("puts"),
    HasPhpTag: hasSymbol("<?php"),
    HasDollarVar: hasDollarVar(text),
    HasNamespace: hasWord("namespace"),
    HasUse: hasWord("use"),
    HasEcho: hasWord("echo"),
    HasThinArrow: hasSymbol("->"),
    HasDoubleColon: hasSymbol("::"),
    HasAsync: hasWord("async"),
    HasAwait: hasWord("await"),
    HasDef: hasWord("def"),
    HasFunction: hasWord("function"),
    HasImport: hasWord("import"),
    HasFromImport: hasWord("from") && hasWord("import"),
    HasFromImportOrder: hasRegex(/^\s*from\s+\S+\s+import\b/m),
    HasImportFromOrder: hasRegex(/^\s*import\b.+\bfrom\s+['"]/m),
    HasImportAs: hasRegex(/^\s*(from\s+\S+\s+import|import)\s+.+\s+as\s+/m),
    HasImportStarAs: hasRegex(/^\s*import\s+\*\s+as\s+/m),
    HasImportQuoteLine: hasRegex(/^\s*import\s+['"]/m),
    HasClass: hasWord("class"),
    HasSemicolon: hasSymbol(";"),
    HasHashComment: hasSymbol("#"),
    HasDecoratorLine: hasLineStartingWith(lines, "@"),
    HasThisDot: hasSymbol("this."),
    HasThisArrow: hasSymbol("$this->"),
    HasDollarThis: hasSymbol("$this"),
    HasKwargs: hasSymbol("**kwargs"),
    HasArgs: hasSymbol("*args"),
    HasAngleQuestion: hasSymbol("<?"),
    HasAngleBrackets: hasSymbol("<") && hasSymbol(">"),
    HasDollarParen: hasSymbol("($"),
    HasDotPlus: hasSymbol(".+"),
    HasPlusEquals: hasSymbol("+="),
    HasAmpDollar: hasSymbol("&$"),
    HasSpaceColonSpace: hasSymbol(" : "),
    HasSlashStarHash: hasSymbol("/*#"),
    HasPrivate: hasWord("private"),
    HasProtected: hasWord("protected"),
    HasBool: hasWord("bool"),
    HasBoolean: hasWord("boolean"),
    HasStr: hasWord("str"),
    HasString: hasWord("string"),
    HasStringTC: hasWord("String"),
    HasStdString: hasWord("std") && hasSymbol("::") && hasWord("string"),
    HasCString: hasWord("cstring"),
    HasChar: hasWord("char"),
    HasByte: hasWord("byte"),
    HasBytes: hasWord("bytes"),
    HasStrlen: hasWord("strlen"),
    HasTabs: text.includes("\t"),
    HasIndentTwoSpaces: hasLeadingIndent(lines, 2),
    HasIndentFourSpaces: hasLeadingIndent(lines, 4),
    HasDefLineColon: hasRegex(/^\s*def\s+\w+\s*\(.*\)\s*:/m),
    HasGenericWildcard: hasRegex(/<\?\s*(extends|super)\b/m),
    HasDiffHeader: hasLineStartingWith(lines, "diff --git"),
    HasDiffHunk: hasLineStartingWith(lines, "@@"),
    HasDiffFilePlus: hasLineStartingWith(lines, "+++"),
    HasDiffFileMinus: hasLineStartingWith(lines, "---"),
    HasMakeTarget: hasRegex(/^[A-Za-z0-9_.-]+:\s/m),
    HasMakeVar: hasSymbol("$("),
    HasPhony: hasWord(".PHONY"),
    HasMakeAssign: hasRegex(/^\s*[A-Za-z0-9_.-]+\s*[:?+]?=/m),
    HasMakeShell: hasSymbol("$(shell "),
    HasMakeRecipeTab: hasRegex(/^\t/m),
    HasIncludeStdIO: hasSymbol("#include <stdio"),
    HasPrintf: hasWord("printf"),
    HasScanf: hasWord("scanf"),
    HasIostream: hasSymbol("#include <iostream"),
    HasStdNamespace: hasSymbol("std::"),
    HasTemplate: hasWord("template"),
    HasRustFnArrow: hasRegex(/\bfn\s+\w+\s*\(.*\)\s*->/m),
    HasRustUse: hasWord("use") && hasSymbol("::"),
    HasRustMacroRules: hasSymbol("macro_rules!"),
    HasRustDerive: hasSymbol("#[derive"),
    HasLetMut: hasWord("let") && hasWord("mut"),
    HasRustMatch: hasWord("match"),
    HasGoPackage: hasWord("package"),
    HasGoImportBlock: hasSymbol("import ("),
    HasCssProperty: hasRegex(/\b(color|display|font|margin|padding|border)\s*:/m),
    HasScssVar: hasRegex(/\$[A-Za-z_-][A-Za-z0-9_-]*\s*:/m),
    HasScssMixin: hasWord("@mixin"),
    HasScssInclude: hasWord("@include"),
    HasScssExtend: hasWord("@extend"),
    HasScssInterp: hasSymbol("#{"),
    HasScssNestedAmp: hasRegex(/&[.:#\[]/m),
    HasLessVar: hasRegex(/@[A-Za-z_-][A-Za-z0-9_-]*\s*:/m),
    HasLessInterp: hasSymbol("@{"),
    HasLessGuard: hasWord("when") && hasSymbol("("),
    HasLessImportOpts: hasSymbol("@import ("),
    HasLessMixinCall: hasRegex(/\.[A-Za-z_-][A-Za-z0-9_-]*\s*\(/m),
    HasLessVarUse: hasRegex(/@[A-Za-z_-][A-Za-z0-9_-]*\b/m),
    HasSqlSelect: hasWordCi("select"),
    HasSqlFrom: hasWordCi("from"),
    HasSqlWhere: hasWordCi("where"),
    HasSqlCreateTable: hasWordCi("create") && hasWordCi("table"),
    HasSqlInsertInto: hasWordCi("insert") && hasWordCi("into"),
    HasJsonObject: hasSymbol("{") && hasSymbol("}") && hasRegex(/"[^"]+"\s*:/m),
    HasIniSection: hasRegex(/^\s*\[[^\]]+\]\s*$/m),
    HasIniKeyValue: hasRegex(/^\s*[A-Za-z0-9_.-]+\s*=\s*.+$/m),
    HasIniComment: hasRegex(/^\s*[#;]/m),
    HasYamlList: hasLineStartingWith(lines, "- "),
    HasYamlDocMarker: hasLineStartingWith(lines, "---") || hasLineStartingWith(lines, "..."),
    HasYamlKeyValue: hasRegex(/^\s*[A-Za-z0-9_.-]+\s*:\s+.+$/m),
    HasMarkdownHeading: hasLineStartingWith(lines, "#"),
    HasMarkdownList: hasRegex(/^\s*[-*+]\s+/m),
    HasMarkdownOrdered: hasRegex(/^\s*\d+\.\s+/m),
    HasMarkdownQuote: hasRegex(/^\s*>/m),
    HasMarkdownUnderlineHeading: hasRegex(/^\s*={2,}\s*$/m)
        || hasRegex(/^\s*-{2,}\s*$/m),
    HasMarkdownFence: hasSymbol("```"),
    HasMarkdownLink: hasSymbol("]("),
    HasXmlDecl: hasSymbol("<?xml"),
    HasXmlNs: hasSymbol("xmlns"),
    HasGraphQlQuery: hasWordCi("query"),
    HasGraphQlMutation: hasWordCi("mutation"),
    HasGraphQlFragment: hasWordCi("fragment"),
    HasGraphQlSpread: hasSymbol("... on"),
    HasTsInterface: hasWord("interface"),
    HasTsTypeAlias: hasWord("type"),
    HasTsImplements: hasWord("implements"),
    HasTsEnum: hasWord("enum"),
    HasTsAs: hasWord("as"),
    HasTsReadonly: hasWord("readonly"),
    HasTsAccessMod: hasWord("public") || hasWord("private") || hasWord("protected"),
    HasShebang: hasLineStartingWith(lines, "#!"),
    HasShellExport: hasWord("export"),
    HasShellIf: hasSymbol("if [") || hasSymbol("if test"),
    HasShellFi: hasWord("fi"),
    HasShellThen: hasWord("then"),
    HasShellDo: hasWord("do"),
    HasShellDone: hasWord("done"),
    HasShellVar: hasRegex(/\$[A-Za-z_][A-Za-z0-9_]*/m),
    HasPerlStrict: hasSymbol("use strict"),
    HasPerlWarnings: hasSymbol("use warnings"),
    HasPerlMySigil: hasSymbol("my $"),
    HasPerlArgv: hasSymbol("@ARGV"),
    HasPerlRegexSub: hasRegex(/\bs\/.+\/.*?\//m),
    HasLuaLocal: hasWord("local"),
    HasLuaThen: hasWord("then"),
    HasLuaElseIf: hasWord("elseif"),
    HasLuaConcat: hasSymbol(".."),
    HasRAssign: hasRegex(/\s<-\s/m),
    HasRLibrary: hasWord("library"),
    HasRDataFrame: hasSymbol("data.frame("),
    HasRGgplot: hasSymbol("ggplot("),
    HasRSetSeed: hasSymbol("set.seed("),
    HasRTrueFalse: hasWord("TRUE") || hasWord("FALSE"),
    HasMainFunc: hasRegex(/\bint\s+main\s*\(/m),
    HasReturnZero: hasRegex(/\breturn\s+0\s*;/m),
    HasVbImports: hasWord("Imports"),
    HasVbModule: hasWord("Module"),
    HasVbSub: hasWord("Sub"),
    HasVbEndSub: hasSymbol("End Sub"),
    HasVbDim: hasWord("Dim"),
    HasVbAs: hasWord("As"),
  };
};

const logitsFor = (x) => {
  const output = new Array(classes.length).fill(0);
  for (let i = 0; i < classes.length; i += 1) {
    let sum = intercept[i] ?? 0;
    const weights = coef[i];
    for (let j = 0; j < x.length; j += 1) {
      sum += weights[j] * x[j];
    }
    output[i] = sum;
  }
  return output;
};

export const predict_language = (snippet) => {
  const featureMap = extractFeatures(snippet);
  const x = Object.values(featureMap).map((value) => (value ? 1 : 0));
  const logits = logitsFor(x);
  const probs = softmax(logits);
  let bestIdx = 0;
  for (let i = 1; i < probs.length; i += 1) {
    if (probs[i] > probs[bestIdx]) {
      bestIdx = i;
    }
  }
  return classes[bestIdx];
};

export const predict_language_with_confidence = (snippet) => {
  const featureMap = extractFeatures(snippet);
  const x = Object.values(featureMap).map((value) => (value ? 1 : 0));
  const logits = logitsFor(x);
  const probs = softmax(logits);
  let bestIdx = 0;
  for (let i = 1; i < probs.length; i += 1) {
    if (probs[i] > probs[bestIdx]) {
      bestIdx = i;
    }
  }
  return {language: classes[bestIdx], confidence: probs[bestIdx]};
};
