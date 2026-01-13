import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data.utils import get_stack_data
from tools import stopwatch


df = pd.read_parquet("features/features.parquet")
# These are ordered by importance (coef size of a prior run)
sorted_features = [';\n', ': ', '$', '	', ';', '":', 'String', '* ', 'self', '#', '</', '>', '&', 'import', '*', 'NA', 'function', 'let', 'include', 'const', 'string', '--', 'this', 'null', 'right', 'then', 'class', 'package', 'return', '- ', 'by', '---', 'sub', '|', 'Set', '_', 'true', 'font', 'var', 'void', 'else', 'position', '?', 'list', 'Object', 'local', 'try', 'set', 'If', 'export', 'print', 'new', 'input', 'Class', 'func', '...', '!', '##', 'any', 'Me', 'color', 'Or', 'Do', 'state', 'get', 'select', 'struct', 'lock', 'my', 'Get', 'fun', 'map', 'With', 'move', 'char', '~', 'Function', 'next', 'offset', 'file', 'val', 'from', 'type', 'public', 'array', 'ref', 'pub', 'index', 'In', 'case', 'New', 'pass', 'On', ']', 'table', 'NULL', 'object', 'auto', 'time', 'call', 'assert', 'some', 'impl', 'go', 'with', 'private', 'Option', 'del', 'base', 'echo', 'query', '$(', 'expect', 'END', 'Call', 'Inf', 'As', 'start', 'background', 'None', 'define', 'late', 'range', 'delete', 'Is', '[', 'bool', 'param', 'Date', 'compl', '@@', 'last', 'register', 'module', 'part', 'For', 'Error', 'int', 'end', 'using', 'false', 'result', 'To', 'Char', 'data', 'require', '/>', 'namespace', 'Not', 'mod', 'out', 'create', 'values', 'margin', 'And', '<-', 'group', 'order', 'Default', '+=', 'for', 'if', 'long', 'event', 'full', 'fi', 'break', 'Event', 'fn', 'default', 'chan', '["', 'nil', 'display', 'number', 'done', 'do', 'die', 'exit', 'open', 'Mod', 'match', 'End', 'extend', 'interface', 'use', 'except', ':=', 'field', 'params', 'cin', 'left', 'switch', 'library', 'static', 'our', 'error', 'Return', 'show', 'template', 'init', 'mut', 'sync', 'each', 'where', 'into', 'global', '```', 'Const', 'Property', 'True', 'def', 'padding', 'when', 'grid', 'Sub', 'Case', 'enum', 'std::', 'or', 'Select', 'in', 'float', 'update', 'short', 'super', 'virtual', '@return', 'not', 'catch', 'info', '](', 'transform', 'join', 'Enum', 'inline', 'function(', 'insert', '+++', 'endif', 'Global', 'actor', 'Next', 'Interface', 'property', 'memory', 'inner', 'limit', 'Lib', 'while', 'final', 'empty', 'internal', 'Module', 'on', 'Public', 'element', 'and', 'False', 'given', 'border', 'continue', 'loop', 'double', 'signed', 'override', 'throw', 'is', 'byte', '<<', 'dyn', 'flex', 'require(', 'as', 'boolean', 'extension', 'SELECT', 'Step', 'Integer', 'eval', 'uint', 'Like', 'Boolean', 'dynamic', 'elif', 'i32', 'begin', 'requires', 'REM', 'warning', 'await', 'shell', 'declare', 'prototype', 'outer', 'async', 'Byte', '#{', 'WHERE', 'protected', 'Private', 'alias', '@{', 'TRUE', '{"', 'schema', 'fragment', '<!--', 'alter', 'Short', 'record', 'crate', 'When', 'Any', 'annotation', 'Each', 'Self', 'Let', 'Single', 'undef', 'goto', '>>', '<?xml', 'drop', 'throws', 'var(', 'hide', 'Long', 'actual', 'fixed', 'constructor', 'extern', '@import', 'checked', 'ensure', 'Partial', 'BEGIN', 'trait', '@class', 'Exit', 'factory', 'unknown', 'until', 'clone', 'finally', 'Static', 'repeat', 'f64', 'unsigned', 'Stop', 'native', 'Null', 'partial', 'protocol', 'i64', 'Dim', 'abstract', 'operator', 'diff --git', 'extends', 'elsif', 'restrict', 'mixin', 'foreach', '@include', 'friend', 'Continue', 'Loop', '@media', 'delegate', 'Decimal', 'data.frame(', 'library(', 'receiver', '?=', 'unless', 'yield', 'FALSE', 'Namespace', 'sizeof', 'cout', 'Double', '#include <iostream>', 'scalar', 'subscript', 'Optional', 'Delegate', '@interface', 'Overrides', '@end', 'asm', 'retry', 'Structure', 'Imports', 'subscription', 'nullptr', 'infer', 'implements', 'ifdef', 'unsafe', 'Then', 'ifndef', '@font-face', 'when (', '<![CDATA[', 'Else', 'decimal', 'undefined', '%:', 'union', 'having', 'never', 'mutation', 'elseif', 'f32', '@keyframes', 'guard', ':root', 'distinct', 'exports.', 'xor', 'static_cast', '@mixin', 'Protected', 'Finally', '?:', 'xmlns', 'mutable', '@property', 'Using', 'external', '@for', 'suspend', 'lateinit', 'Handles', 'Nothing', '@extend', 'lambda', 'wchar_t', '@import (', 'unset', 'esac', 'Erase', '<!DOCTYPE', '@if', '@else', 'Declare', 'explicit', '@charset', '@function', 'instanceof', 'typedef', 'noinline', 'include_once', 'Try', 'module.exports', 'ReadOnly', '@echo', 'ulong', 'constexpr', 'ByVal', 'TypeOf', '$(shell', '#include <vector>', 'NaN', 'defer', '#include <string>', 'WithEvents', 'ElseIf', 'raise', 'decltype', 'each(', 'typename', 'Resume', 'CType', 'calc(', 'typeid', 'Friend', '_Imaginary', 'sealed', 'concept', 'Alias', 'endswitch', 'CStr', 'isset', 'MyBase', 'readonly', 'callable', 'rescue', 'template<', 'static_assert', '@default', 'reified', 'using namespace', 'typeof', 'synchronized', 'ifeq', '@rest', 'GetType', '@selector', 'Shared', 'AddressOf', 'AddHandler', 'CObj', 'fileprivate', '@protocol', 'Catch', 'directive', '@required', 'redo', 'UInteger', 'infix', 'unchecked', '@page', 'Throw', 'ushort', 'volatile', 'ifneq', '@catch', '@finally', '@try', 'While', 'ByRef', 'Implements', '@implementation', 'deferred', 'reinterpret_cast', 'keyof', '@use', 'NotInheritable', 'Inherits', 'AndAlso', '_Bool', 'Xor', 'ULong', 'new file mode', 'anyref', 'IsNot', 'Overloads', '@optional', 'funcref', '@synthesize', 'OrElse', 'Overridable', 'MustInherit', 'Operator', 'RaiseEvent', 'implicit', 'coproc', 'vararg', 'CDbl', 'SByte', 'DirectCast', 'GoTo', 'dynamic_cast', 'CInt', 'CLng', 'ReDim', 'CDate', 'nonisolated', 'not_eq', 'CDec', 'GoSub', 'MustOverride', 'Narrowing', 'fallthrough', 'NA_integer_', 'inout', 'externref', 'transient', 'typeset', 'unalias', 'typealias', 'unexport', '&:extend', 'nonlocal', 'non-sealed', 'noexcept', 'ParamArray', 'RemoveHandler', 'endfor', 'GetXMLNamespace', 'insteadof', 'enddeclare', 'endforeach', 'endef', 'endwhile', 'NotOverridable', 'NA_complex_', 'EndIf', 'MyClass', 'NA_character_', 'NA_real_', 'xor_eq', 'deinit', 'defined?', 'debugger', 'crossinline', '@arguments', '@autoreleasepool', 'or_eq', '@forward', 'CULng', 'CSByte', 'covariant', 'constinit', 'const_cast', 'consteval', 'companion', 'co_await', 'co_return', 'co_yield', 'rename to', 'rename from', 'require_once', '@container', 'CBool', 'CChar', 'CByte', 'permits', 'Shadows', '@each', '@dynamic', 'CUShort', 'CSng', 'CShort', 'CUInt', 'SyncLock', 'deleted file mode', '@synchronized', '@protected', '@public', 'sbyte', '@supports', 'satisfies', 'rethrow', 'char32_t', 'char16_t', 'WriteOnly', 'Widening', 'UShort', 'TryCast', '@while', 'rethrows', 'stackalloc', 'similarity index', 'setparam', 'asserts', 'thread_local', 'tailrec', '.mixin(', 'strictfp', 'bitor', 'bitand', 'atomic_noexcept', 'atomic_commit', 'associatedtype', 'atomic_cancel', 'alignas', '_Complex', 'vpath', 'and_eq', 'alignof', '@namespace', '@layer', '@plugin', '@private', '!:'] # fmt: skip

# Subset of rows
# df = df.groupby("Target", group_keys=False).sample(frac=0.1)

X = df.drop(columns=["Target"])
y = df["Target"]

# Subset of columns
X = X[sorted_features[:400]]

X_trn, X_val, y_trn, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

with stopwatch(f"Training with {len(X.columns)} features and {len(df):,.0f} rows"):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )

    model.fit(X_trn, y_trn)

    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)

    acc = accuracy_score(y_val, preds)
    print(f"Accuracy: {acc:.1%}")


# def top_n_accuracy(y_true, probs, classes, n):
#     top_idx = np.argsort(probs, axis=1)[:, -n:]
#     top_preds = classes[top_idx]
#     return (top_preds == y_true.to_numpy()[:, None]).any(axis=1).mean()
#
#
# top2_acc = top_n_accuracy(y_val, probs, model.classes_, 2)
# top5_acc = top_n_accuracy(y_val, probs, model.classes_, 5)
# top10_acc = top_n_accuracy(y_val, probs, model.classes_, 10)
# print(f"Top-2 accuracy: {top2_acc:.1%}")
# print(f"Top-5 accuracy: {top5_acc:.1%}")
# print(f"Top-10 accuracy: {top10_acc:.1%}")


payload = {
    "model_type": "logistic_regression",
    "classes": model.classes_.tolist(),
    "coef": np.round(model.coef_, 4).tolist(),
    "intercept": np.round(model.intercept_, 4).tolist(),
}

model_path = Path("models") / "logreg.json"
model_path.write_text(json.dumps(payload, indent=2))

# %% - Inspect the wrong answers

snippets = get_stack_data(snippet_limit=10).to_pandas().Snippet
assert len(df) == len(snippets)
result_df = y_val.to_frame()
result_df["Pred"] = preds
result_df["Confidence"] = np.max(probs, axis=1)
result_df["Snippet"] = snippets
wrong_df = result_df[y_val.ne(preds)].copy()

wrong_counts_df = (
    wrong_df.groupby(["Target", "Pred"])
    .size()
    .reset_index(name="Errors")
    .sort_values("Errors", ascending=False)
)

print("Most common mistakes:")
print(wrong_counts_df.head(10))

# print("Examples of failures")
ex_df = wrong_df[wrong_df.Target.eq("SQL")]
# for i, row in ex_df.head(10).iterrows():
#     print(
#         f"********************** {row['Target']} -> {row['Pred']} *************************"
#     )
#     print(row.Snippet)


# %%
importance = np.mean(np.abs(model.coef_), axis=0)
feature_importance_df = (
    pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": importance,
            "Std": np.std(np.abs(model.coef_), axis=0),
            "MeanValue": X.mean(),
            "Occurrences": X.sum(),
        }
    )
    .sort_values("Importance", ascending=False)
    .reset_index()
)

features = feature_importance_df.Feature
