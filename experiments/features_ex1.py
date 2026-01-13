from typing import Dict

import pandas as pd

from data.utils import get_smola_data
from utils import stopwatch


def extract_features(snippet: str) -> Dict[str, bool]:
    # PascalCase keys make later Pandas attribute access like df.HasConst possible.
    return {
        # Common reserved words / tokens
        "HasConst": "const" in snippet,
        "HasLet": "let" in snippet,
        "HasVar": "var" in snippet,
        "HasDef": "def" in snippet,
        "HasClass": "class" in snippet,
        "HasFunction": "function" in snippet,
        "HasFn": "fn" in snippet,
        "HasStruct": "struct" in snippet,
        "HasModule": "module" in snippet,
        "HasNamespace": "namespace" in snippet,
        "HasInterface": "interface" in snippet,
        "HasImplements": "implements" in snippet,
        "HasExtends": "extends" in snippet,
        "HasImport": "import" in snippet,
        "HasUsing": "using" in snippet,
        "HasInclude": "include" in snippet,
        "HasRequire": "require" in snippet,
        "HasPackage": "package" in snippet,
        "HasPublic": "public" in snippet,
        "HasPrivate": "private" in snippet,
        "HasProtected": "protected" in snippet,
        "HasStatic": "static" in snippet,
        "HasThis": "this" in snippet,
        "HasSelf": "self" in snippet,
        "HasNull": "null" in snippet,
        "HasNil": "nil" in snippet,
        "HasNone": "None" in snippet,
        "HasTrueLower": "true" in snippet,
        "HasFalseLower": "false" in snippet,
        "HasTrueUpper": "True" in snippet,
        "HasFalseUpper": "False" in snippet,
    }


if __name__ == "__main__":
    df = get_smola_data()
    with stopwatch("extract features") as sw:
        features = [extract_features(row.Snippet) for i, row in df.iterrows()]
        run_time = sw.get_time_ms()

    print(f"{len(df) / (run_time / 1000):.1f} items/second")

    feature_df = pd.DataFrame(features)
