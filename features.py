from pathlib import Path
import hashlib

import pandas as pd

from data.utils import get_stack_data
from tools import stopwatch
from feature_lists import (
    c,
    cpp,
    c_sharp,
    css,
    dart,
    go,
    graphql,
    html_lang,
    ini,
    java,
    javascript,
    json_lang,
    kotlin,
    lua,
    makefile,
    markdown,
    misc,
    objective_c,
    perl,
    php,
    python as python_features,
    r_lang,
    ruby,
    rust,
    shell,
    sql,
    swift,
    typescript,
    visual_basic_dotnet,
    webassembly,
    xml,
    yaml,
)

# These are LLM-generated reserved words per language, with a few human additions.
# They're later filtered to the most 'important' features.
raw_feature_names = (
    misc.FEATURE_NAMES
    + webassembly.FEATURE_NAMES
    + c.FEATURE_NAMES
    + cpp.FEATURE_NAMES
    + c_sharp.FEATURE_NAMES
    + css.FEATURE_NAMES
    + dart.FEATURE_NAMES
    + go.FEATURE_NAMES
    + graphql.FEATURE_NAMES
    + html_lang.FEATURE_NAMES
    + ini.FEATURE_NAMES
    + json_lang.FEATURE_NAMES
    + java.FEATURE_NAMES
    + javascript.FEATURE_NAMES
    + kotlin.FEATURE_NAMES
    + lua.FEATURE_NAMES
    + makefile.FEATURE_NAMES
    + markdown.FEATURE_NAMES
    + objective_c.FEATURE_NAMES
    + php.FEATURE_NAMES
    + perl.FEATURE_NAMES
    + python_features.FEATURE_NAMES
    + r_lang.FEATURE_NAMES
    + ruby.FEATURE_NAMES
    + rust.FEATURE_NAMES
    + sql.FEATURE_NAMES
    + shell.FEATURE_NAMES
    + swift.FEATURE_NAMES
    + typescript.FEATURE_NAMES
    + visual_basic_dotnet.FEATURE_NAMES
    + xml.FEATURE_NAMES
    + yaml.FEATURE_NAMES
)

# dedupe and sort
feature_names = sorted(set(raw_feature_names))


def generate_features(df: pd.DataFrame | None = None, use_cache=True) -> pd.DataFrame:
    df = get_stack_data() if df is None else df
    lang_counts = df.Language.value_counts().sort_index()
    fingerprint = f"rows={len(df)}\nfeatures={'\n'.join(feature_names)}\n{lang_counts.to_string()}"
    data_hash = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12]
    feature_file = Path(f"features/features_{data_hash}.parquet")

    if feature_file.exists() and use_cache:
        print(f"⚡ Using cached features from {feature_file}")
        return pd.read_parquet(feature_file)

    with stopwatch("extracting features"):
        features_matrix = []
        for snippet in df.Snippet:
            snippet = snippet.replace("\r\n", "\n").replace("\r", "\n")
            features = [feature_name in snippet for feature_name in feature_names]
            features_matrix.append(features)

    features_df = pd.DataFrame(features_matrix, columns=feature_names)
    features_df["Target"] = df.Language.to_list()

    features_df.to_parquet(feature_file, index=False)

    return features_df


if __name__ == "__main__":
    df = get_stack_data()
    features = generate_features(
        df=df.head(),
    )
