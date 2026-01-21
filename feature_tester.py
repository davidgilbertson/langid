from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd


def stats_for_rule(
    df: pd.DataFrame,
    test_func: Callable,
    print_results=False,
):
    df = df.to_dict("records")

    results = defaultdict(list)
    for row in df:
        snippet = row.get("Snippet")
        language = row.get("Language")
        results[language].append(test_func(snippet))

    data = []
    for lang, bools in results.items():
        data.append(
            dict(
                Language=lang,
                NMatches=sum(bools),
                NSamples=len(bools),
                Occurrence=np.mean(bools),
            )
        )

    df = pd.DataFrame(data)

    if print_results:
        res_df = df.sort_values("Occurrence", ascending=False)
        res_df = res_df[res_df.Occurrence > 0]
        for i, row in res_df.iterrows():
            print(f"{row.Language}: {row.Occurrence:.2%}")
    return df


if __name__ == "__main__":
    df = pd.read_parquet("E:/Datasets/the_stack_10_line_snippets.parquet")
    df = df.groupby("Language").sample(frac=0.5)

    test_feature_names = [
        "std::",
        "template<",
        "namespace ",
        "using namespace",
        "cout",
        "cin",
        "class ",
        "new ",
        "delete",
        "nullptr",
        "operator",
        "#include <iostream>",
        "#include <vector>",
        "#include <string>",
        "#include <stdio.h>",
        "printf(",
        "scanf(",
        "malloc(",
        "free(",
        "typedef struct",
        "struct ",
    ]
    for test_feature_name in test_feature_names:
        print(f"\nFeature: {test_feature_name!r}")
        df_results = stats_for_rule(
            df=df,
            test_func=lambda snippet, feature_name=test_feature_name: feature_name
            in snippet,
            print_results=True,
        )
