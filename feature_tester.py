from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd
from data.utils import get_stack_data


def stats_for_rule(
    df: list[dict] | pd.DataFrame,
    test_func: Callable,
    print_results=False,
):
    """
    Dataset must have columns "Snippet" and "Language"
    """
    if isinstance(df, pd.DataFrame):
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
    df = get_stack_data(subset=0.01, snippet_limit=10)
    test_tokens = ["--", ": #"]
    for test_token in test_tokens:
        print(f"\nToken: {test_token!r}")
        df_results = stats_for_rule(
            df=df,
            test_func=lambda snippet, token=test_token: token in snippet,
            print_results=True,
        )
