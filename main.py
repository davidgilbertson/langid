import pandas as pd

from analyze_model import (
    generate_f1_curve,
    find_ideal_size,
    generate_rounding_curve,
)
from train_model import train_model
from tools import get_gzipped_size_kb


if __name__ == "__main__":
    df = pd.read_parquet("E:/Datasets/the_stack_10_line_snippets.parquet")
    # df = pd.read_parquet("E:/Datasets/the_stack_20_line_snippets.parquet")
    # df = pd.read_parquet("E:/Datasets/the_stack_whole_files.parquet")
    # df = df[df.Language.isin(["Go", "Java", "JavaScript", "PHP", "Python", "Ruby"])]

    results = train_model(df)

    # See how changes in number of features affects F1
    f1_df = generate_f1_curve(
        model=results.model,
        X=results.X_val,
        y=results.y_val,
    )

    # See how different rounding values affects F1 and Size
    round_df = generate_rounding_curve(
        model=results.model,
        X=results.X_val,
        y=results.y_val,
    )

    # How many features are needed to achieve (F1 - 1%)?
    n_features, f1 = find_ideal_size(
        model=results.model,
        X=results.X_val,
        y=results.y_val,
        f1_delta=0.01,
    )

    # How big will the model be over the network
    size_kb = get_gzipped_size_kb(
        model=results.model,
        n_features=n_features,
    )

    print(f"F1={f1:.1%} with {n_features} features. {size_kb:,.1f} KB (gzipped).")
