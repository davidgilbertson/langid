import pandas as pd

from analyze_model import (
    generate_f1_curve,
    find_ideal_size,
    generate_rounding_curve,
    get_gzipped_size_kb,
)
from train_model import train_model


if __name__ == "__main__":
    df = pd.read_parquet("E:/Datasets/the_stack_10_line_snippets.parquet")
    # df = pd.read_parquet("E:/Datasets/the_stack_20_line_snippets.parquet")
    # df = pd.read_parquet("E:/Datasets/the_stack_whole_files.parquet")
    # df = df[df.Language.isin(["Go", "Java", "JavaScript", "PHP", "Python", "Ruby"])]

    results = train_model(df)

    f1_df = generate_f1_curve(
        model=results.model,
        X=results.X_val,
        y=results.y_val,
    )

    round_df = generate_rounding_curve(
        model=results.model,
        X=results.X_val,
        y=results.y_val,
    )

    n_features, f1 = find_ideal_size(
        model=results.model,
        X=results.X_val,
        y=results.y_val,
        f1_delta=0.0,
    )

    size_kb = get_gzipped_size_kb(
        model_json_path=results.model_json_file,
        n_features=n_features,
    )

    print(f"F1={f1:.1%} with {n_features} features. Size={size_kb:,.1f} KB (gzipped).")
