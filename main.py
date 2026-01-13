from data.utils import get_stack_data
from analyze_model import generate_f1_curve, find_ideal_size
from features import generate_features
from train_model import train_model


if __name__ == "__main__":
    # Easy mode: 6 languages and snippets are full files
    # df = get_stack_data(languages=["Go", "Java", "JavaScript", "PHP", "Python", "Ruby"])

    # Hard mode: all languages and 10-line snippets
    df = get_stack_data(snippet_limit=10)
    features = generate_features(df=df, save=False)
    results = train_model(features=features)

    f1_df = generate_f1_curve(df=df)

    n_features, f1 = find_ideal_size(df=df.sample(frac=0.5))
