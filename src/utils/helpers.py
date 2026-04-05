import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def validate_title(product_df, title):
    if title not in product_df["title"].values:
        raise ValueError(f"{title} not found in dataset")
