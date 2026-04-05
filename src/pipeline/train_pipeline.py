"""
Train Pipeline
Handles end-to-end model training
"""

import pandas as pd
from src.models.tfidf_model import TFIDFRecommender
from src.models.bert_model import BERTRecommender


def train_pipeline(data_path):
    print("Loading data...")
    product_df = pd.read_csv(data_path)

    print("Training TF-IDF model...")
    tfidf_model = TFIDFRecommender()
    tfidf_model.fit(product_df)

    print("Loading BERT model...")
    bert_model = BERTRecommender(
        model_path="artifacts/bert_model", product_df=product_df
    )

    print("Training complete!")

    return {"tfidf": tfidf_model, "bert": bert_model}


if __name__ == "__main__":
    models = train_pipeline("data/processed/product_df.csv")
