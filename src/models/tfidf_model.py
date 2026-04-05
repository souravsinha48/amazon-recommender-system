import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRecommender:
    def __init__(self, max_features=20000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range, stop_words="english"
        )
        self.tfidf_matrix = None
        self.product_df = None

    def fit(self, product_df: pd.DataFrame):
        """
        Train TF-IDF model on product data
        """
        self.product_df = product_df.copy()

        # Ensure no NaNs
        self.product_df["combined_text"] = self.product_df["combined_text"].fillna("")

        # Fit vectorizer
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.product_df["combined_text"]
        )

        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend(self, product_title: str, top_n: int = 5):
        """
        Recommend similar products
        """
        if product_title not in self.product_df["title"].values:
            return pd.DataFrame()

        idx = self.product_df[self.product_df["title"] == product_title].index[0]

        similarity_scores = cosine_similarity(
            self.tfidf_matrix[idx], self.tfidf_matrix
        ).flatten()

        top_indices = similarity_scores.argsort()[::-1][1 : top_n + 1]

        results = self.product_df.iloc[top_indices].copy()

        # Ranking score (your logic)
        results["score"] = similarity_scores[top_indices]

        return results.sort_values(by="score", ascending=False)[
            [
                "title",
                "asin",
                "category_text",
                "avg_rating",
                "rating_count",
                "popularity",
                "score",
            ]
        ]
