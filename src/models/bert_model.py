from sentence_transformers import SentenceTransformer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class BERTRecommender:
    def __init__(self, model_path, product_df):
        """
        model_path: path to saved BERT model
        product_df: dataframe with product info
        """
        self.model = SentenceTransformer(model_path)
        self.product_df = product_df

        print("Encoding products (one-time)...")
        self.embeddings = self.model.encode(
            product_df["combined_text"].tolist(), show_progress_bar=True
        )

    def recommend(self, product_title, top_n=5):
        if product_title not in self.product_df["title"].values:
            return None

        idx = self.product_df[self.product_df["title"] == product_title].index[0]

        query_embedding = self.embeddings[idx].reshape(1, -1)

        similarity_scores = cosine_similarity(
            query_embedding, self.embeddings
        ).flatten()

        top_indices = similarity_scores.argsort()[::-1][1 : top_n + 1]

        results = self.product_df.iloc[top_indices].copy()

        # Ranking (same as TF-IDF)
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
