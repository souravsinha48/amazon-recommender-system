import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommender:
    def __init__(self, tfidf_model, bert_model, product_df):
        """
        Hybrid recommender combining TF-IDF + BERT + business signals

        Args:
            tfidf_model: trained TFIDFRecommender
            bert_model: trained BERTRecommender
            product_df: dataframe with product info
        """
        self.tfidf_model = tfidf_model
        self.bert_model = bert_model
        self.product_df = product_df

    # -----------------------------
    # Utility: Safe normalization
    # -----------------------------
    def recommend(self, product_title, top_n=5):

        if product_title not in self.product_df["title"].values:
            return None

        idx = self.product_df[self.product_df["title"] == product_title].index[0]

        # --- TF-IDF ---
        tfidf_query = self.tfidf_model.tfidf_matrix[idx]
        tfidf_scores = cosine_similarity(
            tfidf_query, self.tfidf_model.tfidf_matrix
        ).flatten()

        # --- BERT ---
        bert_query = self.bert_model.embeddings[idx].reshape(1, -1)
        bert_scores = cosine_similarity(
            bert_query, self.bert_model.embeddings
        ).flatten()

        # --- Normalize ---
        tfidf_scores = self._normalize(tfidf_scores)
        bert_scores = self._normalize(bert_scores)

        # --- Combine (from tuning) ---
        w_tfidf = 0.5
        w_bert = 0.5

        combined_similarity = w_bert * bert_scores + w_tfidf * tfidf_scores

        # 🔥 IMPORTANT: take larger candidate pool
        top_k_initial = 50
        top_indices = combined_similarity.argsort()[::-1][1 : top_k_initial + 1]

        results = self.product_df.iloc[top_indices].copy()

        # --- Business signals (OPTIONAL layer) ---
        ratings = self._normalize(results["avg_rating"].fillna(0))
        popularity = self._normalize(results["popularity"].fillna(0))  # NO log

        # --- Final score ---
        results["score"] = (
            0.8 * combined_similarity[top_indices]  # ↑ more weight to similarity
            + 0.15 * ratings
            + 0.05 * popularity
        )

        # Final ranking
        results = results.sort_values(by="score", ascending=False).head(top_n)

        return results[
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
