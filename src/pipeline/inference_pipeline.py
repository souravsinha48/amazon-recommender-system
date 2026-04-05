(
    """
Inference Pipeline
Handles model loading and recommendation generation
"""
    """
Inference Pipeline
Handles model loading and recommendation generation
"""
)

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from src.models.hybrid_model import HybridRecommender
from src.models.tfidf_model import TFIDFRecommender
from src.models.bert_model import BERTRecommender


class InferencePipeline:
    def __init__(self, tfidf_model, bert_model, product_df):
        """
        Initialize all models ONCE
        """
        self.product_df = product_df

        self.tfidf = TFIDFRecommender(tfidf_model, product_df)
        self.bert = BERTRecommender(bert_model, product_df)
        self.hybrid = HybridRecommender(tfidf_model, bert_model, product_df)

    # ================================
    # INDEX-BASED (for evaluation)
    # ================================

    def recommend_by_index(self, idx, model="hybrid", top_n=5):
        """
        Used in evaluation pipeline
        """

        if model == "tfidf":
            return self.tfidf.recommend_by_index(idx, top_n)

        elif model == "bert":
            return self.bert.recommend_by_index(idx, top_n)

        elif model == "hybrid":
            return self.hybrid.recommend_by_index(idx, top_n)

        else:
            raise ValueError("Invalid model name")

    # ================================
    # TITLE-BASED (for Gradio/UI)
    # ================================

    def recommend_by_title(self, title, model="hybrid", top_n=5):
        """
        Used in UI (Gradio)
        """

        if title not in self.product_df["title"].values:
            return None

        idx = self.product_df[self.product_df["title"] == title].index[0]

        return self.recommend_by_index(idx, model, top_n)
