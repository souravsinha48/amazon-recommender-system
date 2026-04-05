import gradio as gr
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import pickle

# =========================
# LOAD DATA
# =========================
product_df = pd.read_csv("data/processed/product_df.csv")

# Ensure clean text
product_df["combined_text"] = product_df["combined_text"].fillna("")

# =========================
# TF-IDF (LOAD OR BUILD)
# =========================
try:
    with open("artifacts/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    print("✅ Loaded TF-IDF matrix")
except:
    print("⚡ Building TF-IDF matrix...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(product_df["combined_text"])
    with open("artifacts/tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)

# Precompute similarity
tfidf_sim = cosine_similarity(tfidf_matrix)


# =========================
# BERT (LOAD OR BUILD)
# =========================
try:
    bert_embeddings = np.load("artifacts/bert_embeddings.npy")
    print("✅ Loaded BERT embeddings")
except:
    print("⚡ Building BERT embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    bert_embeddings = model.encode(
        product_df["combined_text"].tolist(), show_progress_bar=True
    )
    np.save("artifacts/bert_embeddings.npy", bert_embeddings)

bert_sim = cosine_similarity(bert_embeddings)


# =========================
# RECOMMENDATION FUNCTIONS
# =========================
def recommend_tfidf(idx, k=5):
    scores = tfidf_sim[idx]
    return np.argsort(scores)[::-1][1 : k + 1]


def recommend_bert(idx, k=5):
    scores = bert_sim[idx]
    return np.argsort(scores)[::-1][1 : k + 1]


def recommend_hybrid(idx, k=5, w_tfidf=0.5, w_bert=0.5):
    scores = w_tfidf * tfidf_sim[idx] + w_bert * bert_sim[idx]
    return np.argsort(scores)[::-1][1 : k + 1]


# =========================
# SEARCH FUNCTION
# =========================
def get_recommendations(query, model, k):

    matches = product_df[product_df["title"].str.contains(query, case=False, na=False)]

    if len(matches) == 0:
        return "❌ No matching product found"

    idx = matches.index[0]

    if model == "TF-IDF":
        recs = recommend_tfidf(idx, k)
    elif model == "BERT":
        recs = recommend_bert(idx, k)
    else:
        recs = recommend_hybrid(idx, k)

    output = ""

    for i in recs:
        row = product_df.loc[i]

        output += f"""
### 📦 {row["title"]}

⭐ Rating: {row.get("avg_rating", "N/A")}  
🔥 Reviews: {row.get("review_count", "N/A")}

---
"""

    return output


# =========================
# CSS (YOUR PALETTE)
# =========================
custom_css = """
body {
    background-color: #F7F4F2;
    font-family: 'Segoe UI', sans-serif;
    color: #2C2C2C;
}

h1 {
    color: #5B2C6F;
    text-align: center;
    font-weight: 700;
}

h3 {
    color: #A9324F;
    text-align: center;
}

textarea, input {
    background-color: #FFFFFF !important;
    color: #2C2C2C !important;
    border: 1px solid #E59866 !important;
    border-radius: 8px !important;
}

button {
    background-color: #5B2C6F !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600;
}

button:hover {
    background-color: #A9324F !important;
}

label {
    color: #5B2C6F !important;
    font-weight: 600;
}
"""

# =========================
# UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("""
    # 🛒 Amazon Recommender System
    
    ### <span style='color:#5B2C6F'>TF-IDF</span> | 
    <span style='color:#A9324F'>BERT</span> | 
    <span style='color:#E59866'>Hybrid</span>
    """)

    query = gr.Textbox(label="🔍 Search Product", placeholder="e.g., washing machine")

    model = gr.Radio(
        ["TF-IDF", "BERT", "Hybrid"], label="🧠 Select Model", value="Hybrid"
    )

    k = gr.Slider(1, 10, value=5, step=1, label="📊 Number of Recommendations")

    btn = gr.Button("🚀 Get Recommendations")

    output = gr.Markdown()

    btn.click(get_recommendations, inputs=[query, model, k], outputs=output)


# =========================
# LAUNCH
# =========================
demo.launch(css=custom_css, theme=gr.themes.Soft())
