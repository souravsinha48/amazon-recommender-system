# Amazon Product Recommender System

This project implements a scalable and modular hybrid recommendation system for Amazon products using a combination of TF-IDF (lexical similarity) and BERT embeddings (semantic similarity). The system is designed as an end-to-end machine learning pipeline with clear separation between data processing, model building, and inference.

---

## Overview

The goal of this project is to recommend similar products based on product metadata (such as title, description, etc.). Traditional keyword-based methods often fail to capture semantic meaning, while deep learning methods can be computationally expensive.  

To address this, we use a hybrid approach:
- TF-IDF → fast, interpretable, keyword-based similarity  
- BERT → captures semantic/contextual similarity  
- Hybrid → combines both to improve ranking quality  

---

## Key Features

- Modular architecture with separate components for preprocessing, modeling, and pipelines  
- Supports both index-based and title-based recommendations  
- Hybrid recommendation strategy combining multiple similarity signals  
- Efficient similarity computation using vectorized operations  
- Designed for scalability (large datasets handled outside repo)  
- Reproducible pipeline for training and inference  

---

## Project Structure

amazon-recommender-system/  
├── src/  
│   ├── models/  
│   │   ├── tfidf_model.py        # TF-IDF similarity logic  
│   │   ├── bert_model.py         # BERT embedding-based similarity  
│   │   └── hybrid_model.py       # Combines TF-IDF + BERT scores  
│   │  
│   ├── pipeline/  
│   │   ├── train_pipeline.py     # Builds models and artifacts  
│   │   └── inference_pipeline.py # Loads models and generates recommendations  
│   │  
│   ├── preprocessing/            # Data cleaning and feature engineering  
│   ├── evaluation/               # Evaluation logic (ranking quality)  
│   └── utils/                    # Helper functions  
│  
├── notebooks/                    # EDA and experimentation  
├── configs/                      # Configuration files  
├── data/                         # (ignored) large datasets  
├── artifacts/                    # (ignored) trained models  
├── requirements.txt  
├── .gitignore  
└── README.md  

---

## Dataset Instructions

Download the required dataset from:

Reviews:
https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Appliances.json.gz

Metadata:
https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Appliances.json.gz

Place files in:
data/raw/

---

## How the System Works

### 1. Preprocessing
- Raw product data is cleaned and standardized  
- Text fields (title, description) are processed  
- Final dataset (`product_df`) is created for modeling  

---

### 2. TF-IDF Model (`tfidf_model.py`)
- Converts product text into sparse TF-IDF vectors  
- Computes cosine similarity between products  
- Fast and effective for keyword-based matching  

---

### 3. BERT Model (`bert_model.py`)
- Uses pretrained BERT to generate dense embeddings  
- Captures semantic similarity between products  
- Handles contextual meaning better than TF-IDF  

---

### 4. Hybrid Model (`hybrid_model.py`)
- Combines TF-IDF and BERT similarity scores  
- Typically uses weighted averaging or ranking fusion  
- Improves recommendation quality by balancing:
  - precision (TF-IDF)
  - semantic understanding (BERT)

---

### 5. Training Pipeline (`train_pipeline.py`)
- Runs full pipeline:
  - preprocessing
  - TF-IDF fitting
  - BERT embedding generation  
- Saves artifacts:
  - TF-IDF model
  - embedding matrices
  - processed dataframe  

---

### 6. Inference Pipeline (`inference_pipeline.py`)
- Loads all models once (efficient design)  
- Provides unified interface:
  - `recommend_by_index()`  
  - `recommend_by_title()`  
- Routes request to:
  - TF-IDF / BERT / Hybrid model  

---

## Setup

1. Clone the repository  
git clone https://github.com/souravsinha48/amazon-recommender-system.git  
cd amazon-recommender-system  

2. Create virtual environment  
python -m venv venv  
source venv/bin/activate  

3. Install dependencies  
pip install -r requirements.txt  

---

## Data & Artifacts

The dataset (~24GB) and trained artifacts (~7GB) are not included due to GitHub size limits.

To regenerate everything:

python -m src.pipeline.train_pipeline  

This will create required artifacts inside the `artifacts/` directory.

---

## Usage

### Recommend by index
pipeline.recommend_by_index(idx=10, model="hybrid", top_n=5)  

### Recommend by title
pipeline.recommend_by_title("Sample Product", model="hybrid", top_n=5)  

Supported models:
- "tfidf"
- "bert"
- "hybrid"

---

## Evaluation Approach

- Index-based evaluation pipeline  
- Compare recommendations across models  
- Focus on ranking quality and relevance  

---

## Design Decisions & Trade-offs

- TF-IDF chosen for speed and interpretability  
- BERT chosen for semantic richness  
- Hybrid model balances performance and computation  

- Large artifacts excluded from repo to:
  - keep repository lightweight  
  - ensure faster cloning  
  - enforce reproducibility via pipeline  

---

## Limitations

- No collaborative filtering (user behavior not used)  
- BERT inference can be computationally expensive  
- Evaluation is primarily offline  

---

## Future Improvements

- Add collaborative filtering / user-based recommendations  
- Use FAISS or ANN for faster similarity search  
- Deploy as API (FastAPI/Flask)  
- Add real-time recommendation capability  
- Incorporate user feedback loop  

---

## Collaborators

- Sourav Sinha  
- Kothawade Sumedh  
- Jishu Dohare  
- Chetti Sai Srikar  
- Anurag Kumar  

---

## Notes

- Large files are excluded using `.gitignore`  
- Artifacts are generated via pipeline for reproducibility  
- Project follows modular and scalable ML design principles  
