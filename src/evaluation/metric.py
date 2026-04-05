import numpy as np


# ================================
# CORE: PRECISION@K (MAIN METRIC)
# ================================


def precision_at_k(sample_idx, recommend_fn, clusters, k=5):
    """
    Cluster-based Precision@K

    Relevant = same cluster
    """

    recs = recommend_fn(sample_idx)

    if recs is None or len(recs) == 0:
        return 0

    recommended_indices = recs.index.tolist()[:k]
    input_cluster = clusters[sample_idx]

    relevant = sum(1 for idx in recommended_indices if clusters[idx] == input_cluster)

    return relevant / k


# ================================
# AUXILIARY METRICS (OPTIONAL)
# ================================


def diversity_score(recs):
    """
    Measures category diversity
    """
    if recs is None or len(recs) == 0:
        return 0

    return recs["category_text"].nunique() / len(recs)


def avg_rating_score(recs):
    """
    Measures average quality
    """
    if recs is None or len(recs) == 0:
        return 0

    return recs["avg_rating"].mean()


def novelty_score(recs):
    """
    Penalizes popular items → encourages discovery
    """
    if recs is None or len(recs) == 0:
        return 0

    return (1 / (1 + recs["popularity"])).mean()


# ================================
# SINGLE SAMPLE EVALUATION
# ================================


def evaluate_single(sample_idx, recommend_fn, product_df, clusters, k=5):
    """
    Evaluate one sample
    """

    recs = recommend_fn(sample_idx)

    if recs is None or len(recs) == 0:
        return {
            "precision@k": 0,
            "diversity": 0,
            "avg_rating": 0,
            "novelty": 0,
        }

    return {
        "precision@k": precision_at_k(sample_idx, recommend_fn, clusters, k),
        "diversity": diversity_score(recs),
        "avg_rating": avg_rating_score(recs),
        "novelty": novelty_score(recs),
    }


# ================================
# FULL MODEL EVALUATION
# ================================


def evaluate_model(model_name, recommend_fn, product_df, clusters, n_samples=50, k=5):
    """
    Evaluate model over multiple samples
    """

    sample_indices = np.random.choice(len(product_df), n_samples, replace=False)

    results = {
        "precision@k": [],
        "diversity": [],
        "avg_rating": [],
        "novelty": [],
    }

    for idx in sample_indices:
        try:
            recs = recommend_fn(idx)

            if recs is None or len(recs) == 0:
                continue

            results["precision@k"].append(
                precision_at_k(idx, recommend_fn, clusters, k)
            )

            results["diversity"].append(diversity_score(recs))
            results["avg_rating"].append(avg_rating_score(recs))
            results["novelty"].append(novelty_score(recs))

        except Exception as e:
            print(f"Error at index {idx}: {e}")
            continue

    def safe_mean(x):
        return np.mean(x) if len(x) > 0 else 0

    return {
        "model": model_name,
        "precision@k": safe_mean(results["precision@k"]),
        "diversity": safe_mean(results["diversity"]),
        "avg_rating": safe_mean(results["avg_rating"]),
        "novelty": safe_mean(results["novelty"]),
    }


# ================================
# HYBRID WEIGHT TESTING
# ================================


def evaluate_weights(
    weight_list, hybrid_fn_builder, product_df, clusters, n_samples=30, k=5
):
    """
    Evaluate different (w_tfidf, w_bert) combinations
    """

    results = []

    for w_tfidf, w_bert in weight_list:
        print(f"Testing weights: TF-IDF={w_tfidf}, BERT={w_bert}")

        recommend_fn = hybrid_fn_builder(w_tfidf, w_bert)

        score = evaluate_model(
            f"hybrid_{w_tfidf}_{w_bert}",
            recommend_fn,
            product_df,
            clusters,
            n_samples,
            k,
        )

        results.append((w_tfidf, w_bert, score["precision@k"]))

    return sorted(results, key=lambda x: x[2], reverse=True)
