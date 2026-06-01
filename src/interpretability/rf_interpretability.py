"""
Random Forest Interpretability Module for the Parkinson's Drug Discovery Framework.

This module provides a suite of global and local explanation tools to demystify
how the Random Forest classifier predicts drug-target interactions. By extracting
feature importance at multiple levels, it allows researchers to verify that the
model is learning true biological signals rather than memorizing dataset artifacts.

Key functionality:
  - Global Impurity Importance: Fast extraction of built-in tree split metrics.
  - Modality Grouping: Aggregates importance across multimodal embeddings
    (drug, target, phenotype) to identify which data source drives predictions.
  - Permutation Importance: A rigorous, shuffle-based evaluation that calculates
    actual performance drop, mitigating the bias impurity metrics have toward
    high-cardinality features.
  - Local Ablation Explanation: Breaks down a single predicted drug-target pair
    to see exactly which features pushed its specific probability up or down.

Output:
  - Structured lists and dictionaries containing ranked feature names, importance
    scores, mean accuracy drops, and probability deltas.

Dependencies:
  - numpy: For fast array manipulation, random permutation, and metric calculations.
"""

from __future__ import annotations

import numpy as np


def _safe_feature_name(feature_names, i):
    """
    Helper to gracefully fall back to generic names (e.g., 'feature_42')
    if a list of human-readable feature names isn't provided or is too short.
    """
    if feature_names is None:
        return f"feature_{i}"
    return feature_names[i] if i < len(feature_names) else f"feature_{i}"


def feature_importance_from_rf(model, feature_names=None, top_k=20, normalize=True):
    """
    Global feature importance from RandomForest built-in impurity importances.

    Why use it: It is computationally free since the RF calculates this during training.
    Caveat: "Gini impurity" importance is famously biased toward continuous features
    or features with many unique values. It should be used as a quick baseline.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not expose feature_importances_.")

    importances = np.asarray(model.feature_importances_, dtype=float)
    if normalize and importances.sum() > 0:
        importances = importances / importances.sum()

    # Sort descending and slice the top_k
    order = np.argsort(importances)[::-1][:top_k]
    return [
        {
            "rank": int(rank + 1),
            "feature": _safe_feature_name(feature_names, i),
            "importance": float(importances[i]),
        }
        for rank, i in enumerate(order)
    ]


def grouped_feature_importance(model, feature_names):
    """
    Aggregate RF importance by modality prefix.

    Why use it: In multimodal biological frameworks, individual embedding features
    (e.g., drug_emb_12) mean nothing on their own. This groups them to answer high-level
    questions like: "Is the model relying more on the drug's molecular structure
    or the target's genetic phenotype to make this prediction?"
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not expose feature_importances_.")

    importances = np.asarray(model.feature_importances_, dtype=float)
    groups = {"drug": 0.0, "target": 0.0, "phenotype": 0.0, "other": 0.0}

    for i, imp in enumerate(importances):
        name = _safe_feature_name(feature_names, i)
        if name.startswith("drug_emb_"):
            groups["drug"] += float(imp)
        elif name.startswith("target_emb_"):
            groups["target"] += float(imp)
        elif name.startswith("pheno_emb_"):
            groups["phenotype"] += float(imp)
        else:
            groups["other"] += float(imp)

    # Normalize groups so they sum to 1.0 (100%)
    total = sum(groups.values())
    if total > 0:
        for k in groups:
            groups[k] /= total
    return groups


def permutation_importance_rf(
    model, X, y, feature_names=None, metric="accuracy", n_repeats=5, random_state=42
):
    """
    Model-agnostic importance: how much performance drops when a feature is shuffled.

    Why use it: This is the "honest" importance metric. By randomly shuffling a single
    feature column and seeing how much the model's accuracy degrades, we bypass the
    biases of impurity metrics. If shuffling a feature doesn't hurt accuracy, it isn't truly important.
    """
    rng = np.random.RandomState(random_state)

    # Calculate baseline performance before any shuffling
    if metric == "accuracy":
        base = np.mean(model.predict(X) == y)

        def score_fn(X_eval):
            return np.mean(model.predict(X_eval) == y)
    else:
        raise ValueError("Currently supported metric: 'accuracy'")

    rows = []
    # Iterate through every single feature column
    for j in range(X.shape[1]):
        drops = []
        # Repeat the shuffle multiple times to get a stable mean and standard deviation
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = X_perm[rng.permutation(X_perm.shape[0]), j]
            drops.append(base - score_fn(X_perm))

        rows.append({
            "feature": _safe_feature_name(feature_names, j),
            "importance_mean_drop": float(np.mean(drops)),
            "importance_std_drop": float(np.std(drops)),
        })

    # Sort so the features that caused the biggest accuracy drop are at the top
    rows.sort(key=lambda r: r["importance_mean_drop"], reverse=True)
    return {"baseline_score": float(base), "rows": rows}


def explain_single_prediction_rf(
    model, x_row, feature_names=None, baseline_row=None, class_index=1, top_k=15
):
    """
    Local explanation via one-feature-at-a-time ablation.

    Why use it: Global metrics tell you how the model works in general. This local metric
    tells you why the model predicted a *specific* drug-target pair. It sets each feature
    to a baseline (usually zero) one by one, measuring how the probability of class 1
    (positive interaction) changes.
    """
    x = np.asarray(x_row, dtype=float).reshape(1, -1)

    # If no baseline is provided, assume 0 is the neutral state
    baseline = (
        np.zeros_like(x)
        if baseline_row is None
        else np.asarray(baseline_row, dtype=float).reshape(1, -1)
    )

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba for local explanation.")

    # The original probability prediction for this specific pair
    p0 = float(model.predict_proba(x)[0, class_index])
    contributions = []

    # Ablation loop: replace one feature with baseline, predict, record difference
    for j in range(x.shape[1]):
        x_mod = x.copy()
        x_mod[0, j] = baseline[0, j]
        p_mod = float(model.predict_proba(x_mod)[0, class_index])
        contributions.append({
            "feature": _safe_feature_name(feature_names, j),
            # A positive delta means this feature pushed the probability UP
            # A negative delta means this feature pushed the probability DOWN
            "delta_proba": p0 - p_mod,
        })

    # Sort by absolute magnitude to find features that had the biggest impact (positive or negative)
    contributions.sort(key=lambda d: abs(d["delta_proba"]), reverse=True)

    return {
        "predicted_probability": p0,
        "top_feature_contributions": contributions[:top_k],
    }