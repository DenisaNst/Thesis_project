"""
rf_interpretability.py
-----------------------
Interpretability tools for the Random Forest classifier.

Functions
---------
feature_importance_from_rf      — global impurity-based importance
grouped_feature_importance      — importance aggregated by modality
permutation_importance_rf       — shuffle-based importance (more honest)
explain_single_prediction_rf    — local ablation explanation for one sample
"""

from __future__ import annotations

import numpy as np


def _safe_feature_name(feature_names, i):
    if feature_names is None:
        return f"feature_{i}"
    return feature_names[i] if i < len(feature_names) else f"feature_{i}"


def feature_importance_from_rf(model, feature_names=None, top_k=20, normalize=True):
    """
    Global feature importance from RandomForest built-in impurity importances.
    Returns top-k features sorted descending.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not expose feature_importances_.")

    importances = np.asarray(model.feature_importances_, dtype=float)
    if normalize and importances.sum() > 0:
        importances = importances / importances.sum()

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
    Aggregate RF importance by modality prefix:
    - drug_emb_
    - target_emb_
    - pheno_emb_
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

    total = sum(groups.values())
    if total > 0:
        for k in groups:
            groups[k] /= total
    return groups


def permutation_importance_rf(
    model, X, y, feature_names=None, metric="accuracy", n_repeats=5, random_state=42
):
    """
    Model-agnostic importance: how much performance drops when a feature is
    shuffled. More reliable than impurity importance when features are correlated.
    """
    rng = np.random.RandomState(random_state)

    if metric == "accuracy":
        base = np.mean(model.predict(X) == y)

        def score_fn(X_eval):
            return np.mean(model.predict(X_eval) == y)
    else:
        raise ValueError("Currently supported metric: 'accuracy'")

    rows = []
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = X_perm[rng.permutation(X_perm.shape[0]), j]
            drops.append(base - score_fn(X_perm))

        rows.append({
            "feature": _safe_feature_name(feature_names, j),
            "importance_mean_drop": float(np.mean(drops)),
            "importance_std_drop": float(np.std(drops)),
        })

    rows.sort(key=lambda r: r["importance_mean_drop"], reverse=True)
    return {"baseline_score": float(base), "rows": rows}


def explain_single_prediction_rf(
    model, x_row, feature_names=None, baseline_row=None, class_index=1, top_k=15
):
    """
    Local explanation via one-feature-at-a-time ablation.
    Returns the top-k features whose removal most changes the predicted probability.
    """
    x = np.asarray(x_row, dtype=float).reshape(1, -1)
    baseline = (
        np.zeros_like(x)
        if baseline_row is None
        else np.asarray(baseline_row, dtype=float).reshape(1, -1)
    )

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba for local explanation.")

    p0 = float(model.predict_proba(x)[0, class_index])
    contributions = []

    for j in range(x.shape[1]):
        x_mod = x.copy()
        x_mod[0, j] = baseline[0, j]
        p_mod = float(model.predict_proba(x_mod)[0, class_index])
        contributions.append({
            "feature": _safe_feature_name(feature_names, j),
            "delta_proba": p0 - p_mod,
        })

    contributions.sort(key=lambda d: abs(d["delta_proba"]), reverse=True)
    return {
        "predicted_probability": p0,
        "top_feature_contributions": contributions[:top_k],
    }