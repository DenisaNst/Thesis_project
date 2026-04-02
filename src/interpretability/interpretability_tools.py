import requests
import numpy as np
import xml.etree.ElementTree as ET

try:
    import torch
except ImportError:
    torch = None


# ----------------------------
# Random Forest Interpretability
# ----------------------------

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
    groups = {
        "drug": 0.0,
        "target": 0.0,
        "phenotype": 0.0,
        "other": 0.0,
    }

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
            groups[k] = groups[k] / total
    return groups


def permutation_importance_rf(model, X, y, feature_names=None, metric="accuracy", n_repeats=5, random_state=42):
    """
    Model-agnostic importance: how much performance drops when a feature is shuffled.
    Works better than impurity importance when features are correlated.
    """
    rng = np.random.RandomState(random_state)

    if metric == "accuracy":
        base = np.mean(model.predict(X) == y)

        def score_fn(X_eval):
            return np.mean(model.predict(X_eval) == y)
    else:
        raise ValueError("Currently supported metric: 'accuracy'")

    n_features = X.shape[1]
    rows = []

    for j in range(n_features):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            perm_idx = rng.permutation(X_perm.shape[0])
            X_perm[:, j] = X_perm[perm_idx, j]
            s = score_fn(X_perm)
            drops.append(base - s)

        rows.append(
            {
                "feature": _safe_feature_name(feature_names, j),
                "importance_mean_drop": float(np.mean(drops)),
                "importance_std_drop": float(np.std(drops)),
            }
        )

    rows = sorted(rows, key=lambda r: r["importance_mean_drop"], reverse=True)
    return {"baseline_score": float(base), "rows": rows}


def explain_single_prediction_rf(model, x_row, feature_names=None, baseline_row=None, class_index=1, top_k=15):
    """
    Local explanation via one-feature-at-a-time ablation:
    - Compute original predicted probability for class_index
    - Replace each feature with baseline value (default 0) and measure probability drop
    Higher drop => stronger positive contribution for this sample.
    """
    x = np.asarray(x_row, dtype=float).reshape(1, -1)
    baseline = np.zeros_like(x) if baseline_row is None else np.asarray(baseline_row, dtype=float).reshape(1, -1)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba for local explanation.")

    p0 = float(model.predict_proba(x)[0, class_index])
    contributions = []

    for j in range(x.shape[1]):
        x_mod = x.copy()
        x_mod[0, j] = baseline[0, j]
        p_mod = float(model.predict_proba(x_mod)[0, class_index])
        contributions.append(
            {
                "feature": _safe_feature_name(feature_names, j),
                "delta_proba": p0 - p_mod,  # positive means feature supports class_index prediction
            }
        )

    contributions = sorted(contributions, key=lambda d: abs(d["delta_proba"]), reverse=True)[:top_k]
    return {
        "predicted_probability": p0,
        "top_feature_contributions": contributions,
    }


# ----------------------------
# GNN Saliency
# ----------------------------

def compute_gnn_saliency(model, x, edge_index, target_class=0, normalize=True):
    """
    Gradient-based node saliency for graph model.
    Returns one saliency score per node.
    """
    if torch is None:
        raise ImportError("PyTorch is required for GNN saliency.")

    x = x.clone().detach().requires_grad_(True)
    batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    output = model(x, edge_index, batch=batch)

    if output.ndim == 1:
        score = output[0]
    else:
        score = output[0, target_class]

    score.backward()
    saliency = x.grad.abs().sum(dim=1).detach().cpu().numpy()

    if normalize and saliency.max() > 0:
        saliency = saliency / saliency.max()
    return saliency


# ----------------------------
# Scientific RAG (PubMed)
# ----------------------------

class ScientificRAG:
    def __init__(self, email=None, api_key=None):
        self.email = email
        self.api_key = api_key

    def _base_params(self):
        params = {"retmode": "json"}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _esearch(self, query, retmax=5):
        params = self._base_params()
        params.update(
            {
                "db": "pubmed",
                "term": query,
                "retmax": retmax,
                "sort": "relevance",
            }
        )
        resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("esearchresult", {}).get("idlist", [])

    def _esummary(self, pmid):
        params = self._base_params()
        params.update({"db": "pubmed", "id": str(pmid)})
        resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json().get("result", {})
        rec = data.get(str(pmid), {})
        return {
            "pmid": str(pmid),
            "title": rec.get("title", ""),
            "pubdate": rec.get("pubdate", ""),
            "source": rec.get("source", ""),
        }

    def _fetch_abstracts(self, pmids):
        """
        Fetch abstracts via efetch XML and map by PMID.
        """
        if not pmids:
            return {}

        params = self._base_params()
        params.update(
            {
                "db": "pubmed",
                "id": ",".join(str(p) for p in pmids),
                "rettype": "abstract",
                "retmode": "xml",
            }
        )
        resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params=params,
            timeout=45,
        )
        resp.raise_for_status()

        abstract_map = {}
        root = ET.fromstring(resp.text)

        for article in root.findall(".//PubmedArticle"):
            pmid_node = article.find(".//PMID")
            if pmid_node is None or pmid_node.text is None:
                continue
            pmid = pmid_node.text.strip()

            parts = []
            for ab in article.findall(".//Abstract/AbstractText"):
                txt = "".join(ab.itertext()).strip()
                if txt:
                    parts.append(txt)
            abstract_map[pmid] = " ".join(parts)

        return abstract_map

    def query_pubmed(self, drug_name, symptom, top_k=5):
        query = f"({drug_name}) AND (Parkinson disease) AND ({symptom})"
        print(f"Querying PubMed for: {query}")

        pmids = self._esearch(query, retmax=top_k)
        summaries = []
        for pmid in pmids:
            try:
                summaries.append(self._esummary(pmid))
            except Exception as exc:
                print(f"[warn] Failed summary for PMID {pmid}: {exc}")

        abstract_map = self._fetch_abstracts([a["pmid"] for a in summaries])

        articles = []
        for a in summaries:
            pmid = a["pmid"]
            a["abstract"] = abstract_map.get(pmid, "")
            articles.append(a)
        return articles

    def generate_justification(self, drug_name, symptom, articles, max_items=3):
        if not articles:
            return "No PubMed evidence returned for this query."

        lines = [f"Evidence for {drug_name} and symptom '{symptom}':"]
        used = 0
        for art in articles:
            if used >= max_items:
                break
            title = art.get("title", "").strip()
            pubdate = art.get("pubdate", "").strip()
            pmid = art.get("pmid", "").strip()
            abstract = art.get("abstract", "").strip()

            snippet = (abstract[:260] + "...") if len(abstract) > 260 else abstract
            lines.append(f"- PMID {pmid} | {title} ({pubdate})")
            if snippet:
                lines.append(f"  Abstract snippet: {snippet}")
            used += 1

        lines.append("Note: Literature retrieval supports hypothesis generation, not clinical proof.")
        return "\n".join(lines)


if __name__ == "__main__":
    print("Interpretability tools loaded.")
