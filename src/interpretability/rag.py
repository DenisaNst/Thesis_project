"""
rag.py
-------
Scientific RAG via PubMed for the Parkinson's Drug Discovery Framework.

Retrieves and summarises literature evidence for a given (drug, symptom) pair
using NCBI E-utilities (no API key required, but recommended for higher rate limits).

Usage
-----
    from src.interpretability.rag import ScientificRAG

    rag = ScientificRAG(email="you@example.com")
    articles = rag.query_pubmed("Levodopa", "tremor", top_k=5)
    print(rag.generate_justification("Levodopa", "tremor", articles))
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import requests


class ScientificRAG:
    def __init__(self, email: str | None = None, api_key: str | None = None):
        """
        Parameters
        ----------
        email   : registered email for NCBI E-utilities (recommended)
        api_key : NCBI API key for higher rate limits (10 req/s vs 3 req/s)
        """
        self.email   = email
        self.api_key = api_key

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _base_params(self) -> dict:
        params = {"retmode": "json"}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _esearch(self, query: str, retmax: int = 5) -> list[str]:
        params = {**self._base_params(), "db": "pubmed", "term": query,
                  "retmax": retmax, "sort": "relevance"}
        resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=params, timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("esearchresult", {}).get("idlist", [])

    def _esummary(self, pmid: str) -> dict:
        params = {**self._base_params(), "db": "pubmed", "id": str(pmid)}
        resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params=params, timeout=30,
        )
        resp.raise_for_status()
        rec = resp.json().get("result", {}).get(str(pmid), {})
        return {
            "pmid":    str(pmid),
            "title":   rec.get("title", ""),
            "pubdate": rec.get("pubdate", ""),
            "source":  rec.get("source", ""),
        }

    def _fetch_abstracts(self, pmids: list[str]) -> dict[str, str]:
        if not pmids:
            return {}
        params = {
            **self._base_params(),
            "db": "pubmed",
            "id": ",".join(str(p) for p in pmids),
            "rettype": "abstract",
            "retmode": "xml",
        }
        resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params=params, timeout=45,
        )
        resp.raise_for_status()

        abstract_map: dict[str, str] = {}
        root = ET.fromstring(resp.text)
        for article in root.findall(".//PubmedArticle"):
            pmid_node = article.find(".//PMID")
            if pmid_node is None or not pmid_node.text:
                continue
            pmid = pmid_node.text.strip()
            parts = [
                "".join(ab.itertext()).strip()
                for ab in article.findall(".//Abstract/AbstractText")
            ]
            abstract_map[pmid] = " ".join(p for p in parts if p)
        return abstract_map

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query_pubmed(self, drug_name: str, symptom: str, top_k: int = 5) -> list[dict]:
        """
        Searches PubMed for articles linking drug_name, Parkinson's disease,
        and a specific symptom. Returns a list of article dicts with keys:
        pmid, title, pubdate, source, abstract.
        """
        query = f"({drug_name}) AND (Parkinson disease) AND ({symptom})"
        print(f"[PubMed] {query}")

        pmids = self._esearch(query, retmax=top_k)
        summaries = []
        for pmid in pmids:
            try:
                summaries.append(self._esummary(pmid))
            except Exception as exc:
                print(f"[warn] PMID {pmid}: {exc}")

        abstract_map = self._fetch_abstracts([a["pmid"] for a in summaries])
        for a in summaries:
            a["abstract"] = abstract_map.get(a["pmid"], "")
        return summaries

    def generate_justification(
        self, drug_name: str, symptom: str, articles: list[dict], max_items: int = 3
    ) -> str:
        """
        Formats retrieved articles into a human-readable justification string.
        """
        if not articles:
            return "No PubMed evidence returned for this query."

        lines = [f"Evidence for '{drug_name}' targeting symptom '{symptom}':"]
        for art in articles[:max_items]:
            snippet = art.get("abstract", "")
            if len(snippet) > 260:
                snippet = snippet[:260] + "..."
            lines.append(
                f"\n- PMID {art['pmid']} | {art['title']} ({art['pubdate']})"
            )
            if snippet:
                lines.append(f"  {snippet}")

        lines.append(
            "\nNote: Literature retrieval supports hypothesis generation, not clinical proof."
        )
        return "\n".join(lines)