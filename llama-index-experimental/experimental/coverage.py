from __future__ import annotations
from typing import List, Dict, Any
import os

SYSTEM = (
    "You are a fact extractor. Given a user question, break it down into a list of 1–4 atomic facts or sub-questions "
    "that must be satisfied for a complete answer. Respond as a numbered list only."
)

PROMPT = """Question:
{q}
List the atomic facts required:"""

class FactExtractor:
    """
    Model-agnostic fact extractor. Expects gen_fn(prompt) -> str
    """
    def __init__(self, gen_fn):
        self.gen_fn = gen_fn

    def extract(self, q: str) -> List[str]:
        txt = self.gen_fn(SYSTEM + "\n\n" + PROMPT.format(q=q))
        lines = [l.strip(" -0123456789.").strip() for l in txt.splitlines() if l.strip()]
        return [l for l in lines if l]

def compute_coverage(facts: List[str], enriched_table: List[Dict]) -> List[Dict]:
    """
    For each fact, check if any supported sentence overlaps with it (lexical or entailment ≥ threshold).
    """
    from rapidfuzz.fuzz import token_set_ratio
    rows = []
    for f in facts:
        best_ent = 0.0
        for r in enriched_table:
            if r.get("entail", 0) >= 0.5:
                overlap = token_set_ratio(f, r["sentence"]) / 100.0
                if overlap > 0.6:
                    best_ent = max(best_ent, r["entail"])
        rows.append({"fact": f, "covered": best_ent >= 0.5, "entail": round(best_ent,3)})
    return rows
