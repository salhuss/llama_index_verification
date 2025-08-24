from __future__ import annotations
from typing import List, Dict, Any, Optional
import re
import numpy as np
from rapidfuzz.fuzz import token_set_ratio
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def _split_sents(text: str) -> List[str]:
    parts = [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+', text) if s.strip()]
    return parts or [text]

class _NLI:
    def __init__(self, model_name="facebook/bart-large-mnli", device="cpu"):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.mdl.to(device); self.mdl.eval()
        self.device = device; self.ent_idx = 2

    @torch.no_grad()
    def entail(self, premise: str, hypothesis: str) -> float:
        enc = self.tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        probs = torch.softmax(self.mdl(**enc).logits[0], dim=-1).cpu().numpy()
        return float(probs[self.ent_idx])

class SentenceLevelCitations:
    """
    Postprocesses generated text:
      - splits into sentences
      - for each sentence, finds best supporting node (entailment + overlap)
      - appends bracketed citation index like [1]
      - returns enriched text + mapping table + unsupported flags
    """
    def __init__(self, model_name="facebook/bart-large-mnli", device="cpu", entail_thresh: float = 0.5):
        self.nli = _NLI(model_name, device)
        self.entail_thresh = entail_thresh

    def __call__(self, text: str, nodes: List[Any]) -> Dict[str, Any]:
        sents = _split_sents(text)
        supports = []
        # build candidate corpus
        corpus = [n.node.get_content() if hasattr(n, "node") else n.get_content() for n in nodes]
        idx_map = list(range(len(corpus)))

        for sent in sents:
            best = (-1.0, -1.0, -1)  # score, entail, idx
            for i, ctx in enumerate(corpus):
                ent = self.nli.entail(ctx, sent)
                ov = token_set_ratio(ctx, sent) / 100.0
                score = ent + 0.25 * ov
                if score > best[0]:
                    best = (score, ent, i)
            supports.append({"sentence": sent, "best_idx": best[2], "entail": round(best[1],3)})

        # Build cited text
        cited_parts = []
        table = []
        for k, sup in enumerate(supports):
            idx = sup["best_idx"]
            ent = sup["entail"]
            if idx >= 0 and ent >= self.entail_thresh:
                cited_parts.append(f"{sup['sentence']} [{idx+1}]")
            else:
                cited_parts.append(f"{sup['sentence']} [?]")
            table.append({
                "sent_idx": k,
                "sentence": sup["sentence"],
                "citation": (idx+1) if idx>=0 else None,
                "entail": ent
            })

        return {
            "text_with_citations": " ".join(cited_parts),
            "table": table,
            "sources": [
                {
                    "index": i+1,
                    "preview": (corpus[i][:300] + ("…" if len(corpus[i])>300 else "")),
                    "metadata": getattr(nodes[i].node if hasattr(nodes[i],"node") else nodes[i], "metadata", {}),
                } for i in range(len(corpus))
            ]
        }
