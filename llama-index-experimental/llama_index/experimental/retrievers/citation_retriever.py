from __future__ import annotations
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.callbacks import CallbackManager

# lightweight NLI verifier
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rapidfuzz.fuzz import token_set_ratio

@dataclass
class _NLIConfig:
    model_name: str = "facebook/bart-large-mnli"
    device: str = "cpu"
    max_length: int = 512

class _NLI:
    def __init__(self, cfg: _NLIConfig):
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
        self.mdl.to(cfg.device)
        self.mdl.eval()
        self.device = cfg.device
        self.max_length = cfg.max_length
        # BART MNLI: [contradiction, neutral, entailment] = [0,1,2]
        self.ent_idx = 2

    @torch.no_grad()
    def entail_prob(self, premise: str, hypothesis: str) -> float:
        enc = self.tok(
            premise, hypothesis, return_tensors="pt",
            truncation=True, max_length=self.max_length
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.mdl(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return float(probs[self.ent_idx])

def _split_sents(text: str) -> List[str]:
    import re
    parts = [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+', text) if s.strip()]
    return parts or [text]

class CitationConstrainedRetriever(BaseRetriever):
    """
    Wraps a base retriever. Reranks nodes by NLI entailment wrt the query and
    stores sentence-level evidence spans for later citation mapping.

    Returns NodeWithScore with metadata:
      - 'evidence_sents': top supporting sentences (list[str])
      - 'entail_score': float
    """
    def __init__(
        self,
        base_retriever: BaseRetriever,
        nli_model: str = "facebook/bart-large-mnli",
        device: str = "cpu",
        top_k: int = 8,
        evidence_per_node: int = 2,
        callback_manager: Optional[CallbackManager] = None,
    ):
        super().__init__(callback_manager=callback_manager)
        self.base = base_retriever
        self.top_k = top_k
        self.evidence_per_node = evidence_per_node
        self.nli = _NLI(_NLIConfig(model_name=nli_model, device=device))

    def _verify_node(self, node_text: str, query: str) -> Dict[str, Any]:
        sents = _split_sents(node_text)
        # Use both NLI and lexical overlap for robustness
        scored = []
        for s in sents:
            ent = self.nli.entail_prob(premise=s, hypothesis=query)
            overlap = token_set_ratio(s, query) / 100.0
            score = ent + 0.25 * overlap
            scored.append((score, ent, overlap, s))
        scored.sort(reverse=True, key=lambda x: x[0])
        top = scored[: self.evidence_per_node]
        return {
            "entail_score": float(np.mean([t[1] for t in top])) if top else 0.0,
            "evidence_sents": [t[3] for t in top],
        }

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        cand = self.base.retrieve(query_bundle)
        enriched: List[NodeWithScore] = []
        for nws in cand:
            meta = self._verify_node(nws.node.get_content(), query_bundle.query_str)
            # attach metadata non-destructively
            md = dict(nws.node.metadata or {})
            md["evidence_sents"] = meta["evidence_sents"]
            nws.node.metadata = md
            nws.score = float(meta["entail_score"])
            enriched.append(nws)
        # sort by entailment score
        enriched.sort(key=lambda x: x.score or 0.0, reverse=True)
        return enriched[: self.top_k]
