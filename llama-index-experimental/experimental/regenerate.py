from __future__ import annotations
from typing import List, Dict, Any, Optional

SYSTEM = (
    "You rewrite ONLY the provided sentence so that it is fully supported by the given sources. "
    "Do not introduce new facts. If the sentence cannot be supported, produce a shorter, safely hedged version."
)

PROMPT = """Sources:
{sources}

Original sentence:
{sentence}

Rewrite the sentence so it is fully supported by the sources. Keep it concise and factual."""
    
class SentenceRegenerator:
    """
    Model-agnostic: expects a callable `gen_fn(prompt: str) -> str` (OpenAI, HF, or your adapter).
    """
    def __init__(self, gen_fn):
        self.gen_fn = gen_fn

    def _format_sources(self, source_texts: List[str], max_chars: int = 1200) -> str:
        merged = []
        budget = max_chars
        for s in source_texts:
            s = s.strip()
            if not s: 
                continue
            take = s[: min(len(s), budget)]
            merged.append(take)
            budget -= len(take)
            if budget <= 0:
                break
        return "\n---\n".join(merged)

    def regenerate(self, sentence: str, source_texts: List[str]) -> str:
        ctx = self._format_sources(source_texts)
        prompt = PROMPT.format(sources=ctx, sentence=sentence.strip())
        return self.gen_fn(prompt)
