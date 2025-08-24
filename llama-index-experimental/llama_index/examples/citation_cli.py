import typer
from typing import Optional, List
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.experimental.citation_retrieval import (
    CitationConstrainedRetriever, SentenceLevelCitations, SentenceRegenerator
)

app = typer.Typer(help="Citation-constrained RAG with sentence-level verification & optional regeneration.")

# --- minimal adapters for regeneration ---
def openai_gen(model: str):
    from openai import OpenAI
    import os
    api = os.getenv("OPENAI_API_KEY")
    if not api:
        raise RuntimeError("OPENAI_API_KEY not set for regeneration.")
    client = OpenAI(api_key=api)
    def _gen(prompt: str) -> str:
        r = client.chat.completions.create(
            model=model, temperature=0.2,
            messages=[{"role":"system","content":"You are a careful editor."},
                      {"role":"user","content":prompt}]
        )
        return r.choices[0].message.content.strip()
    return _gen

@app.command()
def run(
    data_dir: str = typer.Argument(..., help="Folder with .txt/.md/.pdf to index"),
    question: str = typer.Argument(..., help="User question"),
    device: str = typer.Option("cpu", help="cpu or cuda"),
    nli_model: str = typer.Option("facebook/bart-large-mnli"),
    top_k: int = typer.Option(6),
    entail_thresh: float = typer.Option(0.55, help="Min entailment per sentence"),
    try_regen: bool = typer.Option(False, help="Regenerate unsupported sentences"),
    regen_model: str = typer.Option("gpt-4o-mini", help="OpenAI model for regeneration"),
):
    # 1) Ingest & index
    docs = SimpleDirectoryReader(data_dir).load_data()
    idx = VectorStoreIndex.from_documents(docs)
    base = idx.as_retriever(similarity_top_k=20)
    retr = CitationConstrainedRetriever(
        base_retriever=base, top_k=top_k, device=device, nli_model=nli_model
    )
    synth = get_response_synthesizer(response_mode="compact")
    engine = RetrieverQueryEngine(retriever=retr, response_synthesizer=synth)

    # 2) Ask & verify
    resp = engine.query(question)
    post = SentenceLevelCitations(device=device, entail_thresh=entail_thresh)
    enriched = post(str(resp), nodes=resp.source_nodes)

    print("\n=== Answer (citations) ===\n")
    print(enriched["text_with_citations"])

    # 3) Optionally regenerate unsupported sentences
    unsupported = [r for r in enriched["table"] if (r["entail"] < entail_thresh)]
    if try_regen and unsupported:
        print(f"\nRegenerating {len(unsupported)} unsupported sentence(s)…")
        # Gather source texts
        source_texts = []
        for sn in resp.source_nodes:
            source_texts.append(sn.node.get_content())

        regen = SentenceRegenerator(gen_fn=openai_gen(regen_model))
        rebuilt = []
        for row in enriched["table"]:
            sent = row["sentence"]
            if row["entail"] >= entail_thresh:
                rebuilt.append(sent)
            else:
                new_sent = regen.regenerate(sent, source_texts)
                rebuilt.append(new_sent)
        regenerated_text = " ".join(rebuilt)
        enriched2 = post(regenerated_text, nodes=resp.source_nodes)

        print("\n=== Regenerated (citations) ===\n")
        print(enriched2["text_with_citations"])

if __name__ == "__main__":
    app()
