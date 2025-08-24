import json, math
from pathlib import Path
import typer
from statistics import mean
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.experimental.citation_retrieval import CitationConstrainedRetriever, SentenceLevelCitations

app = typer.Typer(help="Compute % supported sentences ≥ threshold over a small JSONL set.")

@app.command()
def main(
    data_dir: str = typer.Argument(..., help="Docs folder"),
    dataset: str = typer.Argument(..., help="JSONL with {'question': ...} per line"),
    device: str = typer.Option("cpu"),
    nli_model: str = typer.Option("facebook/bart-large-mnli"),
    top_k: int = typer.Option(6),
    entail_thresh: float = typer.Option(0.55),
    out: str = typer.Option("eval_out.json"),
):
    docs = SimpleDirectoryReader(data_dir).load_data()
    idx = VectorStoreIndex.from_documents(docs)
    base = idx.as_retriever(similarity_top_k=20)
    retr = CitationConstrainedRetriever(base_retriever=base, top_k=top_k, device=device, nli_model=nli_model)
    synth = get_response_synthesizer(response_mode="compact")
    engine = RetrieverQueryEngine(retriever=retr, response_synthesizer=synth)
    post = SentenceLevelCitations(device=device, entail_thresh=entail_thresh)

    rows = []
    with open(dataset, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            q = ex["question"]
            resp = engine.query(q)
            enriched = post(str(resp), nodes=resp.source_nodes)
            ents = [r["entail"] for r in enriched["table"] if r["sentence"].strip()]
            support_mask = [e >= entail_thresh for e in ents]
            support_rate = mean(support_mask) if ents else 0.0
            rows.append({
                "question": q,
                "n_sents": len(ents),
                "support_rate": round(support_rate, 3),
                "avg_entail": round(mean(ents), 3) if ents else 0.0
            })
            print(f"{q[:60]:60s}  support={support_rate:.2f}  avg_ent={mean(ents) if ents else 0:.2f}")

    summary = {
        "entail_thresh": entail_thresh,
        "n": len(rows),
        "avg_support_rate": round(mean([r["support_rate"] for r in rows]), 3) if rows else 0.0,
        "avg_entail": round(mean([r["avg_entail"] for r in rows]), 3) if rows else 0.0,
        "rows": rows,
    }
    Path(out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n✅ Wrote {out}\nAvg support={summary['avg_support_rate']:.3f}  Avg entail={summary['avg_entail']:.3f}")

if __name__ == "__main__":
    app()
