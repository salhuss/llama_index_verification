from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer, ServiceContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.experimental.citation_retrieval import CitationConstrainedRetriever, SentenceLevelCitations

# 1) Ingest (drop a few .txt/.md/.pdf files into ./data)
docs = SimpleDirectoryReader("./data").load_data()
idx = VectorStoreIndex.from_documents(docs)

# 2) Base retriever (vector) -> wrap with CitationConstrainedRetriever
base = idx.as_retriever(similarity_top_k=20)
retriever = CitationConstrainedRetriever(base_retriever=base, top_k=6, device="cpu")

# 3) Standard synthesizer
synth = get_response_synthesizer(response_mode="compact")

# 4) Query engine
engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synth)

# 5) Ask
q = "Who discovered penicillin and in what year?"
resp = engine.query(q)

# 6) Postprocess with sentence-level citations
post = SentenceLevelCitations(device="cpu", entail_thresh=0.55)
enriched = post(str(resp), nodes=resp.source_nodes)

print("\n=== Answer with citations ===\n")
print(enriched["text_with_citations"])
print("\n=== Sources ===")
for s in enriched["sources"]:
    print(f"[{s['index']}] {s['preview']}".replace("\n"," "))
