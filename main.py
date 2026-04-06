from core.ingestion import ResumeIngestor
from core.privacy import PrivacyScrubber
from core.vector_store import VectorStore
from core.bm25_retriever import BM25Retriever
from core.hybrid_retriever import HybridRetriever

def main():
    resume_path = "resume.pdf"
    job_desc = """
    Looking for a data engineer with Snowflake, PySpark, Airflow,
    AWS Lambda, and real-time data pipeline experience.
    """
    ingestor = ResumeIngestor()
    scrubber = PrivacyScrubber()

    chunks = ingestor.process(resume_path)
    scrubbed_chunks = scrubber.scrub(chunks)

    vector_store = VectorStore()
    vector_store.upsert_resume(scrubbed_chunks)

    bm25 = BM25Retriever()
    bm25.index(scrubbed_chunks)

    hybrid = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25,
        rrf_k=60,
        dense_weight=1.0,
        sparse_weight=1.0,
    )

    results = hybrid.search(
        query=job_desc,
        top_k=5,
        dense_top_k=10,
        sparse_top_k=10
    )

    print("\nTop hybrid results:\n")
    for i, r in enumerate(results, start=1):
        print(f"[{i}] {r['section']}")
        print(f"RRF Score: {r['rrf_ssrc']:.6f}")
        print(f"Sources: {r['sources']}")
        print(f"Dense Rank: {r['dense_rank']} | Sparse Rank: {r['sparse_rank']}")
        print(r["content"][:300])
        print("-" * 80)

if __name__ == "__main__":
    main()