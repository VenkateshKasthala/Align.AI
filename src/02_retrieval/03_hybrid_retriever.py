import logging
from typing import List, Dict, Any, Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlignAI.HybridRetriever")


class HybridRetriever:
    def __init__(
        self,
        vector_store,
        bm25_retriever,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def _rrf_score(self, rank: int, weight: float = 1.0) -> float:
        return weight * (1.0 / (self.rrf_k + rank))

    def _normalize_dense_results(self, dense_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []

        for item in dense_results:
            normalized.append({
                "id": item.get("id"),
                "section": item.get("metadata", {}).get("section", "GENERAL"),
                "content": item.get("document", ""),
                "metadata": item.get("metadata", {}) or {},
                "raw_score": item.get("distance"),
                "retriever": "dense"
            })

        return normalized

    def _normalize_sparse_results(self, sparse_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []

        for item in sparse_results:
            normalized.append({
                "id": item.get("id"),
                "section": item.get("section", "GENERAL"),
                "content": item.get("content", ""),
                "metadata": item.get("metadata", {}) or {},
                "raw_score": item.get("score"),
                "retriever": "bm25"
            })

        return normalized

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_top_k: int = 10,
        sparse_top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        dense_results = self.vector_store.search(query, top_k=dense_top_k)
        sparse_results = self.bm25_retriever.search(query, top_k=sparse_top_k)

        dense_results = self._normalize_dense_results(dense_results)
        sparse_results = self._normalize_sparse_results(sparse_results)

        fused: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(dense_results, start=1):
            doc_id = item["id"]
            if not doc_id:
                continue

            if doc_id not in fused:
                fused[doc_id] = {
                    "id": doc_id,
                    "section": item["section"],
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "rrf_score": 0.0,
                    "sources": [],
                    "dense_rank": None,
                    "sparse_rank": None,
                    "dense_raw_score": None,
                    "sparse_raw_score": None,
                }

            fused[doc_id]["rrf_score"] += self._rrf_score(rank, self.dense_weight)
            fused[doc_id]["sources"].append("dense")
            fused[doc_id]["dense_rank"] = rank
            fused[doc_id]["dense_raw_score"] = item["raw_score"]

        for rank, item in enumerate(sparse_results, start=1):
            doc_id = item["id"]
            if not doc_id:
                continue

            if doc_id not in fused:
                fused[doc_id] = {
                    "id": doc_id,
                    "section": item["section"],
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "rrf_score": 0.0,
                    "sources": [],
                    "dense_rank": None,
                    "sparse_rank": None,
                    "dense_raw_score": None,
                    "sparse_raw_score": None,
                }

            fused[doc_id]["rrf_score"] += self._rrf_score(rank, self.sparse_weight)
            fused[doc_id]["sources"].append("bm25")
            fused[doc_id]["sparse_rank"] = rank
            fused[doc_id]["sparse_raw_score"] = item["raw_score"]

        ranked = sorted(
            fused.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        logger.info(
            "Hybrid retrieval complete | dense=%s sparse=%s fused=%s returning=%s",
            len(dense_results),
            len(sparse_results),
            len(ranked),
            min(top_k, len(ranked))
        )

        return ranked[:top_k]