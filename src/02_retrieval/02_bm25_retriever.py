import re
import logging
import hashlib
from typing import List, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlignAI.BM25Retriever")


class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []

    def _normalize_text(self, text: str) -> str:
        return " ".join((text or "").strip().split())

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _make_chunk_id(self, chunk: Dict[str, Any], index: int) -> str:
        source = chunk.get("metadata", {}).get("source", "unknown_source")
        section = chunk.get("section", "GENERAL")
        content = self._normalize_text(chunk.get("content", ""))
        raw = f"{source}|{section}|{content}|{index}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"{source}::{section}::{digest}"

    def index(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            logger.warning("No chunks provided to BM25 index.")
            return 0

        self.documents = []
        self.tokenized_corpus = []

        for i, chunk in enumerate(chunks):
            content = self._normalize_text(chunk.get("content", ""))
            if not content:
                continue

            doc = {
                "id": self._make_chunk_id(chunk, i),
                "section": chunk.get("section", "GENERAL"),
                "content": content,
                "metadata": chunk.get("metadata", {}) or {},
            }
            self.documents.append(doc)
            self.tokenized_corpus.append(self._tokenize(content))

        if not self.tokenized_corpus:
            raise ValueError("No valid text chunks found for BM25 indexing.")

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 indexed %s chunks.", len(self.documents))
        return len(self.documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            raise ValueError("BM25 index is not built. Call index() first.")

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue

            doc = self.documents[idx]
            results.append({
                "id": doc["id"],
                "section": doc["section"],
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": score,
                "retriever": "bm25"
            })

        return results