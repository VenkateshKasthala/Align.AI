import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from google import genai
from google.genai import types
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlignAI.VectorStore")


class VectorStore:
    def __init__(
        self,
        db_path: str = "./db/align_ai",
        collection_name: str = "resume_assets",
        embedding_model: str = "gemini-embedding-001",
    ):
        self.api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or 
            os.getenv('API_KEY')
        )

        if not self.api_key:
            raise ValueError(
                "No API key found. Set  API_KEY "
            )

        self.client = genai.Client(api_key=self.api_key)
        self.embedding_model = embedding_model

        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

        logger.info(
            "VectorStore initialized | db_path=%s | collection=%s | model=%s",
            db_path,
            collection_name,
            embedding_model,
        )

    def _normalize_text(self, text: str) -> str:
        return " ".join((text or "").strip().split())

    def _make_chunk_id(self, chunk: Dict[str, Any], index: int) -> str:
        source = chunk.get("metadata", {}).get("source", "unknown_source")
        section = chunk.get("section", "GENERAL")
        content = self._normalize_text(chunk.get("content", ""))

        raw = json.dumps(
            {
                "source": source,
                "section": section,
                "content": content,
                "index": index,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"{source}::{section}::{digest}"

    def _embed_text(self, text: str, is_query: bool = False) -> List[float]:
        clean_text = self._normalize_text(text)
        if not clean_text:
            raise ValueError("Cannot embed empty text.")

        task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"

        try:
            response = self.client.models.embed_content(
                model=self.embedding_model,
                contents=clean_text,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return response.embeddings[0].values
        except Exception as e:
            raise RuntimeError(
                f"Embedding failed using model '{self.embedding_model}': {e}"
            ) from e

    def _prepare_metadata(self, chunk: Dict[str, Any], index: int) -> Dict[str, Any]:
        metadata = chunk.get("metadata", {}) or {}
        content = self._normalize_text(chunk.get("content", ""))

        return {
            "section": chunk.get("section", "GENERAL"),
            "source": metadata.get("source", "unknown_source"),
            "chunk_index": index,
            "char_count": len(content),
        }

    def upsert_resume(self, scrubbed_chunks: List[Dict[str, Any]]) -> int:
        if not scrubbed_chunks:
            logger.warning("No chunks provided to upsert_resume.")
            return 0

        ids, embeddings, documents, metadatas = [], [], [], []

        for i, chunk in enumerate(scrubbed_chunks):
            content = self._normalize_text(chunk.get("content", ""))
            if not content:
                logger.warning("Skipping empty chunk at index %s", i)
                continue

            try:
                vector = self._embed_text(content, is_query=False)
                chunk_id = self._make_chunk_id(chunk, i)
                metadata = self._prepare_metadata(chunk, i)

                ids.append(chunk_id)
                embeddings.append(vector)
                documents.append(content)
                metadatas.append(metadata)

            except Exception as e:
                logger.exception("Failed to embed chunk %s | section=%s | error=%s",
                                 i, chunk.get("section", "UNKNOWN"), e)

        if not ids:
            raise RuntimeError("No chunks were indexed. Embedding failed for all chunks.")

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info("Successfully indexed %s chunks into ChromaDB.", len(ids))
        return len(ids)

    def search(self, job_desc: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query = self._normalize_text(job_desc)
        if not query:
            raise ValueError("job_desc cannot be empty.")

        query_vector = self._embed_text(query, is_query=True)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        output = []
        for i in range(len(documents)):
            output.append(
                {
                    "id": ids[i] if i < len(ids) else None,
                    "document": documents[i],
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "distance": distances[i] if i < len(distances) else None,
                }
            )

        return output

    def delete_collection(self) -> None:
        name = self.collection.name
        self.chroma_client.delete_collection(name=name)
        logger.info("Deleted collection: %s", name)

    def reset_collection(self) -> None:
        name = self.collection.name
        self.chroma_client.delete_collection(name=name)
        self.collection = self.chroma_client.get_or_create_collection(name=name)
        logger.info("Reset collection: %s", name)


if __name__ == "__main__":
    v_store = VectorStore()

    test_data = [
        {
            "section": "EXPERIENCE: Flexon Technologies",
            "content": "Built real-time data streaming platform with Kafka and AWS Lambda.",
            "metadata": {"source": "resume.pdf"},
        },
        {
            "section": "EDUCATION",
            "content": "Master of Science in Computer Security from Rowan University.",
            "metadata": {"source": "resume.pdf"},
        },
    ]

    print("Step 1: Indexing Resume...")
    indexed = v_store.upsert_resume(test_data)
    print(f"Indexed {indexed} chunks.")

    print("\nStep 2: Testing Search...")
    query = "Data engineering and cloud infrastructure experience"
    results = v_store.search(query, top_k=1)

    if results:
        top = results[0]
        print(f"Top Match Found in [{top['metadata'].get('section', 'UNKNOWN')}]:\n")
        print(top["document"])