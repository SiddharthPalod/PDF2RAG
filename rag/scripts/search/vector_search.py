"""
RAG Vector Search — Our Project.
Diagnostic/development tool for inspecting ChromaDB retrieval quality.
Searches the knowledge base with a plain-text query and prints ranked results,
highlighting any image-caption matches.
"""
import json
import logging
import os
import sys
from typing import List, Optional

# ---------------------------------------------------------------------------
# Bootstrap — required when run as a direct script:
#   python /abs/path/to/rag/scripts/search/vector_search.py
# rag/scripts/search/vector_search.py → dirname x4 → PDF/ (project root)
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import chromadb
from rag.scripts.config import CHROMA_DB_PATH, COLLECTION_NAME
from app.embedding_service import get_embedding_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DEFAULT_N_RESULTS: int = 5


class VectorSearchService:
    """Thin wrapper around ChromaDB for ad-hoc retrieval diagnostics.

    Uses the Singleton pattern matching EmbeddingService / RAGChatService so
    the client and embedding model are only initialised once per process.
    """

    _instance: Optional["VectorSearchService"] = None
    _initialized: bool = False

    def __new__(cls) -> "VectorSearchService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if VectorSearchService._initialized:
            return
        try:
            self._client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            self._collection = self._client.get_collection(name=COLLECTION_NAME)
            self._embedder = get_embedding_service()
            VectorSearchService._initialized = True
            logger.info(
                "VectorSearchService initialised (db=%s, collection=%s)",
                CHROMA_DB_PATH,
                COLLECTION_NAME,
            )
        except Exception as e:
            logger.error("VectorSearchService initialisation failed: %s", e)
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, user_question: str, n_results: int = _DEFAULT_N_RESULTS) -> List[dict]:
        """Embed the question and retrieve the top-N closest chunks.

        Args:
            user_question: Natural-language search query.
            n_results: Number of results to return.

        Returns:
            List of result dicts with keys: document, metadata, distance.
        """
        logger.info("Embedding query: '%s'", user_question)
        try:
            query_vector = self._embedder.create_embedding(user_question)
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
            )
        except Exception as e:
            logger.error("Vector search failed: %s", e)
            return []

        documents: List[str] = results["documents"][0]
        metadatas: List[dict] = results["metadatas"][0]
        distances: List[float] = results["distances"][0]

        return [
            {"document": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    def print_results(self, results: List[dict]) -> None:
        """Pretty-print retrieval results to stdout (for CLI diagnostics).

        Args:
            results: List of result dicts as returned by search().
        """
        print("\n" + "=" * 50)
        print("🔍 RETRIEVAL RESULTS")
        print("=" * 50)

        for i, r in enumerate(results):
            meta = r["metadata"]
            print(f"\nResult {i + 1} | Distance: {r['distance']:.4f}")

            if meta.get("type") == "image_caption":
                print("🖼️  [IMAGE MATCH] This text describes an image!")
                print(f"🔗  Image Path: {meta.get('image_path')}")

            print(f"Metadata: {json.dumps(meta, indent=2)}")
            print(f"Content:\n{r['document']}")
            print("-" * 50)


def get_vector_search_service() -> VectorSearchService:
    """Factory function — return the singleton VectorSearchService instance."""
    return VectorSearchService()


if __name__ == "__main__":
    query = (
        sys.argv[1] if len(sys.argv) > 1
        else "show me a diagram of the system model or architecture"
    )
    service = get_vector_search_service()
    hits = service.search(query)
    service.print_results(hits)
