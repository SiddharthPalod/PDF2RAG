"""
RAG Database Indexer — Our Project.
Reads the JSON export produced by pdf-embeddings-app and upserts all embedding
rows into a ChromaDB persistent collection for later vector search.
"""
import glob
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Bootstrap — required when run as a direct script:
#   python /abs/path/to/rag/scripts/db/indexer.py
# rag/scripts/db/indexer.py → dirname x4 → PDF/ (project root)
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import chromadb
from rag.scripts.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    TESTDATA_PATH,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGDatabase:
    """ChromaDB-backed vector store for PDF embedding rows.

    Uses the Singleton pattern so the ChromaDB client is only opened once per
    process regardless of how many callers import this module.
    """

    _instance: Optional["RAGDatabase"] = None
    _initialized: bool = False

    def __new__(cls, db_path: str = CHROMA_DB_PATH) -> "RAGDatabase":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db_path: str = CHROMA_DB_PATH) -> None:
        if RAGDatabase._initialized:
            return
        try:
            self._client = chromadb.PersistentClient(path=db_path)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            RAGDatabase._initialized = True
            logger.info(
                "RAGDatabase initialised (db=%s, collection=%s)", db_path, COLLECTION_NAME
            )
        except Exception as e:
            logger.error("RAGDatabase initialisation failed: %s", e)
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_from_json(self, json_filepath: str) -> int:
        """Read a pdf-embeddings-app JSON export and upsert rows into ChromaDB.

        Skips any row whose embedding vector does not match the expected
        dimension (EMBEDDING_DIM) to avoid corrupting the index.

        Args:
            json_filepath: Absolute path to the JSON export file.

        Returns:
            Number of rows successfully indexed.
        """
        logger.info("Ingesting from %s", json_filepath)
        try:
            with open(json_filepath, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to read %s: %s", json_filepath, e)
            return 0

        rows: List[Dict[str, Any]] = data.get("rows", [])
        ids: List[str] = []
        documents: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []

        for i, row in enumerate(rows):
            emb = row.get("embedding", [])

            # Unwrap nested list if the JSON was serialised with an extra dimension.
            if len(emb) == 1 and isinstance(emb[0], list):
                emb = emb[0]

            if len(emb) != EMBEDDING_DIM:
                logger.warning(
                    "Skipping row %d — invalid embedding dimension: %d (expected %d)",
                    i, len(emb), EMBEDDING_DIM,
                )
                continue

            content = row.get("content")
            if not content:
                logger.warning("Skipping row %d — empty content.", i)
                continue

            unique_id = f"item_{i}_page_{row.get('page_number')}_chunk_{row.get('chunk_index')}"
            if row.get("image_index") is not None:
                unique_id += f"_img_{row.get('image_index')}"

            meta: Dict[str, Any] = {
                "source": row.get("source", ""),
                "page_number": row.get("page_number", 0),
                "section_title": row.get("section_title", ""),
                "chunk_type": row.get("chunk_type", ""),
            }

            # Text-Bridge: attach image path to BLIP-2 caption rows so the
            # Electron frontend can display the referenced image alongside the answer.
            if row.get("source") == "image_blip2":
                page_num = row.get("page_number", 0)
                img_idx = row.get("image_index", 0)
                meta["type"] = "image_caption"
                meta["image_path"] = f"/testdata/images/page{page_num}_img{img_idx}.png"

            ids.append(unique_id)
            documents.append(content)
            embeddings.append(emb)
            metadatas.append(meta)

        if not ids:
            logger.warning("No valid rows found in %s — nothing indexed.", json_filepath)
            return 0

        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info("Indexed %d chunks from %s", len(ids), json_filepath)
        except Exception as e:
            logger.error("ChromaDB upsert failed: %s", e)
            return 0

        return len(ids)


def get_rag_database(db_path: str = CHROMA_DB_PATH) -> RAGDatabase:
    """Factory function — return the singleton RAGDatabase instance."""
    return RAGDatabase(db_path=db_path)


if __name__ == "__main__":
    db = get_rag_database()
    json_files = glob.glob(os.path.join(TESTDATA_PATH, "*.json"))

    if not json_files:
        logger.warning("No JSON files found in %s", TESTDATA_PATH)
    else:
        for json_file in json_files:
            logger.info("Processing %s ...", json_file)
            db.ingest_from_json(json_file)
