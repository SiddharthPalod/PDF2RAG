"""
RAG Answer Generator — Our Project.
Implements the full Retrieval-Augmented Generation pipeline:
  Phase 1 — Retrieval: embed the user query, search ChromaDB.
  Phase 2 — Augmentation: build a grounded context block from retrieved chunks.
  Phase 3 — Generation: stream an answer from the local Ollama LLM.
"""
import json
import logging
import os
import sys
from typing import List, Optional

# ---------------------------------------------------------------------------
# Bootstrap — required when Electron spawns this as a direct script:
#   python /abs/path/to/rag/scripts/chat/answer_generator.py "question"
# rag/scripts/chat/answer_generator.py → dirname x4 → PDF/ (project root)
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import chromadb
import ollama
from rag.scripts.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    DEFAULT_LLM_MODEL,
    DEFAULT_N_RESULTS,
)
from app.embedding_service import get_embedding_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Protocol constant — must match the marker parsed in rag/frontend/src/App.jsx
_IMAGES_MARKER: str = "__IMAGES_JSON__:"


class RAGChatService:
    """Retrieval-Augmented Generation chat service over a ChromaDB knowledge base.

    Uses the Singleton pattern (matching EmbeddingService / BLIP2ImageCaptioner)
    so the ChromaDB client and embedding model are only initialised once per
    process regardless of how many callers import this module.
    """

    _instance: Optional["RAGChatService"] = None
    _initialized: bool = False

    def __new__(cls) -> "RAGChatService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if RAGChatService._initialized:
            return
        try:
            self._client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            self._collection = self._client.get_collection(name=COLLECTION_NAME)
            self._embedder = get_embedding_service()
            RAGChatService._initialized = True
            logger.info(
                "RAGChatService initialised (db=%s, collection=%s)",
                CHROMA_DB_PATH,
                COLLECTION_NAME,
            )
        except Exception as e:
            logger.error("RAGChatService initialisation failed: %s", e)
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        user_question: str,
        llm_model: str = DEFAULT_LLM_MODEL,
        n_results: int = DEFAULT_N_RESULTS,
    ) -> None:
        """Run the full RAG pipeline, streaming the answer to stdout.

        Args:
            user_question: The natural-language question from the user.
            llm_model: Ollama model tag to use for generation.
            n_results: Number of ChromaDB results to include in context.
        """
        # --- Phase 1: Retrieval ---
        logger.info("Phase 1 — Retrieval: embedding query and searching ChromaDB.")
        try:
            query_vector = self._embedder.create_embedding(user_question)
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
            )
        except Exception as e:
            logger.error("Retrieval failed: %s", e)
            raise

        documents: List[str] = results["documents"][0]
        metadatas: List[dict] = results["metadatas"][0]

        # --- Phase 2: Augmentation ---
        logger.info("Phase 2 — Augmentation: assembling context block.")
        context_text, image_references = self._build_context(documents, metadatas)
        prompt = self._build_prompt(user_question, context_text)

        # --- Phase 3: Generation ---
        logger.info("Phase 3 — Generation: streaming answer via %s.", llm_model)
        try:
            self._stream_answer(prompt, llm_model)
        except Exception as e:
            logger.error("Generation failed: %s", e)
            raise

        # Emit image references in a parseable format for the Electron frontend.
        if image_references:
            print(f"\n{_IMAGES_MARKER} {json.dumps(image_references)}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_context(
        self, documents: List[str], metadatas: List[dict]
    ) -> tuple[str, List[str]]:
        """Build a grounded context string and collect image references.

        Args:
            documents: Retrieved document content strings.
            metadatas: Corresponding ChromaDB metadata dicts.

        Returns:
            Tuple of (context_text, image_references).
        """
        context_text = ""
        image_references: List[str] = []

        # Find the pages of the most relevant TEXT chunks
        relevant_pages = set()
        for meta in metadatas:
            if meta.get("type") != "image_caption" and "page_number" in meta:
                relevant_pages.add(meta["page_number"])

        for doc, meta in zip(documents, metadatas):
            context_text += f"---\n[Source: Page {meta.get('page_number', 'Unknown')}]\n{doc}\n"
            
            img_path = meta.get("image_path")
            # Only include the image if its page contains relevant text chunks,
            # or if the image caption itself was the ONLY thing retrieved (fallback)
            if img_path and img_path not in image_references:
                if not relevant_pages or meta.get("page_number") in relevant_pages:
                    image_references.append(img_path)

        return context_text, image_references

    def _build_prompt(self, user_question: str, context_text: str) -> str:
        """Construct the strict RAG prompt.

        Args:
            user_question: The original user question.
            context_text: The assembled context block from retrieved chunks.

        Returns:
            The formatted prompt string for the LLM.
        """
        return (
            "You are a highly accurate assistant analyzing a document.\n"
            "Answer the user's question using ONLY the context provided below.\n"
            'If the answer is not in the context, say "I cannot answer this based on the provided document."\n'
            "Do not use outside knowledge.\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION:\n{user_question}"
        )

    def _stream_answer(self, prompt: str, llm_model: str) -> None:
        """Stream the LLM response token-by-token to stdout.

        Args:
            prompt: The fully assembled RAG prompt.
            llm_model: Ollama model tag.
        """
        stream = ollama.chat(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
        # Flush a final newline so the last token is not left dangling.
        print("", flush=True)


def get_rag_chat_service() -> RAGChatService:
    """Factory function — return the singleton RAGChatService instance."""
    return RAGChatService()


if __name__ == "__main__":
    question = (
        sys.argv[1] if len(sys.argv) > 1
        else "What does the system architecture diagram on page 5 show?"
    )
    get_rag_chat_service().ask(question)
