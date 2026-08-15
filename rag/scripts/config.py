"""
RAG Pipeline Configuration — Our Project.
Centralised paths and constants shared across scripts/chat, scripts/db, and scripts/search.

NOTE ON PATHS
  This file lives at:  rag/scripts/config.py
  Project root is:     PDF/                     (three dirname() calls up from here)
  Backend package is:  PDF/pdf-embeddings-app/backend/

  When any script in this package is run directly by Electron (e.g.
  `python /abs/path/to/rag/scripts/chat/answer_generator.py`), it must first
  add the project root to sys.path before importing this module.  Each
  entry-point script includes a 4-level bootstrap block for that purpose.
  Importing this config then adds BACKEND_PATH so `from app.* import ...`
  also resolves cleanly.
"""
import os
import sys
from typing import Final

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------

#: rag/scripts/config.py  →  dirname x3  →  PDF/ (project root)
PROJECT_ROOT: Final[str] = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

#: Path to the pdf-embeddings-app Python backend package
BACKEND_PATH: Final[str] = os.path.join(PROJECT_ROOT, "pdf-embeddings-app", "backend")

#: Path to the persistent ChromaDB database directory
CHROMA_DB_PATH: Final[str] = os.path.join(PROJECT_ROOT, "chroma_db")

#: Path to the testdata directory (JSON exports + images)
TESTDATA_PATH: Final[str] = os.path.join(PROJECT_ROOT, "testdata")

# ---------------------------------------------------------------------------
# ChromaDB Constants
# ---------------------------------------------------------------------------

#: ChromaDB collection name — must match the name used during indexing
COLLECTION_NAME: Final[str] = "pdf_knowledge_base"

#: Expected embedding dimension produced by all-MiniLM-L6-v2
EMBEDDING_DIM: Final[int] = 384

# ---------------------------------------------------------------------------
# LLM Constants
# ---------------------------------------------------------------------------

#: Default Ollama model for answer generation
DEFAULT_LLM_MODEL: Final[str] = "llama3.2:1b"

#: Default number of results to retrieve from ChromaDB for RAG context
DEFAULT_N_RESULTS: Final[int] = 20

# ---------------------------------------------------------------------------
# Bootstrap: add project root + backend path to sys.path so that both
# `from rag.scripts.*` and `from app.*` imports resolve in all contexts.
# ---------------------------------------------------------------------------
for _p in [PROJECT_ROOT, BACKEND_PATH]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
