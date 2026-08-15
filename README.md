# Local Multimodal PDF RAG System

A full-stack, 100% local Retrieval-Augmented Generation (RAG) system for chatting with PDF documents and their embedded images.

This project extracts text and embedded images from PDFs, runs OCR and BLIP-2 for image captioning, generates vector embeddings using SentenceTransformers, and stores them in a local ChromaDB vector database. A React/Electron frontend allows users to query their documents using a local Ollama LLM, with responses grounded in the retrieved text and inline image references.

**Key Features:**
- **100% Local Privacy:** No API keys required, no cloud uploads. Everything runs on your machine.
- **Multimodal RAG:** Searches across text, OCR'd images, and AI-generated image captions simultaneously.
- **Context-Aware Visuals:** The chat UI displays exactly which diagrams or images the LLM referenced to answer your question.
- **Cross-Document Querying:** Index multiple PDFs and search across your entire knowledge base.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INDEXING PIPELINE                            │
│                                                                     │
│  PDF File  ──▶  pdf-embeddings-app  ──▶  JSON Export               │
│              (Extract + OCR + BLIP +        │                       │
│               Embeddings via FastAPI)        │                       │
│                                             ▼                       │
│                                    rag/scripts/db                   │
│                                    (Index into ChromaDB)            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                              │
│                                                                     │
│  User types question                                                │
│       │                                                             │
│       ▼                                                             │
│  rag/frontend  ──IPC──▶  rag/scripts/chat                          │
│  (Electron UI)           (Embed query → ChromaDB search →          │
│       │                   Ollama LLM → stream answer)              │
│       ◀──────────────────────────────────────────────              │
│  (Streamed answer + images displayed)                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
PDF/
├── pdf-embeddings-app/          ← Step 1: Process PDFs into embeddings
│   ├── backend/
│   │   ├── app/
│   │   │   ├── main.py          # FastAPI: /api/process-pdf, /api/embeddings/download
│   │   │   ├── pipeline.py      # Orchestrates text + image → OCR + BLIP → embeddings
│   │   │   ├── pdf_extractor.py # PyMuPDF text extraction + advanced OCR
│   │   │   ├── blip_captioner.py# BLIP-2 image captioning + image embeddings
│   │   │   └── embedding_service.py  # Sentence-transformers (all-MiniLM-L6-v2)
│   │   └── requirements.txt
│   └── frontend/
│       ├── src/App.jsx          # Upload PDF → Process → Download embeddings UI
│       ├── main.cjs             # Electron main process (spawns FastAPI backend)
│       └── package.json
│
├── rag/                         ← Step 2 & 3: Index + Chat
│   ├── scripts/                 # Python RAG pipeline scripts
│   │   ├── config.py            # Centralised paths & constants
│   │   ├── chat/
│   │   │   └── answer_generator.py  # RAGChatService: retrieve → augment → generate
│   │   ├── db/
│   │   │   └── indexer.py       # RAGDatabase: ingest JSON exports into ChromaDB
│   │   ├── search/
│   │   │   └── vector_search.py # VectorSearchService: diagnostic retrieval tool
│   │   └── requirements.txt
│   └── frontend/                # Electron chat UI
│       ├── src/App.jsx          # Chat window with streaming + image display
│       ├── main.cjs             # Electron main process (spawns answer_generator.py)
│       ├── preload.cjs          # Secure IPC bridge
│       └── package.json
│
├── chroma_db/                   ← ChromaDB vector database (auto-created)
└── testdata/                    ← JSON exports + extracted images
    └── images/
```

---

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | ≥ 3.10 | Backend scripts |
| Node.js | ≥ 18 | Electron frontends |
| [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) | ≥ 5.0 | Text extraction from images |
| [Ollama](https://ollama.com) | latest | Local LLM for RAG answers |
| CUDA GPU *(optional)* | — | Faster BLIP-2 inference |

---

## Setup

### 1 — PDF Embeddings App (Process PDFs)

```bash
# Terminal 1: Start the Python backend
cd pdf-embeddings-app/backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

```bash
# Terminal 2: Start the Electron frontend
cd pdf-embeddings-app/frontend
npm install
npm run electron:dev
```

The UI opens at `http://localhost:3000`. Upload a PDF → click **Process PDF** → click **Download Embeddings** (choose JSON format). Save the file into `testdata/`.

---

### 2 — Index into ChromaDB

```bash
cd pdf-embeddings-app/backend
venv\Scripts\activate
cd ../../                      # back to PDF/ root

python rag/scripts/db/indexer.py
# Reads all *.json files from testdata/ and indexes them into chroma_db/
```

You only need to do this once per PDF (or whenever you add new documents).

---

### 3 — Pull the LLM

```bash
ollama pull llama3.2:1b        # fast, ~800MB, good for testing
# or
ollama pull llama3.2           # better quality, ~2GB
```

---

### 4 — Chat with Your Documents

```bash
cd rag/frontend
npm install
npm run electron:dev
```

The chat window opens — type any question about your processed PDFs.

---

## Design Patterns

Both apps follow the same architecture established by the senior team:

### Python Backend
| Pattern | Implementation |
|---|---|
| **Singleton services** | `EmbeddingService`, `BLIP2ImageCaptioner`, `RAGChatService`, `RAGDatabase`, `VectorSearchService` all use `__new__` + `_initialized` to ensure one instance per process |
| **Factory functions** | `get_embedding_service()`, `get_rag_chat_service()`, `get_rag_database()` etc. — clean public API to obtain service instances |
| **Centralised config** | `rag/scripts/config.py` — all paths and constants in one place, no magic strings scattered across files |
| **Module-level logging** | `logger = logging.getLogger(__name__)` in every file, no bare `print()` |
| **Availability flags** | `FITZ_AVAILABLE`, `SENTENCE_TRANSFORMERS_AVAILABLE` — graceful degradation if a dependency is missing |
| **Type hints** | Full `List[Dict[str, Any]]`, `Optional[str]` annotations throughout |

### Frontend
| Pattern | Implementation |
|---|---|
| **Inline styles** | Zero CSS frameworks — all styling via React inline style objects with a `COLORS` token constant |
| **Named export default** | `export default function App()` pattern |
| **Lifecycle management** | `will-quit` handler in `main.cjs` kills any spawned Python process on exit |

---

## API Reference (pdf-embeddings-app)

| Endpoint | Method | Description |
|---|---|---|
| `/api/process-pdf` | `POST` | Upload a PDF; runs extraction + OCR + BLIP + embeddings |
| `/api/embeddings/download` | `GET` | Download last result as `?format=json\|csv\|xlsx` |
| `/api/health` | `GET` | Health check |
| `/docs` | — | Interactive Swagger UI |

**Session header:** `X-Session-ID: <your-id>` — allows multiple concurrent users.

---

## Diagnostic Tools

```bash
# Test retrieval quality directly (without the LLM)
python rag/scripts/search/vector_search.py "your search query"

# Re-index a specific file
python rag/scripts/db/indexer.py
# (reads all JSON files from testdata/)

# Run the full RAG chat in the terminal
python rag/scripts/chat/answer_generator.py "your question"
```

---

## Data Flow

```
PDF file
  │
  ▼  [pdf-embeddings-app/backend]
PyMuPDF ──▶ text chunks (per page)
         ──▶ embedded images
               │
               ├──▶ pytesseract OCR  ──▶ OCR text + text embedding (384-dim)
               └──▶ BLIP-2           ──▶ image caption + caption embedding
                                     ──▶ image embedding (vision encoder)
  │
  ▼  [Download JSON]
testdata/*.json  →  { rows: [ { source, page, content, embedding[] } ] }
  │
  ▼  [rag/scripts/db/indexer.py]
ChromaDB (chroma_db/)  →  cosine-similarity vector index
  │
  ▼  [rag/scripts/chat/answer_generator.py]
Query  →  embed (384-dim)  →  top-20 ChromaDB results  →  Ollama LLM  →  streamed answer
```
