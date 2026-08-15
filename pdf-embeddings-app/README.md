# PDF Embeddings (Text + Image)

Standalone app for **extracting multiple images and text from PDFs**, with **OCR + BLIP-based captions** and **embeddings** (text, OCR, caption, and image embeddings). Simple UI: upload PDFs and download embeddings.

This folder contains only **our code** extracted for GitHub: the logic comes from our work in **DL_Eprior** (document_parser image extraction + OCR + BLIP integration, embedding_service), **blip_local** (BLIP-2 captioning), and a **simplified pipeline** (no DB/Faiss).

## Features

- **PDF extraction**: Page-level text chunks + embedded images (PyMuPDF)
- **OCR**: Text from images (pytesseract + preprocessing)
- **BLIP-2**: Image captions and image embeddings (Hugging Face BLIP-2)
- **Text embeddings**: Sentence-transformers (all-MiniLM-L6-v2) for text chunks, OCR text, and BLIP caption text
- **Download**: JSON, CSV, or Excel with embeddings

## Quick Start

### Backend

```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
npm run electron:dev
```

Open http://localhost:3000 — upload a PDF, click **Process PDF**, then **Download Embeddings** (choose format: JSON / CSV / Excel).

## API

- **POST /api/process-pdf** — Upload a PDF; runs extraction + OCR + BLIP + embeddings. Store result per session (header `X-Session-ID` optional).
- **GET /api/embeddings/download?format=json|csv|xlsx** — Download last processed embeddings for the session.
- **GET /api/health** — Health check.

## Project Structure

```
pdf-embeddings-app/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI: process-pdf, download
│   │   ├── pipeline.py       # PDF → text + images → OCR + BLIP → embeddings
│   │   ├── pdf_extractor.py  # Text (page-level) + image extraction + OCR
│   │   ├── blip_captioner.py # BLIP-2 captions and image embeddings
│   │   └── embedding_service.py  # Sentence-transformers text embeddings
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Upload PDF + Process + Download
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## Dependencies

- **Backend**: PyMuPDF, Pillow, pytesseract, opencv-python, sentence-transformers, torch/transformers (BLIP-2), FastAPI, pandas/openpyxl.
- **Frontend**: React 18, Vite, axios.