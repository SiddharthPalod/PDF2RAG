"""
PDF Embeddings App - Our project API.
Upload PDFs, extract text + images, OCR + BLIP, generate embeddings, download.
"""
import io
import json
import logging
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Query, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from app.pipeline import run_pipeline, rows_to_export_format

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Embeddings API",
    description="Extract text and images from PDFs, run OCR + BLIP, generate embeddings. Download embeddings.",
    version="1.0.0",
    docs_url="/docs",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for last result per session (no DB)
_session_results: dict = {}
DEFAULT_SESSION = "default"


def _get_session_id(request: Request) -> str:
    return request.headers.get("X-Session-ID") or DEFAULT_SESSION


@app.post("/api/process-pdf")
async def process_pdf(
    request: Request,
    file: UploadFile = File(...),
    ocr_language: str = Query("eng", description="OCR language code"),
    include_ocr: bool = Query(True),
    include_blip_captions: bool = Query(True),
    include_text_embeddings: bool = Query(True),
    include_ocr_embeddings: bool = Query(True),
    include_blip_image_embeddings: bool = Query(True),
    include_caption_embeddings: bool = Query(True),
    max_chunk_chars: int = Query(500, description="Max characters per text chunk (default 500 ≈ 120 tokens)"),
    overlap_chars: int = Query(100, description="Overlap characters between consecutive chunks"),
):
    """
    Upload a PDF, run extraction + OCR + BLIP + embeddings. Result is stored for this session;
    use GET /api/embeddings/download to download.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File is empty")
        session_id = _get_session_id(request)
        logger.info(f"Processing PDF for session {session_id}: {file.filename}")
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        image_save_dir = os.path.join(project_root, "testdata", "images")
        
        result = run_pipeline(
            content,
            filename=file.filename,
            ocr_language=ocr_language,
            include_ocr=include_ocr,
            include_blip_captions=include_blip_captions,
            include_text_embeddings=include_text_embeddings,
            include_ocr_embeddings=include_ocr_embeddings,
            include_blip_image_embeddings=include_blip_image_embeddings,
            include_caption_embeddings=include_caption_embeddings,
            image_save_dir=image_save_dir,
            max_chunk_chars=max_chunk_chars,
            overlap_chars=overlap_chars,
        )
        _session_results[session_id] = result
        meta = result.get("metadata", {})
        return JSONResponse(content={
            "message": "Processing complete. Use Download Embeddings to get the file.",
            "filename": file.filename,
            "total_rows": meta.get("total_rows", 0),
            "total_text_chunks": meta.get("total_text_chunks", 0),
            "total_images": meta.get("total_images", 0),
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Process PDF failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/embeddings/download")
async def download_embeddings(
    request: Request,
    format: str = Query("json", description="json, csv, or xlsx"),
    include_embedding_vector: bool = Query(True, description="Include full embedding in output"),
):
    """
    Download the last processed embeddings for this session.
    Requires having called POST /api/process-pdf first.
    """
    session_id = _get_session_id(request)
    result = _session_results.get(session_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail="No embeddings found. Upload and process a PDF first.",
        )
    rows = result.get("rows", [])
    if not rows:
        raise HTTPException(status_code=404, detail="No embedding rows to export.")
    export_rows = rows_to_export_format(rows, include_embedding_vector=include_embedding_vector)
    filename_base = result.get("filename", "document").replace(".pdf", "")

    if format == "json":
        buf = io.BytesIO(json.dumps({"rows": export_rows, "metadata": result.get("metadata", {})}, indent=2).encode("utf-8"))
        return StreamingResponse(
            buf,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename_base}_embeddings.json"},
        )
    if format == "csv":
        import csv
        buf = io.StringIO()
        if export_rows:
            # Flatten: content + embedding as list or preview
            keys = ["source", "page_number", "chunk_index", "image_index", "content", "section_title", "chunk_type", "embedding_dim"]
            if include_embedding_vector and export_rows[0].get("embedding"):
                keys.append("embedding")
            else:
                keys.append("embedding_preview")
            writer = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for r in export_rows:
                row = {k: r.get(k) for k in keys}
                if "embedding" in row and isinstance(row["embedding"], list):
                    row["embedding"] = ",".join(str(x) for x in row["embedding"])
                writer.writerow(row)
        buf.seek(0)
        return StreamingResponse(
            io.BytesIO(buf.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename_base}_embeddings.csv"},
        )
    if format == "xlsx":
        try:
            import pandas as pd
            df = pd.DataFrame(export_rows)
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="embeddings")
            buf.seek(0)
            return StreamingResponse(
                buf,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename={filename_base}_embeddings.xlsx"},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Excel export failed: {e}")
    raise HTTPException(status_code=400, detail="format must be json, csv, or xlsx")


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "pdf-embeddings-app"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
