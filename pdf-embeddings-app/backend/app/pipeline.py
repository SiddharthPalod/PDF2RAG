"""
PDF → text + images → OCR + BLIP → embeddings. Our project pipeline.
"""
import io
import json
import logging
from typing import List, Dict, Any, Optional

from app.pdf_extractor import (
    extract_text_chunks,
    extract_images_from_pdf,
    extract_text_with_ocr,
)
from app.embedding_service import get_embedding_service
from app.blip_captioner import create_blip_captioner, BLIP2ImageCaptioner

logger = logging.getLogger(__name__)


def run_pipeline(
    file_content: bytes,
    filename: str = "document.pdf",
    ocr_language: str = "eng",
    include_ocr: bool = True,
    include_blip_captions: bool = True,
    include_text_embeddings: bool = True,
    include_ocr_embeddings: bool = True,
    include_blip_image_embeddings: bool = True,
    include_caption_embeddings: bool = True,
    blip_model_name: str = "Salesforce/blip2-opt-2.7b",
    image_save_dir: Optional[str] = None,
    max_chunk_chars: int = 500,
    overlap_chars: int = 100,
) -> Dict[str, Any]:
    """
    Extract text + images from PDF, run OCR and BLIP, generate all embeddings.
    Returns a structure suitable for export (rows with source, content, embedding, etc.).
    """
    result = {
        "filename": filename,
        "text_chunks": [],
        "image_chunks": [],
        "rows": [],
        "metadata": {"total_text_chunks": 0, "total_images": 0, "ocr_language": ocr_language},
    }

    # 1) Text chunks — paragraph-aware overlapping split
    text_chunks = extract_text_chunks(
        file_content,
        max_chunk_chars=max_chunk_chars,
        overlap_chars=overlap_chars,
    )
    result["text_chunks"] = text_chunks
    result["metadata"]["total_text_chunks"] = len(text_chunks)

    # 2) Images
    images = extract_images_from_pdf(file_content)
    result["metadata"]["total_images"] = len(images)

    embedding_service = get_embedding_service() if (include_text_embeddings or include_ocr_embeddings or include_caption_embeddings) else None
    blip_captioner: Optional[BLIP2ImageCaptioner] = None
    if include_blip_captions or include_blip_image_embeddings or include_caption_embeddings:
        try:
            blip_captioner = create_blip_captioner(blip_model_name)
        except Exception as e:
            logger.warning(f"BLIP not available: {e}")

    # 3) Text chunk rows with embeddings
    for idx, ch in enumerate(text_chunks):
        text = (ch.get("text_chunk") or "").strip()
        if not text:
            continue
        embed = embedding_service.create_embedding(text) if embedding_service else []
        result["rows"].append({
            "source": "text",
            "page_number": ch.get("page_number", 0),
            "chunk_index": idx,
            "image_index": None,
            "content": text,
            "section_title": ch.get("section_title"),
            "chunk_type": ch.get("chunk_type", "content"),
            "embedding": embed,
            "embedding_dim": len(embed) if embed else None,
        })

    # 4) Process images: OCR + BLIP
    for img in images:
        page_num = img.get("page_number", 0)
        img_idx = img.get("image_index", 0)
        img_b64 = img.get("image_data", "")

        # OCR
        if include_ocr and img_b64:
            try:
                import base64
                from PIL import Image
                import os
                
                img_bytes = base64.b64decode(img_b64)
                pil_image = Image.open(io.BytesIO(img_bytes))
                
                # Save the image to disk if requested
                if image_save_dir:
                    os.makedirs(image_save_dir, exist_ok=True)
                    save_path = os.path.join(image_save_dir, f"page{page_num}_img{img_idx}.png")
                    pil_image.save(save_path, format="PNG")
                    logger.info(f"Saved image to {save_path}")
                    
                ocr_text = extract_text_with_ocr(pil_image, ocr_language)
                img["ocr_text"] = ocr_text
                if ocr_text.strip() and include_ocr_embeddings and embedding_service:
                    img["ocr_embedding"] = embedding_service.create_embedding(ocr_text)
                else:
                    img["ocr_embedding"] = []
            except Exception as e:
                logger.warning(f"OCR failed for image: {e}")
                img["ocr_text"] = ""
                img["ocr_embedding"] = []
        else:
            img["ocr_text"] = ""
            img["ocr_embedding"] = []

        # BLIP caption + image embeddings
        if blip_captioner and img_b64:
            try:
                caption = blip_captioner.generate_caption(img_b64)
                img["blip_caption"] = caption
                if include_blip_image_embeddings:
                    blip_emb = blip_captioner.get_image_embeddings(img_b64)
                    img["blip_embedding"] = blip_emb.tolist() if blip_emb is not None else []
                else:
                    img["blip_embedding"] = []
                if include_caption_embeddings and embedding_service and caption.strip():
                    img["blip_caption_embedding"] = embedding_service.create_embedding(caption)
                else:
                    img["blip_caption_embedding"] = []
            except Exception as e:
                logger.warning(f"BLIP failed for image: {e}")
                img["blip_caption"] = ""
                img["blip_embedding"] = []
                img["blip_caption_embedding"] = []
        else:
            img["blip_caption"] = ""
            img["blip_embedding"] = []
            img["blip_caption_embedding"] = []

        # Rows: OCR
        if include_ocr and (img.get("ocr_text") or "").strip():
            ocr_embed = img.get("ocr_embedding") or []
            result["rows"].append({
                "source": "image_ocr",
                "page_number": page_num,
                "chunk_index": None,
                "image_index": img_idx,
                "content": img["ocr_text"].strip(),
                "section_title": f"Image {img_idx + 1} OCR",
                "chunk_type": "image_ocr",
                "embedding": ocr_embed,
                "embedding_dim": len(ocr_embed) if ocr_embed else None,
            })
        # Rows: BLIP caption
        if include_blip_captions and (img.get("blip_caption") or "").strip():
            cap_embed = img.get("blip_caption_embedding") or []
            result["rows"].append({
                "source": "image_blip2",
                "page_number": page_num,
                "chunk_index": None,
                "image_index": img_idx,
                "content": img["blip_caption"].strip(),
                "section_title": f"Image {img_idx + 1} Caption",
                "chunk_type": "image_blip2",
                "embedding": cap_embed,
                "embedding_dim": len(cap_embed) if cap_embed else None,
            })
        # Rows: BLIP image embedding (no text content)
        if include_blip_image_embeddings and (img.get("blip_embedding") or []):
            result["rows"].append({
                "source": "image_embedding",
                "page_number": page_num,
                "chunk_index": None,
                "image_index": img_idx,
                "content": None,
                "section_title": f"Image {img_idx + 1}",
                "chunk_type": "image_embedding",
                "embedding": img["blip_embedding"],
                "embedding_dim": len(img["blip_embedding"]) if img["blip_embedding"] else None,
            })

    result["metadata"]["total_rows"] = len(result["rows"])
    return result


def rows_to_export_format(rows: List[Dict], include_embedding_vector: bool = True) -> List[Dict]:
    """Convert pipeline rows to export-friendly format (embedding as list or preview)."""
    out = []
    for r in rows:
        emb = r.get("embedding") or []
        if include_embedding_vector:
            out.append({
                "source": r.get("source"),
                "page_number": r.get("page_number"),
                "chunk_index": r.get("chunk_index"),
                "image_index": r.get("image_index"),
                "content": r.get("content"),
                "section_title": r.get("section_title"),
                "chunk_type": r.get("chunk_type"),
                "embedding_dim": r.get("embedding_dim"),
                "embedding": emb,
            })
        else:
            preview = ",".join([f"{x:.4f}" for x in emb[:8]]) if emb else None
            out.append({
                "source": r.get("source"),
                "page_number": r.get("page_number"),
                "chunk_index": r.get("chunk_index"),
                "image_index": r.get("image_index"),
                "content": r.get("content"),
                "section_title": r.get("section_title"),
                "chunk_type": r.get("chunk_type"),
                "embedding_dim": r.get("embedding_dim"),
                "embedding_preview": preview,
            })
    return out
