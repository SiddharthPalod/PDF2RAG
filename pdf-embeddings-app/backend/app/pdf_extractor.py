"""
PDF extraction: text (page-level) + images + OCR. Our project code.
Uses PyMuPDF for PDF/images and pytesseract + OpenCV for OCR.
"""
import io
import base64
import logging
from typing import List, Dict, Any

from PIL import Image

try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking constants
# ---------------------------------------------------------------------------

#: Default max characters per text chunk.
#: ~500 chars ≈ 100-120 tokens — safely under the 256-token limit of all-MiniLM-L6-v2.
DEFAULT_CHUNK_CHARS: int = 500

#: Characters of overlap between consecutive chunks so cross-boundary context
#: is not silently dropped at split points.
DEFAULT_OVERLAP_CHARS: int = 100

#: Sentence-ending punctuation used to find clean split points.
_SENTENCE_END = frozenset("!?.") 


def extract_text_chunks_simple(file_content: bytes) -> List[Dict[str, Any]]:
    # Legacy: one chunk per page. Use extract_text_chunks() for production RAG.
    """Extract text as one chunk per page (no CRF/1A pipeline)."""
    if not FITZ_AVAILABLE:
        logger.warning("PyMuPDF not available")
        return []
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        chunks = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            if text:
                chunks.append({
                    "page_number": page_num,
                    "text_chunk": text,
                    "section_title": f"Page {page_num + 1}",
                    "chunk_type": "content",
                })
        doc.close()
        return chunks
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Production chunking — paragraph-aware sliding window
# ---------------------------------------------------------------------------

def _last_sentence_boundary(text: str, search_from: int, search_to: int) -> int:
    """Return the position just after the last sentence-ending punctuation in [search_from, search_to).

    Validates that the punctuation is followed by whitespace (not e.g. '3.14').

    Args:
        text: The full text string to search within.
        search_from: Start of the search window (inclusive).
        search_to: End of the search window (exclusive).

    Returns:
        Position just after the boundary, or -1 if none found.
    """
    for i in range(min(search_to, len(text)) - 1, search_from - 1, -1):
        if text[i] in _SENTENCE_END:
            if i + 1 < len(text) and text[i + 1] in (" ", "\n", "\r"):
                return i + 1
    return -1


def _split_text_into_chunks(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """Split a text string into overlapping chunks of at most max_chars characters.

    Prefers sentence boundaries (. ! ?), then word boundaries, to avoid cutting
    mid-sentence. Overlap preserves cross-boundary context for the embedder.

    Args:
        text: The text to split.
        max_chars: Maximum number of characters per chunk.
        overlap_chars: Characters of overlap between consecutive chunks.

    Returns:
        List of non-empty text chunk strings.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + max_chars

        if end >= len(text):
            tail = text[start:].strip()
            if tail:
                chunks.append(tail)
            break

        # Prefer a sentence boundary in the last 25% of the window.
        search_from = start + int(max_chars * 0.75)
        split_pos = _last_sentence_boundary(text, search_from, end)

        if split_pos <= start:
            # Fall back to word boundary.
            split_pos = text.rfind(" ", start, end)

        if split_pos <= start:
            # Hard cut — no whitespace found.
            split_pos = end

        chunk = text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)

        # Overlap: next chunk starts overlap_chars before the split point.
        start = max(start + 1, split_pos - overlap_chars)

    return chunks


def extract_text_chunks(
    file_content: bytes,
    max_chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> List[Dict[str, Any]]:
    """Extract text from PDF using paragraph-aware, overlapping chunk splitting.

    This is the production chunking function for the RAG pipeline.
    Unlike extract_text_chunks_simple (one chunk per whole page), this function:

    1. Splits each page's text into paragraphs by double-newline boundaries.
    2. Accumulates consecutive paragraphs until the chunk would exceed max_chunk_chars.
    3. Oversized paragraphs are further split by _split_text_into_chunks with
       overlap_chars overlap so cross-boundary context is never silently dropped.
    4. Every chunk stays well under the 256-token limit of all-MiniLM-L6-v2
       (500 chars ≈ 100-120 tokens by default).

    Args:
        file_content: Raw PDF bytes.
        max_chunk_chars: Maximum characters per chunk (default 500).
        overlap_chars: Characters of overlap between consecutive chunks (default 100).

    Returns:
        List of chunk dicts with keys:
          page_number, chunk_index, text_chunk, section_title, chunk_type.
    """
    if not FITZ_AVAILABLE:
        logger.warning("PyMuPDF not available — cannot extract text chunks.")
        return []

    all_chunks: List[Dict[str, Any]] = []
    chunk_idx = 0
    num_pages = 0

    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        num_pages = len(doc)

        for page_num in range(num_pages):
            page = doc[page_num]
            raw_text = page.get_text().strip()
            if not raw_text:
                continue

            # Split into paragraphs; double-newline is the primary boundary.
            paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

            accumulated = ""
            for para in paragraphs:
                candidate = (accumulated + "\n\n" + para).strip() if accumulated else para

                if len(candidate) <= max_chunk_chars:
                    # Fits — keep accumulating paragraphs.
                    accumulated = candidate
                else:
                    # Flush accumulated text first.
                    if accumulated:
                        for sub in _split_text_into_chunks(accumulated, max_chunk_chars, overlap_chars):
                            all_chunks.append({
                                "page_number": page_num,
                                "chunk_index": chunk_idx,
                                "text_chunk": sub,
                                "section_title": f"Page {page_num + 1}",
                                "chunk_type": "content",
                            })
                            chunk_idx += 1

                    # Start fresh with this paragraph.
                    accumulated = para

            # Flush remaining text for this page.
            if accumulated:
                for sub in _split_text_into_chunks(accumulated, max_chunk_chars, overlap_chars):
                    all_chunks.append({
                        "page_number": page_num,
                        "chunk_index": chunk_idx,
                        "text_chunk": sub,
                        "section_title": f"Page {page_num + 1}",
                        "chunk_type": "content",
                    })
                    chunk_idx += 1

        doc.close()
        logger.info(
            "Extracted %d text chunks from %d pages "
            "(max_chunk_chars=%d, overlap=%d)",
            len(all_chunks), num_pages, max_chunk_chars, overlap_chars,
        )
    except Exception as e:
        logger.error("Text chunking failed: %s", e)

    return all_chunks


def extract_images_from_pdf(file_content: bytes) -> List[Dict[str, Any]]:
    """Extract embedded images from PDF with base64 PNG data."""
    if not FITZ_AVAILABLE:
        return []
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        extracted = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            for img_index, img in enumerate(page.get_images()):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha < 4:
                        img_data = pix.tobytes("png")
                        pil_image = Image.open(io.BytesIO(img_data))
                        buf = io.BytesIO()
                        pil_image.save(buf, format="PNG")
                        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                        rects = page.get_image_rects(xref)
                        img_rect = None
                        if rects:
                            r = rects[0]
                            img_rect = [float(getattr(r, "x0", 0)), float(getattr(r, "y0", 0)),
                                        float(getattr(r, "x1", 0)), float(getattr(r, "y1", 0))]
                        extracted.append({
                            "page_number": page_num,
                            "image_index": img_index,
                            "image_data": img_base64,
                            "format": "PNG",
                            "width": pil_image.width,
                            "height": pil_image.height,
                            "bbox": img_rect,
                        })
                    pix = None
                except Exception as img_error:
                    logger.warning(f"Image {img_index} page {page_num + 1}: {img_error}")
        doc.close()
        return extracted
    except Exception as e:
        logger.error(f"Image extraction failed: {e}")
        return []


def extract_text_with_ocr(image: Image.Image, language: str = "eng") -> str:
    """OCR on image using our pipeline (pytesseract + preprocessing)."""
    try:
        return _image_to_string_advanced(image, language)
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ""


def _preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    try:
        import cv2
        import numpy as np
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        return Image.fromarray(img)
    except Exception as e:
        logger.warning(f"Preprocess failed: {e}")
        return image


def _preprocess_for_red_text(image: Image.Image) -> Image.Image:
    try:
        import cv2
        import numpy as np
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 70, 50]), np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        result = np.full_like(mask, 255)
        result[mask > 0] = 0
        return Image.fromarray(result)
    except Exception as e:
        logger.warning(f"Red preprocess failed: {e}")
        return image


def _choose_best_text(texts: List[str]) -> str:
    texts = [t.strip() for t in texts if t.strip()]
    return max(texts, key=len) if texts else ""


def _merge_texts(texts: List[str]) -> str:
    texts = [t.strip() for t in texts if t.strip()]
    merged, seen = [], set()
    for t in texts:
        for line in t.splitlines():
            line = line.strip()
            if line and line not in seen:
                merged.append(line)
                seen.add(line)
    return "\n".join(merged)


def _image_to_string_advanced(image: Image.Image, language: str) -> str:
    import pytesseract
    import numpy as np

    def extract_boxes(img_like, lang, source_tag):
        data = pytesseract.image_to_data(img_like, lang=lang, output_type=pytesseract.Output.DICT)
        boxes = []
        n = len(data.get("text", []))
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            conf_list = data.get("conf") or ["-1"]
            conf = float(conf_list[i]) if i < len(conf_list) and str(conf_list[i]) not in ("-1", "-1.0", "") else -1.0
            boxes.append({
                "x": int(data["left"][i]), "y": int(data["top"][i]),
                "w": int(data["width"][i]), "h": int(data["height"][i]),
                "text": text, "conf": conf, "source": source_tag,
            })
        return boxes

    def iou(a, b):
        ax1, ay1 = a["x"], a["y"]
        ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
        bx1, by1 = b["x"], b["y"]
        bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter_area / (area_a + area_b - inter_area + 1e-6)

    def deduplicate_boxes(all_boxes):
        priority = {"red": 1, "gray": 0, "normal": 0}
        sorted_boxes = sorted(all_boxes, key=lambda b: (b.get("conf", -1.0), priority.get(b.get("source"), 0)), reverse=True)
        kept = []
        for box in sorted_boxes:
            if all(iou(box, k) < 0.5 for k in kept):
                kept.append(box)
        return kept

    def group_and_reconstruct(kept_boxes):
        if not kept_boxes:
            return ""
        boxes = sorted(kept_boxes, key=lambda b: (b["y"], b["x"]))
        lines = []
        for box in boxes:
            placed = False
            for line in lines:
                ly_top = min(w["y"] for w in line)
                ly_bot = max(w["y"] + w["h"] for w in line)
                inter_top = max(ly_top, box["y"])
                inter_bot = min(ly_bot, box["y"] + box["h"])
                overlap_h = max(0, inter_bot - inter_top)
                min_h = min(box["h"], ly_bot - ly_top)
                if min_h > 0 and (overlap_h / float(min_h)) >= 0.5:
                    line.append(box)
                    placed = True
                    break
            if not placed:
                lines.append([box])
        lines = [sorted(line, key=lambda b: b["x"]) for line in lines]
        lines = sorted(lines, key=lambda line: int(np.median([b["y"] for b in line])))
        return "\n".join(" ".join(w["text"] for w in line) for line in lines)

    processed = _preprocess_image_for_ocr(image)
    boxes1 = extract_boxes(processed, language, "normal")
    gray = image.convert("L")
    boxes2 = extract_boxes(gray, language, "gray")
    red_processed = _preprocess_for_red_text(image)
    boxes3 = extract_boxes(red_processed, language, "red")
    kept = deduplicate_boxes(boxes1 + boxes2 + boxes3)
    reconstructed = group_and_reconstruct(kept)
    if reconstructed.strip():
        return reconstructed
    text1 = pytesseract.image_to_string(processed, lang=language)
    text2 = pytesseract.image_to_string(gray, lang=language)
    text3 = pytesseract.image_to_string(red_processed, lang=language)
    temp = _choose_best_text([text1, text2])
    return _merge_texts([temp, text3])
