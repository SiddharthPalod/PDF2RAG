"""
Text embedding service using sentence-transformers (our project).
Used for text chunks, OCR text, and BLIP caption embeddings.
"""
import hashlib
import logging
from typing import List

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingService:
    """Sentence-transformer embeddings for text (all-MiniLM-L6-v2)."""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if EmbeddingService._initialized:
            return
        self.model_name = "all-MiniLM-L6-v2"
        self.model = None
        self.embedding_cache = {}
        self.cache_max_size = 10000
        self._load_model()
        EmbeddingService._initialized = True
        logger.info("EmbeddingService initialized")

    def _load_model(self):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not available")
            return
        try:
            self.model = SentenceTransformer(self.model_name, cache_folder=r"D:\cache\huggingface\hub")
            logger.info(f"Model {self.model_name} loaded")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def create_embedding(self, text: str) -> List[float]:
        if not self.model:
            return [0.0] * 384
        try:
            h = self._get_text_hash(text)
            if h in self.embedding_cache:
                return self.embedding_cache[h]
            processed = self._preprocess_text(text)
            if not processed.strip():
                return [0.0] * 384
            emb = self.model.encode(processed, convert_to_tensor=False)
            emb_list = emb.tolist()
            self._cache_embedding(h, emb_list)
            return emb_list
        except Exception as e:
            logger.error(f"create_embedding failed: {e}")
            return []

    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not self.model:
            return [[0.0] * 384 for _ in texts]
        try:
            embeddings = []
            uncached_indices = []
            uncached_texts = []
            for i, text in enumerate(texts):
                h = self._get_text_hash(text)
                if h in self.embedding_cache:
                    embeddings.append(self.embedding_cache[h])
                else:
                    embeddings.append(None)
                    uncached_indices.append(i)
                    uncached_texts.append(self._preprocess_text(text))
            if uncached_texts:
                batch = self.model.encode(uncached_texts, convert_to_tensor=False, batch_size=32)
                for idx, vec in zip(uncached_indices, batch.tolist()):
                    embeddings[idx] = vec
                    self._cache_embedding(self._get_text_hash(texts[idx]), vec)
            return embeddings
        except Exception as e:
            logger.error(f"create_embeddings_batch failed: {e}")
            return [[0.0] * 384 for _ in texts]

    def _preprocess_text(self, text: str) -> str:
        if not text:
            return ""
        return " ".join(text.strip().split())

    def _get_text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _cache_embedding(self, text_hash: str, embedding: List[float]):
        if len(self.embedding_cache) >= self.cache_max_size:
            for key in list(self.embedding_cache.keys())[:100]:
                del self.embedding_cache[key]
        self.embedding_cache[text_hash] = embedding


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
