"""
BLIP-2 Image Captioning for PDF Embeddings (our project).
Generates image captions and embeddings for knowledge graph / retrieval.
Based on Hugging Face BLIP-2: https://huggingface.co/blog/blip-2
"""
import torch
import requests
from PIL import Image
from transformers import Blip2Model, Blip2ForConditionalGeneration, AutoProcessor
import logging
import base64
import io
from typing import List, Dict, Any, Optional, Union
import numpy as np
import os

# 1. Global fallback for Hugging Face
os.environ["HF_HOME"] = r"D:\cache\huggingface"

logger = logging.getLogger(__name__)


class BLIP2ImageCaptioner:
    """BLIP-2 captioning and image embeddings for PDF-extracted images."""
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device: str = "auto"):
        if BLIP2ImageCaptioner._initialized:
            return
        self.model_name = model_name
        self.device = self._get_device(device)
        self.cache_dir = r"D:\cache\huggingface\hub"  # 2. Set explicit cache directory
        self.processor = None
        self.model = None
        self._load_model()
        BLIP2ImageCaptioner._initialized = True

    def _get_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _load_model(self):
        try:
            logger.info(f"Loading BLIP-2: {self.model_name} into {self.cache_dir}")
            try:
                # 3. Add cache_dir to processor
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name, use_fast=True, cache_dir=self.cache_dir
                )
            except TypeError:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name, cache_dir=self.cache_dir
                )
            
            is_cuda = self.device == "cuda"
            torch_dtype = torch.float16 if is_cuda else torch.float32
            
            # 4. Add cache_dir to model kwargs
            kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": None,
                "low_cpu_mem_usage": True,  # Fixes OOM spikes during initialization
                "use_safetensors": True,
                "trust_remote_code": False,
                "cache_dir": self.cache_dir, 
            }
            
            # Only load the main model! Loading Blip2Model redundantly was eating an extra 15GB of RAM and causing the Access Violation!
            self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_name, **kwargs)
            self.embed_model = None  # We don't need this, we use self.model.vision_model anyway
            
            if is_cuda:
                self.model = self.model.to("cuda")
            else:
                self.model = self.model.to("cpu")
                
            self.model.eval()
            logger.info(f"BLIP-2 loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load BLIP-2: {e}")
            raise

    def generate_caption(self, image: Union[Image.Image, str, bytes], prompt: Optional[str] = None, max_new_tokens: int = 50) -> str:
        try:
            pil_image = self._prepare_image(image)
            if prompt:
                inputs = self.processor(pil_image, text=prompt, return_tensors="pt").to(
                    self.device, torch.float16 if self.device == "cuda" else torch.float32
                )
            else:
                inputs = self.processor(pil_image, return_tensors="pt").to(
                    self.device, torch.float16 if self.device == "cuda" else torch.float32
                )
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            if prompt and text.startswith(prompt):
                text = text[len(prompt):].strip()
            return text
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return ""

    def get_image_embeddings(self, image: Union[Image.Image, str, bytes]) -> Optional[np.ndarray]:
        try:
            pil_image = self._prepare_image(image)
            inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                vision_outputs = self.model.vision_model(**inputs)
                vision_embeds = vision_outputs.last_hidden_state
                pooled = vision_embeds.mean(dim=1)
            return pooled.cpu().numpy()
        except Exception as e:
            logger.error(f"Image embeddings failed: {e}")
            return None

    def _prepare_image(self, image: Union[Image.Image, str, bytes]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, str):
            try:
                image_data = base64.b64decode(image)
                return Image.open(io.BytesIO(image_data))
            except Exception:
                try:
                    response = requests.get(image)
                    return Image.open(io.BytesIO(response.content))
                except Exception:
                    raise ValueError("Invalid image string format")
        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image))
        raise ValueError("Unsupported image type")


def create_blip_captioner(model_name: str = "Salesforce/blip2-opt-2.7b") -> BLIP2ImageCaptioner:
    return BLIP2ImageCaptioner(model_name=model_name)