from sentence_transformers import SentenceTransformer
from typing import List
import logging
import numpy as np
import torch

from config import config

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # fp16 halves GPU memory and load time; FAISS still gets float32 vectors
        model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" and config.MODELS_FP16 else {}
        self.model = SentenceTransformer(model_name, device=device, model_kwargs=model_kwargs)
        self._validate_dimension()

    def _validate_dimension(self):
        """Validate embedding dimension matches config"""
        if not config.ASSERT_EMBEDDING_DIM:
            return

        actual_dim = self.model.get_sentence_embedding_dimension()

        if actual_dim != config.EMBEDDING_DIMENSION:
            raise ValueError(
                f"CRITICAL: Embedding dimension mismatch!\n"
                f"  Expected: {config.EMBEDDING_DIMENSION}\n"
                f"  Actual:   {actual_dim}\n"
                f"  Model:    {config.EMBEDDING_MODEL_NAME}\n\n"
                f"Action required:\n"
                f"  1. Update EMBEDDING_DIMENSION in config.py to {actual_dim}\n"
                f"     OR change EMBEDDING_MODEL_NAME to match dimension {config.EMBEDDING_DIMENSION}\n"
                f"  2. Delete and rebuild FAISS index: python rebuild_index.py\n"
            )

        logger.info(f"✓ Embedding dimension validated: {actual_dim} == {config.EMBEDDING_DIMENSION}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate L2-normalized embeddings for a list of texts, so the inner
        product used by FAISS IndexFlatIP is cosine similarity.
        Returns a float32 array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(np.float32, copy=False)

    def generate_query_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single query string.
        Returns a numpy array of shape (1, embedding_dim).
        """
        return self.generate_embeddings([text])


embedding_service = EmbeddingService(model_name=config.EMBEDDING_MODEL_NAME)
