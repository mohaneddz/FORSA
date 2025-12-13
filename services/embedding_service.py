from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class EmbeddingService:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self._validate_dimension()
    
    def _validate_dimension(self):
        """Validate embedding dimension matches config"""
        from config import config
        
        if not config.ASSERT_EMBEDDING_DIM:
            return
        
        # Generate a test embedding to check dimension
        test_embedding = self.model.encode(["test"], convert_to_numpy=True)
        actual_dim = test_embedding.shape[-1]
        
        if actual_dim != config.EMBEDDING_DIMENSION:
            raise ValueError(
                f"CRITICAL: Embedding dimension mismatch!\n"
                f"  Expected: {config.EMBEDDING_DIMENSION}\n"
                f"  Actual:   {actual_dim}\n"
                f"  Model:    {self.model._modules['0'].auto_model.config._name_or_path}\n\n"
                f"Action required:\n"
                f"  1. Update EMBEDDING_DIMENSION in config.py to {actual_dim}\n"
                f"     OR change EMBEDDING_MODEL_NAME to match dimension {config.EMBEDDING_DIMENSION}\n"
                f"  2. Delete and rebuild FAISS index: python rebuild_index.py\n"
            )
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"✓ Embedding dimension validated: {actual_dim} == {config.EMBEDDING_DIMENSION}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        Returns a numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Normalize embeddings for cosine similarity (dot product on normalized vectors)
        # Faiss IndexFlatIP uses dot product.
        faiss.normalize_L2(embeddings)
        return embeddings

    def generate_query_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single query string.
        Returns a numpy array of shape (1, embedding_dim).
        """
        return self.generate_embeddings([text])

# Global instance
# In a real app, you might want to load this lazily or via dependency injection
import faiss # Import here to use normalize_L2
from config import config

embedding_service = EmbeddingService(model_name=config.EMBEDDING_MODEL_NAME)
