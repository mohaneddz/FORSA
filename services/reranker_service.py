from typing import List
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from schemas import SearchResult
from config import config

logger = logging.getLogger(__name__)


class RerankerService:
    def __init__(self):
        self.model_name = "BAAI/bge-reranker-v2-m3"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the reranker model and tokenizer."""
        try:
            logger.info(f"Loading reranker model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Reranker model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise

    def rerank(self, query: str, results: List[SearchResult], top_n: int = None) -> List[SearchResult]:
        """
        Rerank search results using the BGE reranker model.
        
        Args:
            query: The user query
            results: List of SearchResult objects to rerank
            top_n: Number of top results to return (if None, returns all reranked)
            
        Returns:
            Reranked list of SearchResult objects
        """
        if not results:
            return []

        if self.model is None or self.tokenizer is None:
            logger.warning("Reranker model not loaded, returning original results")
            return results

        try:
            # Prepare pairs of (query, document)
            pairs = []
            for result in results:
                content = result.chunk.content or ""
                pairs.append([query, content])

            # Tokenize and get scores
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
                scores = scores.cpu().numpy()

            # Update SearchResult scores with reranker scores
            reranked = []
            for result, score in zip(results, scores):
                reranked.append(SearchResult(
                    chunk=result.chunk,
                    score=float(score)
                ))

            # Sort by reranker score
            reranked.sort(key=lambda x: x.score, reverse=True)

            # Return top_n if specified
            if top_n is not None:
                reranked = reranked[:top_n]

            logger.info(f"Reranked {len(results)} results, returning top {len(reranked)}")
            return reranked

        except Exception as e:
            logger.exception(f"Reranking failed: {e}")
            return results  # Return original results on failure


# Singleton instance
reranker_service = RerankerService()
