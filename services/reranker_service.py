from typing import List, Sequence, Tuple
import logging
import threading

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from schemas import SearchResult
from config import config

logger = logging.getLogger(__name__)


class RerankerService:
    """
    Cross-encoder reranker (BAAI/bge-reranker-v2-m3).

    The model loads on first use, so a deployment with USE_RERANKER = False
    never pays for it. Scores are sigmoid(logit) in [0, 1], which makes
    RERANK_MIN_SCORE a fixed, query-independent threshold.
    """

    def __init__(self):
        self.model_name = config.RERANKER_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self):
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            logger.info(f"Loading reranker model: {self.model_name}")
            dtype = torch.float16 if self.device == "cuda" and config.MODELS_FP16 else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, torch_dtype=dtype)
            model.to(self.device)
            model.eval()
            self.model = model
            logger.info(f"Reranker model loaded on {self.device} ({dtype})")

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """Relevance score in [0, 1] for each (query, passage) pair."""
        if not pairs:
            return []
        self._ensure_loaded()
        # Batch pairs of similar length: one long chunk would otherwise pad its
        # whole batch to RERANK_MAX_LENGTH (median chunk is under 100 tokens).
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        scores: List[float] = [0.0] * len(pairs)
        batch_size = config.RERANK_BATCH_SIZE
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                batch_ids = order[start:start + batch_size]
                batch = [list(pairs[i]) for i in batch_ids]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=config.RERANK_MAX_LENGTH,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
                for i, score in zip(batch_ids, torch.sigmoid(logits).cpu().tolist()):
                    scores[i] = score
        return scores

    def rerank(self, query: str, results: List[SearchResult], top_n: int = None) -> List[SearchResult]:
        """
        Rerank search results for one query; each result's score becomes the
        reranker score. Returns them sorted, cut to top_n if given.
        """
        if not results:
            return []
        scores = self.score_pairs([(query, r.chunk.content or "") for r in results])
        reranked = [SearchResult(chunk=r.chunk, score=s) for r, s in zip(results, scores)]
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_n] if top_n is not None else reranked


reranker_service = RerankerService()
