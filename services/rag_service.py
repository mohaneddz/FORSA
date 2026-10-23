from typing import List, Dict, Tuple
import logging
import hashlib
import time

from schemas import SearchResult
from config import config
from utils.logging_utils import log_execution_time
from services.vector_store import vector_store
from services.embedding_service import embedding_service
from services.reranker_service import reranker_service

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.reranker = reranker_service
        self.use_reranker = getattr(config, "USE_RERANKER", True)

        # Hybrid knobs
        self.k_sem = getattr(config, "TOP_K_SEMANTIC", 40)
        self.k_lex = getattr(config, "TOP_K_LEXICAL", 40)
        self.w_sem = getattr(config, "HYBRID_W_SEM", 0.65)
        self.w_lex = getattr(config, "HYBRID_W_LEX", 0.35)
        
        # Top N after reranking (use semantic as default)
        self.top_n = self.k_sem

    # ---------------------------
    # Hybrid helpers
    # ---------------------------
    @staticmethod
    def _chunk_key(chunk) -> str:
        """
        Stable dedupe key.
        Prefer chunk_id; fallback to hash(content + file).
        """
        md = chunk.metadata or {}
        cid = md.get("chunk_id")
        if cid:
            return f"cid:{cid}"
        file_name = md.get("file", "Unknown")
        h = hashlib.sha1((file_name + "\n" + (chunk.content or "")).encode("utf-8", errors="ignore")).hexdigest()
        return f"hash:{h}"

    @staticmethod
    def _minmax_norm(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        vals = list(scores.values())
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            # all same score => map to 1.0
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    def _merge_hybrid(
        self,
        sem: List[Tuple[object, float]],
        lex: List[Tuple[object, float]],
        limit: int
    ) -> List[Tuple[object, float, float, float]]:
        """
        Merge by chunk key. Return list of (chunk, hybrid_score, sem_score_norm, lex_score_norm).
        """
        sem_map: Dict[str, Tuple[object, float]] = {}
        lex_map: Dict[str, Tuple[object, float]] = {}

        for ch, s in sem:
            sem_map[self._chunk_key(ch)] = (ch, float(s))
        for ch, s in lex:
            lex_map[self._chunk_key(ch)] = (ch, float(s))

        # normalize within each channel so weights behave
        sem_scores = {k: v[1] for k, v in sem_map.items()}
        lex_scores = {k: v[1] for k, v in lex_map.items()}
        sem_n = self._minmax_norm(sem_scores)
        lex_n = self._minmax_norm(lex_scores)

        all_keys = set(sem_map.keys()) | set(lex_map.keys())
        merged = []
        for k in all_keys:
            ch = sem_map.get(k, lex_map.get(k))[0]
            s_sem = sem_n.get(k, 0.0)
            s_lex = lex_n.get(k, 0.0)
            hybrid = self.w_sem * s_sem + self.w_lex * s_lex
            merged.append((ch, hybrid, s_sem, s_lex))

        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:limit]

    # ---------------------------
    # Main retrieval
    # ---------------------------
    @log_execution_time
    def retrieve_and_rerank(self, query: str, top_k: int = None, timings: Dict[str, float] = None) -> List[SearchResult]:
        """
        Hybrid retrieval:
        1) Analyze query -> normalized + key_terms
        2) Semantic retrieval (k_sem)
        3) Lexical retrieval (k_lex)
        4) Merge + hybrid score
        5) Threshold filter on hybrid score (or keep all if you prefer)
        6) Rerank with BGE reranker
        7) Return top_n

        If a timings dict is passed, per-stage durations in seconds are written into it.
        """
        stage_times = timings if timings is not None else {}
        t0 = time.perf_counter()
        from services.query_service import query_service

        # Use k_sem as default if top_k not specified
        k_final = top_k if top_k is not None else self.k_sem
        k_sem = min(self.k_sem, k_final * 2)  # safe bound
        k_lex = min(self.k_lex, k_final * 2)

        # 1) Analyze query
        qa = query_service.enhance_query_for_retrieval(query)
        normalized_query = qa["normalized"]
        key_terms = qa.get("key_terms", [])
        stage_times["analyze"] = time.perf_counter() - t0

        logger.info(
            f"Query analysis: {qa.get('question_type')}, families: {qa.get('document_families')}, key_terms: {key_terms}"
        )

        # 2) Semantic search
        sem_results: List[Tuple[object, float]] = []
        try:
            t0 = time.perf_counter()
            q_emb = self.embedding_service.generate_query_embedding(normalized_query)
            stage_times["embed"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            sem_results = self.vector_store.search(q_emb, k=k_sem) or []
            stage_times["faiss"] = time.perf_counter() - t0
        except Exception as e:
            logger.exception(f"Semantic search failed: {e}")

        # 3) Lexical search
        lex_results: List[Tuple[object, float]] = []
        t0 = time.perf_counter()
        if key_terms:
            try:
                lex_results = self.vector_store.keyword_search(key_terms, k=k_lex) or []
            except Exception as e:
                logger.exception(f"Lexical search failed: {e}")

        stage_times["lexical"] = time.perf_counter() - t0

        logger.info(f"Semantic retrieved: {len(sem_results)} | Lexical retrieved: {len(lex_results)}")

        # If both empty => try variations (semantic) as last resort
        if not sem_results and not lex_results:
            vars_ = qa.get("variations", [])
            for variant in vars_[1:]:
                q_emb = self.embedding_service.generate_query_embedding(variant)
                sem_results = self.vector_store.search(q_emb, k=k_sem) or []
                if sem_results:
                    logger.info(f"Query variation worked: {variant[:60]}...")
                    break

        # 4) Merge
        t0 = time.perf_counter()
        merged = self._merge_hybrid(sem_results, lex_results, limit=max(k_final, self.top_n, 30))

        if config.DEBUG_LOG_RETRIEVAL:
            logger.info("=" * 80)
            logger.info(f"HYBRID RETRIEVAL TRANSPARENCY - Query: {query[:100]}")
            logger.info("=" * 80)
            for i, (chunk, h, s_sem, s_lex) in enumerate(merged[:10], 1):
                fn = (chunk.metadata or {}).get("file", "Unknown")
                head = (chunk.content or "")[:150].replace("\n", " ")
                logger.info(f"  [{i}] HYB={h:.3f} SEM={s_sem:.3f} LEX={s_lex:.3f} | File: {fn}")
                logger.info(f"      Content: {head}...")
            logger.info("=" * 80)

        # 5) Threshold filter (on hybrid score)
        threshold = config.SIMILARITY_THRESHOLD  # reuse existing
        kept: List[SearchResult] = []
        for chunk, hybrid_score, _, _ in merged:
            if hybrid_score >= threshold:
                kept.append(SearchResult(chunk=chunk, score=float(hybrid_score)))

        # If threshold wipes everything, keep a small fallback (don’t return empty unless truly empty)
        if not kept and merged:
            logger.warning("Hybrid threshold removed all results; keeping top few as fallback.")
            kept = [SearchResult(chunk=ch, score=float(h)) for ch, h, _, _ in merged[: min(10, len(merged))]]

        stage_times["merge"] = time.perf_counter() - t0

        # 6) Rerank with BGE reranker
        if self.use_reranker and self.reranker and kept:
            rerank_start = time.time()
            logger.info(f"[RERANK START] Reranking {len(kept)} results with BGE reranker")
            # Rerank more candidates than final top_n for better quality
            rerank_candidates = min(len(kept), max(self.top_n * 2, 20))
            to_rerank = kept[:rerank_candidates]
            kept = self.reranker.rerank(query, to_rerank, top_n=None)
            rerank_duration = time.time() - rerank_start
            logger.info(f"[RERANK END] Reranking completed in {rerank_duration:.2f}s - kept {len(kept)} results")
            stage_times["rerank"] = rerank_duration

        # 7) final cut
        kept.sort(key=lambda x: x.score, reverse=True)
        final_results = kept[: self.top_n]
        logger.info(f"Final results: {len(final_results)} (hybrid threshold={threshold})")
        return final_results

    def format_context_blocks(self, chunks: List[SearchResult]) -> str:
        """
        Format retrieved chunks into context blocks for the LLM.
        
        Args:
            chunks: List of SearchResult objects
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        context_parts = []
        for i, result in enumerate(chunks, 1):
            chunk = result.chunk
            content = chunk.content or ""
            metadata = chunk.metadata or {}
            source = metadata.get('file', metadata.get('source', 'Unknown'))
            
            context_parts.append(f"[Source {i}: {source}]\n{content}")
        
        return "\n\n".join(context_parts)


# Singleton instance
rag_service = RAGService()