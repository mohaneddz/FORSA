from typing import List, Dict, Tuple
import logging
import hashlib
import re
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
    def _script(text: str) -> str:
        """'ar' when Arabic letters outnumber Latin ones, else 'fr'."""
        arabic = len(re.findall(r"[\u0600-\u06FF]", text))
        latin = len(re.findall(r"[A-Za-zÀ-ÿ]", text))
        return "ar" if arabic > latin else "fr"

    def _order(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Sort by score, with a small bonus for chunks in the question's language.
        Most documents exist in a French and an Arabic version that score
        almost the same, and the system prompt requires answering in the
        question's language.
        """
        query_script = self._script(query)
        bonus = config.SAME_LANGUAGE_BONUS
        return sorted(
            results,
            key=lambda r: r.score + (bonus if self._script(r.chunk.content or "") == query_script else 0.0),
            reverse=True,
        )

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
        Retrieve context for one query. See retrieve_many.
        If a timings dict is passed, per-stage durations in seconds are written into it.
        """
        return self.retrieve_many([query], top_k=top_k, timings=timings)[0]

    def retrieve_many(self, queries: List[str], top_k: int = None, timings: Dict[str, float] = None) -> List[List[SearchResult]]:
        """
        Hybrid retrieval for a batch of queries:
        1) Analyze each query -> normalized + key_terms
        2) Semantic retrieval: one embedding call and one FAISS search for the whole batch
        3) Lexical retrieval per query
        4) Merge + hybrid score
        5) Threshold filter on hybrid score
        6) Rerank every (query, candidate) pair in one batched cross-encoder pass,
           keep candidates scoring >= RERANK_MIN_SCORE, at most MAX_SOURCES
        7) Without the reranker, return the top_n hybrid results

        Batching matters for /api/chat, which receives many questions at once:
        per-question threads contend for the GPU and the GIL instead.
        If a timings dict is passed, per-stage durations (seconds, summed over
        the batch) are written into it.
        """
        from services.query_service import query_service

        stage_times = timings if timings is not None else {}

        # Use k_sem as default if top_k not specified
        k_final = top_k if top_k is not None else self.k_sem
        k_sem = min(self.k_sem, k_final * 2)  # safe bound
        k_lex = min(self.k_lex, k_final * 2)

        # 1) Analyze queries
        t0 = time.perf_counter()
        analyses = [query_service.enhance_query_for_retrieval(q) for q in queries]
        stage_times["analyze"] = time.perf_counter() - t0

        # 2) Semantic search
        sem_batches: List[List[Tuple[object, float]]] = [[] for _ in queries]
        try:
            t0 = time.perf_counter()
            q_embs = self.embedding_service.generate_embeddings([qa["normalized"] for qa in analyses])
            stage_times["embed"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            sem_batches = self.vector_store.search_batch(q_embs, k=k_sem)
            stage_times["faiss"] = time.perf_counter() - t0
        except Exception as e:
            logger.exception(f"Semantic search failed: {e}")

        # 3) Lexical search
        t0 = time.perf_counter()
        lex_batches: List[List[Tuple[object, float]]] = []
        for qa in analyses:
            key_terms = qa.get("key_terms", [])
            lex_results: List[Tuple[object, float]] = []
            if key_terms:
                try:
                    lex_results = self.vector_store.keyword_search(key_terms, k=k_lex) or []
                except Exception as e:
                    logger.exception(f"Lexical search failed: {e}")
            lex_batches.append(lex_results)
        stage_times["lexical"] = time.perf_counter() - t0

        # 4) Merge + 5) hybrid threshold
        t0 = time.perf_counter()
        threshold = config.SIMILARITY_THRESHOLD
        candidates: List[List[SearchResult]] = []
        for query, qa, sem_results, lex_results in zip(queries, analyses, sem_batches, lex_batches):
            # If both empty => try variations (semantic) as last resort
            if not sem_results and not lex_results:
                for variant in qa.get("variations", [])[1:]:
                    q_emb = self.embedding_service.generate_query_embedding(variant)
                    sem_results = self.vector_store.search(q_emb, k=k_sem) or []
                    if sem_results:
                        logger.info(f"Query variation worked: {variant[:60]}...")
                        break

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

            kept = [SearchResult(chunk=ch, score=float(h)) for ch, h, _, _ in merged if h >= threshold]
            # If threshold wipes everything, keep a small fallback (don't return empty unless truly empty)
            if not kept and merged:
                logger.warning("Hybrid threshold removed all results; keeping top few as fallback.")
                kept = [SearchResult(chunk=ch, score=float(h)) for ch, h, _, _ in merged[: min(10, len(merged))]]
            candidates.append(kept)
        stage_times["merge"] = time.perf_counter() - t0

        # 6) Rerank
        if self.use_reranker and self.reranker:
            t0 = time.perf_counter()
            results = self._rerank_batch(queries, candidates)
            stage_times["rerank"] = time.perf_counter() - t0
            return results

        # 7) final cut
        return [self._order(q, kept)[: self.top_n] for q, kept in zip(queries, candidates)]

    def _rerank_batch(self, queries: List[str], candidates: List[List[SearchResult]]) -> List[List[SearchResult]]:
        to_rerank = [kept[: config.RERANK_CANDIDATES] for kept in candidates]
        pairs = [(q, r.chunk.content or "") for q, kept in zip(queries, to_rerank) for r in kept]
        try:
            scores = self.reranker.score_pairs(pairs)
        except Exception as e:
            logger.exception(f"Reranking failed, falling back to hybrid order: {e}")
            return [kept[: config.MAX_SOURCES] for kept in to_rerank]

        results = []
        offset = 0
        for query, kept in zip(queries, to_rerank):
            scored = [SearchResult(chunk=r.chunk, score=s) for r, s in zip(kept, scores[offset:offset + len(kept)])]
            offset += len(kept)
            # An empty list means nothing in the corpus answers the question;
            # the chat endpoint then refuses without calling the LLM.
            passing = [r for r in scored if r.score >= config.RERANK_MIN_SCORE]
            results.append(self._order(query, passing)[: config.MAX_SOURCES])
        return results

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