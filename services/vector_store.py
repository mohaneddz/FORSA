import faiss
import numpy as np
import pickle
import os
import logging
from typing import List, Dict, Tuple
from config import config
from schemas import DocumentChunk

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks: List[DocumentChunk] = [] # Metadata storage
        self.dimension = config.EMBEDDING_DIMENSION
        self.index_path = config.FAISS_INDEX_PATH
        self.similarity_metric = config.SIMILARITY_METRIC
        self.index_type = config.INDEX_TYPE
        self._lowered = None  # cached (content, file, source) in lowercase for keyword_search
        self._initialize_index()

    def _initialize_index(self):
        if os.path.exists(self.index_path):
            self.load_index()
        else:
            self.index = self._create_new_index()
            logger.info(f"Created new {self.index_type} index with {self.similarity_metric} similarity")
    
    def _create_new_index(self):
        """Create a new FAISS index based on configuration."""
        if self.index_type == 'Flat':
            if self.similarity_metric == 'cosine' or self.similarity_metric == 'inner_product':
                # IndexFlatIP for cosine similarity (with normalized vectors) or inner product
                return faiss.IndexFlatIP(self.dimension)
            elif self.similarity_metric == 'l2':
                # IndexFlatL2 for L2 (Euclidean) distance
                return faiss.IndexFlatL2(self.dimension)
        elif self.index_type == 'IVF':
            # IVF index for faster approximate search
            quantizer = faiss.IndexFlatIP(self.dimension) if self.similarity_metric != 'l2' else faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, config.IVF_NLIST)
            return index
        elif self.index_type == 'HNSW':
            # HNSW index for very fast approximate search
            index = faiss.IndexHNSWFlat(self.dimension, config.HNSW_M)
            index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
            index.hnsw.efSearch = config.HNSW_EF_SEARCH
            return index
        else:
            logger.warning(f"Unknown index type {self.index_type}, defaulting to Flat")
            return faiss.IndexFlatIP(self.dimension)

    def add_documents(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the index.
        """
        if self.index is None:
            self.index = self._create_new_index()
        
        # Train IVF index if necessary
        if self.index_type == 'IVF' and not self.index.is_trained:
            logger.info(f"Training IVF index with {len(embeddings)} vectors...")
            self.index.train(embeddings)
            logger.info("IVF index training complete")
             
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self._lowered = None
        logger.info(f"Added {len(chunks)} documents to index. Total documents: {len(self.chunks)}")
        self.save_index()

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for top k similar documents.
        Returns a list of (DocumentChunk, score).
        
        Note: For cosine similarity, query_vector should already be L2-normalized
        by the embedding service.
        """
        return self.search_batch(query_vector, k)[0]

    def search_batch(self, query_vectors: np.ndarray, k: int = 5) -> List[List[Tuple[DocumentChunk, float]]]:
        """
        Search for top k similar documents for each row of query_vectors.
        Returns one list of (DocumentChunk, score) per query.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty or not initialized")
            return [[] for _ in range(len(query_vectors))]

        # Validate query vector dimension
        if query_vectors.shape[-1] != self.dimension:
            raise ValueError(
                f"Query vector dimension mismatch! "
                f"Expected {self.dimension}, got {query_vectors.shape[-1]}"
            )

        # Set nprobe for IVF index
        if self.index_type == 'IVF' and hasattr(self.index, 'nprobe'):
            self.index.nprobe = config.IVF_NPROBE

        distances, indices = self.index.search(query_vectors, k)

        results = []
        for row_distances, row_indices in zip(distances, indices):
            results.append([
                (self.chunks[idx], float(dist))
                for idx, dist in zip(row_indices, row_distances)
                if idx != -1 and idx < len(self.chunks)
            ])
        return results
    
    def keyword_search(self, keywords: List[str], k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """
        Fallback keyword-based search when vector search fails.
        Searches through chunk content and metadata for keyword matches.
        
        Args:
            keywords: List of keywords to search for
            k: Maximum number of results to return
            
        Returns:
            List of (DocumentChunk, relevance_score) tuples
        """
        if not self.chunks:
            logger.warning("No chunks available for keyword search")
            return []
        
        logger.info(f"Performing keyword search with {len(keywords)} keywords: {keywords}")
        
        if self._lowered is None or len(self._lowered) != len(self.chunks):
            self._lowered = [
                (
                    chunk.content.lower(),
                    (chunk.metadata or {}).get('file', '').lower(),
                    (chunk.metadata or {}).get('source', '').lower(),
                )
                for chunk in self.chunks
            ]

        keywords_lower = [keyword.lower() for keyword in keywords]
        scored_chunks = []

        for chunk, (content_lower, file_name, source_path) in zip(self.chunks, self._lowered):
            score = 0.0
            
            # Search in content
            for keyword_lower in keywords_lower:
                # Count occurrences (with diminishing returns)
                occurrences = content_lower.count(keyword_lower)
                if occurrences > 0:
                    # Score: 1.0 for first match, 0.5 for second, 0.25 for third, etc.
                    score += min(1.0, 1.0 / (1 + occurrences * 0.5))
            
            # Boost score if keywords appear in metadata
            for keyword_lower in keywords_lower:
                if keyword_lower in file_name:
                    score += 0.3  # Boost for file name match
                if keyword_lower in source_path:
                    score += 0.2  # Boost for source path match
            
            if score > 0:
                scored_chunks.append((chunk, score))
        
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k with normalized scores (0-1 range)
        results = scored_chunks[:k]
        
        if results:
            max_score = max(score for _, score in results)
            # Normalize scores to 0-1 range
            results = [(chunk, score / max_score) for chunk, score in results]
            logger.info(f"Keyword search returned {len(results)} results (max score: {max_score:.3f})")
        else:
            logger.warning("Keyword search returned no results")
        
        return results

    def save_index(self):
        if self.index:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.index_path) if os.path.dirname(self.index_path) else '.', exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            # Save chunks metadata separately
            with open(self.index_path + ".meta", "wb") as f:
                pickle.dump(self.chunks, f)
            logger.debug(f"Index saved to {self.index_path}")

    def load_index(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            if os.path.exists(self.index_path + ".meta"):
                with open(self.index_path + ".meta", "rb") as f:
                    self.chunks = pickle.load(f)
            self._lowered = None
            logger.info(f"Loaded index from {self.index_path} with {len(self.chunks)} documents")
        else:
            self.index = self._create_new_index()
            self.chunks = []
            logger.info("No existing index found, created new index")

    def clear_index(self):
        self.index = self._create_new_index()
        self.chunks = []
        self._lowered = None
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.index_path + ".meta"):
            os.remove(self.index_path + ".meta")
        logger.info("Index cleared")
    
    def count(self) -> int:
        """
        Return the total number of documents in the vector store.
        """
        return len(self.chunks)

vector_store = VectorStore()
