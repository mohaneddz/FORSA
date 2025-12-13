import logging
from typing import List, Dict, Any
from config import config

logger = logging.getLogger(__name__)

class ChunkingService:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        logger.info(f"ChunkingService initialized with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with overlap.
        """
        if metadata is None:
            metadata = {}
        
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'metadata': {**metadata, 'chunk_index': len(chunks)}
            })
        
        return chunks

chunking_service = ChunkingService()
