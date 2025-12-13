from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import json
import hashlib
import re

from services.embedding_service import embedding_service
from services.vector_store import vector_store
from services.document_processor import document_processor
from services.chunking_service import chunking_service
from schemas import DocumentChunk
from utils.logging_utils import log_execution_time

router = APIRouter()
logger = logging.getLogger(__name__)


def normalize_text_for_hash(text: str) -> str:
    """Normalize text for consistent hashing: strip whitespace, fix hyphenation, lowercase."""
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Fix hyphenation at line breaks (common in PDFs)
    text = re.sub(r'-(\s+)', '', text)
    # Lowercase for case-insensitive dedup
    text = text.lower()
    return text

# Ensure folders exist
DB_FOLDER = Path("db")
RAW_FOLDER = Path("data")
GEN_FOLDER = Path("gen")

DB_FOLDER.mkdir(exist_ok=True)
GEN_FOLDER.mkdir(exist_ok=True)

class AddDocumentsRequest(BaseModel):
    documents: List[str]
    metadatas: List[Dict[str, Any]]

@router.post("/add")
@log_execution_time
async def add_documents(payload: AddDocumentsRequest):
    """
    Generate embeddings for documents and add them to the FAISS index.
    """
    if len(payload.documents) != len(payload.metadatas):
        raise HTTPException(status_code=400, detail="Number of documents and metadatas must match.")

    try:
        # Generate embeddings
        embeddings = embedding_service.generate_embeddings(payload.documents)
        
        # Create DocumentChunk objects
        chunks = []
        for i, doc in enumerate(payload.documents):
            chunks.append(DocumentChunk(
                content=doc,
                metadata=payload.metadatas[i]
            ))
        
        # Add to vector store
        vector_store.add_documents(chunks, embeddings)
        
        return {"status": "success", "added_count": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/gen")
async def generate_embeddings(file: UploadFile = File(None)):
    """
    Generate embeddings from AI-preprocessed Markdown files in gen/ folder.
    Falls back to basic chunking if gen/ folder doesn't exist.
    Stores the vector database in the db folder.
    """
    try:
        # Check if gen folder exists with preprocessed Markdown files
        if GEN_FOLDER.exists():
            md_files = list(GEN_FOLDER.glob("**/*.md"))
            if md_files:
                logger.info(f"Found {len(md_files)} preprocessed Markdown files in gen/ folder")
                return await _generate_from_markdown_chunks(md_files)
        
        # Fallback to basic chunking from data folder
        logger.info("No preprocessed files found. Using basic chunking from data/ folder")
        return await _generate_from_raw_files()
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _generate_from_markdown_chunks(md_files: List[Path]):
    """
    Generate embeddings from chunk-per-file Markdown.
    Each .md file in gen/ is considered ONE chunk.
    Expected header:
      # SOURCE_FILE: Offers/Argumentaires des offres AR FR/Argumentaire offre Idoom Medium Business acquisition et modernisation V.2original_name.pdf
    """
    import re
    import hashlib

    logger.info("Processing chunk-per-file Markdown chunks...")
    all_chunks = []

    for md_file in md_files:
        # Skip README.md
        if md_file.name == "README.md":
            continue
            
        try:
            content = md_file.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning(f"Failed to read {md_file}: {e}")
            continue

        if len(content) < 80:
            continue

        # Extract SOURCE_FILE from header if present
        m = re.search(r"^#\s*SOURCE_FILE:\s*(.+)\s*$", content, flags=re.MULTILINE)
        source_file = m.group(1).strip() if m else md_file.stem

        # Remove that header line from embedding text (optional but recommended)
        content_wo_header = re.sub(
            r"^#\s*SOURCE_FILE:.*\n+", "", content, flags=re.MULTILINE
        ).strip()

        if len(content_wo_header) < 80:
            continue

        # Optional: detect language very lightly
        has_ar = bool(re.search(r"[\u0600-\u06FF]", content_wo_header))
        has_fr = bool(re.search(r"[A-Za-zÀ-ÿ]", content_wo_header))
        language = "mixed" if has_ar and has_fr else ("ar" if has_ar else ("fr" if has_fr else "mixed"))

        # Dedup hash
        text_hash = hashlib.sha1((source_file + "||" + content_wo_header).encode("utf-8")).hexdigest()[:16]

        metadata = {
            "file": source_file,
            "path": str(md_file),
            "type": "md_chunk_file",
            "language": language,
            "chunk_id": md_file.stem,
            "text_hash": text_hash,
            "source": "deepseek_markdown_chunks",
        }

        all_chunks.append({"text": content_wo_header, "metadata": metadata})

    logger.info(f"Total Markdown chunk files loaded: {len(all_chunks)}")
    if not all_chunks:
        raise HTTPException(status_code=404, detail="No valid chunks found in preprocessed Markdown files")

    # Dedup by text_hash
    seen = set()
    deduped = []
    for c in all_chunks:
        h = c["metadata"]["text_hash"]
        if h in seen:
            continue
        seen.add(h)
        deduped.append(c)
    all_chunks = deduped
    logger.info(f"After dedup: {len(all_chunks)} chunks")

    # Generate embeddings
    texts = [c["text"] for c in all_chunks]
    embeddings = embedding_service.generate_embeddings(texts)

    # Store
    document_chunks = [DocumentChunk(content=c["text"], metadata=c["metadata"]) for c in all_chunks]
    vector_store.add_documents(document_chunks, embeddings)

    return {
        "message": "Embeddings generated from chunk-per-file Markdown files",
        "source": "deepseek_markdown_chunks",
        "files_processed": len(md_files),
        "chunks_created": len(all_chunks),
        "total_documents_in_store": vector_store.count(),
    }

async def _generate_from_raw_files():
    """Fallback: Generate embeddings using basic chunking from raw files."""
    logger.info("Starting embeddings generation from raw folder (basic chunking)")
    
    # Check if raw folder exists
    if not RAW_FOLDER.exists():
        raise HTTPException(status_code=404, detail="Raw folder not found")
    
    # Get all PDF and DOCX files from raw folder
    raw_files = []
    raw_files.extend(list(RAW_FOLDER.glob("**/*.pdf")))
    raw_files.extend(list(RAW_FOLDER.glob("**/*.docx")))
    
    if not raw_files:
        raise HTTPException(status_code=404, detail="No PDF or DOCX files found in raw folder")
    
    logger.info(f"Found {len(raw_files)} files in raw folder")
    
    # Process files and generate embeddings
    all_chunks = []
    for file_path in raw_files:
        logger.info(f"Processing file: {file_path}")
        
        # Extract text with page tracking (for PDFs)
        pages_data = document_processor.extract_text_with_pages(file_path)
        
        if not pages_data:
            logger.warning(f"No content extracted from {file_path}, skipping...")
            continue
        
        # Process each page separately to maintain page numbers
        for page_data in pages_data:
            content = page_data.get("text", "")
            page_num = page_data.get("page")
            
            if not content or not content.strip():
                continue
            
            # Split page content into chunks
            base_metadata = {
                'file': str(file_path.name),
                'path': str(file_path),
                'type': file_path.suffix,
                'source': 'basic_chunking'
            }
            
            # Add page number if available (PDFs)
            if page_num is not None:
                base_metadata['page'] = page_num
            
            chunks = chunking_service.chunk_text(content, metadata=base_metadata)
            all_chunks.extend(chunks)
    
    logger.info(f"Total chunks created: {len(all_chunks)}")
    
    # Generate embeddings for all chunks
    logger.info("Generating embeddings...")
    texts = [chunk['text'] for chunk in all_chunks]
    embeddings = embedding_service.generate_embeddings(texts)
    
    # Create DocumentChunk objects and add to FAISS vector store
    logger.info("Storing embeddings in FAISS vector database...")
    document_chunks = []
    for chunk, embedding in zip(all_chunks, embeddings):
        document_chunks.append(DocumentChunk(
            content=chunk['text'],
            metadata=chunk.get('metadata', {})
        ))
    
    # Add to FAISS vector store
    vector_store.add_documents(document_chunks, embeddings)
    
    logger.info(f"Embeddings generation complete. FAISS index updated.")
    
    return {
        "message": "Embeddings generated using basic chunking",
        "source": "basic_chunking",
        "files_processed": len(raw_files),
        "chunks_created": len(all_chunks),
        "total_documents_in_store": vector_store.count()
    }

class PreprocessRequest(BaseModel):
    input_folder: Optional[str] = "data"
    output_folder: Optional[str] = "gen"  # ✅ keep gen so /gen finds json files
    model: Optional[str] = "deepseek-v3.1"
    extensions: Optional[List[str]] = [".pdf", ".docx"]
    pages_per_batch: Optional[int] = 8

@router.post("/preprocess")
@log_execution_time
async def preprocess_documents(payload: PreprocessRequest = None):
    """
    Preprocess documents using DeepSeekChunker to generate structured Markdown chunks.
    Converts PDF/DOCX files into retrieval-ready Markdown format with semantic chunking.
    """
    try:
        # Import here to avoid circular imports
        from preprocess import DeepSeekChunker
        
        # Use defaults if no payload
        if payload is None:
            payload = PreprocessRequest()
        
        input_folder = Path(payload.input_folder)
        output_folder = Path(payload.output_folder)
        
        if not input_folder.exists():
            raise HTTPException(status_code=404, detail=f"Input folder not found: {input_folder}")
        
        logger.info(f"Starting preprocessing: {input_folder} -> {output_folder}")
        
        # Initialize DeepSeek chunker
        chunker = DeepSeekChunker(model_name=payload.model)
        
        # Process folder (using chunk_folder method from DeepSeekChunker)
        stats = chunker.chunk_folder(
            input_folder=input_folder,
            output_folder=output_folder,
            extensions=payload.extensions
        )
        
        logger.info(f"Preprocessing complete: {stats['successful']} files processed")
        
        return {
            "message": "Preprocessing completed successfully",
            "stats": stats,
            "output_folder": str(output_folder)
        }
        
    except ImportError as e:
        logger.error(f"DeepSeek API not configured: {e}")
        raise HTTPException(
            status_code=500, 
            detail="DeepSeek API not configured. Please set DEEPSEEK_TOKEN in .env.local"
        )
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
