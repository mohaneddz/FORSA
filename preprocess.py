"""
DeepSeek-based Markdown Preprocessing for RAG
Converts PDFs/DOCX into clean, semantically-chunked Markdown files.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from services.document_processor import DocumentProcessor
from services.llm_service import LLMService
from prompts import get_markdown_chunk_prompt
from config import config

logger = logging.getLogger(__name__)

class DeepSeekChunker:
    """
    Preprocesses documents using DeepSeek to create clean Markdown chunks.
    
    Workflow:
    1. Extract raw text from PDF/DOCX (with page markers for PDFs)
    2. Send to DeepSeek with normalization + chunking prompt
    3. DeepSeek returns clean Markdown with <---CHUNK---> delimiters
    4. Split by delimiters and write individual chunk files to gen/
    """
    
    CHUNK_DELIMITER = "<---CHUNK--->"
    MIN_CHUNK_LENGTH = 120  # Minimum characters for a valid chunk
    
    def __init__(self, model_name: str = None):
        """
        Initialize the DeepSeek chunker.
        
        Args:
            model_name: DeepSeek model to use (defaults to config.MODEL_NAME)
        """
        self.model_name = model_name or config.MODEL_NAME
        self.document_processor = DocumentProcessor()
        self.llm_service = LLMService()
        logger.info(f"Initialized DeepSeekChunker with model: {self.model_name}")
    
    def chunk_document(self, file_path: Path, output_folder: Path) -> Dict[str, Any]:
        """
        Process a single document and write chunks to output folder.
        
        Args:
            file_path: Path to PDF or DOCX file
            output_folder: Directory to write chunk files
            
        Returns:
            Dict with processing stats: {
                "success": bool,
                "file": str,
                "chunks_created": int,
                "error": str (if failed)
            }
        """
        logger.info(f"Processing document: {file_path}")
        
        try:
            # Step 1: Extract raw text
            raw_text = self._extract_raw_text(file_path)
            if not raw_text or len(raw_text.strip()) < 50:
                logger.warning(f"Insufficient text extracted from {file_path}")
                return {
                    "success": False,
                    "file": str(file_path),
                    "chunks_created": 0,
                    "error": "Insufficient text extracted"
                }
            
            logger.info(f"Extracted {len(raw_text)} characters from {file_path.name}")
            
            # Step 2: Call DeepSeek for Markdown normalization + chunking
            markdown_with_delimiters = self._call_deepseek_markdown(
                file_name=file_path.name,
                file_path=str(file_path),
                raw_text=raw_text
            )
            
            if not markdown_with_delimiters:
                logger.error(f"DeepSeek returned empty response for {file_path}")
                return {
                    "success": False,
                    "file": str(file_path),
                    "chunks_created": 0,
                    "error": "Empty response from DeepSeek"
                }
            
            # Step 3: Split into chunks
            chunks = self._split_chunks(markdown_with_delimiters)
            
            if not chunks:
                logger.warning(f"No valid chunks extracted from {file_path}")
                return {
                    "success": False,
                    "file": str(file_path),
                    "chunks_created": 0,
                    "error": "No valid chunks extracted"
                }
            
            # Step 4: Write chunk files
            chunks_written = self._write_chunks(
                chunks=chunks,
                source_file=file_path.name,
                output_folder=output_folder
            )
            
            logger.info(f"Successfully created {chunks_written} chunks for {file_path.name}")
            
            return {
                "success": True,
                "file": str(file_path),
                "chunks_created": chunks_written,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            return {
                "success": False,
                "file": str(file_path),
                "chunks_created": 0,
                "error": str(e)
            }
    
    def chunk_folder(
        self, 
        input_folder: Path, 
        output_folder: Path, 
        extensions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Process all documents in a folder.
        
        Args:
            input_folder: Folder containing PDF/DOCX files
            output_folder: Folder to write chunk files
            extensions: List of file extensions to process (default: ['.pdf', '.docx'])
            
        Returns:
            Dict with batch processing stats: {
                "total_files": int,
                "successful": int,
                "failed": int,
                "files_processed": List[Dict]
            }
        """
        if extensions is None:
            extensions = ['.pdf', '.docx']
        
        # Ensure output folder exists
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Find all matching files
        files_to_process = []
        for ext in extensions:
            files_to_process.extend(input_folder.rglob(f'*{ext}'))
        
        logger.info(f"Found {len(files_to_process)} files to process in {input_folder}")
        
        results = []
        successful = 0
        failed = 0
        
        for file_path in files_to_process:
            result = self.chunk_document(file_path, output_folder)
            results.append(result)
            
            if result["success"]:
                successful += 1
            else:
                failed += 1
        
        summary = {
            "total_files": len(files_to_process),
            "successful": successful,
            "failed": failed,
            "files_processed": results
        }
        
        logger.info(f"Batch processing complete: {successful}/{len(files_to_process)} successful")
        
        return summary
    
    def _extract_raw_text(self, file_path: Path) -> str:
        """
        Extract raw text from PDF or DOCX.
        For PDFs: includes page markers (--- PAGE N ---)
        For DOCX: sequential text extraction
        
        Args:
            file_path: Path to document
            
        Returns:
            Raw extracted text with page markers (PDFs) or sequential text (DOCX)
        """
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            # Extract with page tracking
            pages_data = self.document_processor.extract_text_with_pages(file_path)
            
            if not pages_data:
                return ""
            
            # Format with page markers
            formatted_text = ""
            for page_data in pages_data:
                page_num = page_data.get("page", 0)
                text = page_data.get("text", "")
                formatted_text += f"--- PAGE {page_num + 1} ---\n{text}\n\n"
            
            return formatted_text
            
        elif suffix == '.docx':
            # Extract sequential text
            text = self.document_processor.extract_text(file_path)
            return text or ""
            
        else:
            logger.warning(f"Unsupported file extension: {suffix}")
            return ""
    
    def _call_deepseek_markdown(
        self, 
        file_name: str, 
        file_path: str, 
        raw_text: str
    ) -> str:
        """
        Call DeepSeek API to normalize raw text into clean Markdown with chunk delimiters.
        
        Args:
            file_name: Original filename
            file_path: Full file path
            raw_text: Raw extracted text (with page markers for PDFs)
            
        Returns:
            Clean Markdown with <---CHUNK---> delimiters
        """
        logger.info(f"Calling DeepSeek for markdown normalization: {file_name}")
        
        # Build prompt
        user_prompt = get_markdown_chunk_prompt(file_name, file_path, raw_text)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You convert Algérie Télécom agent documents into clean Markdown for RAG. "
                    "Output MARKDOWN ONLY. No JSON. No extra text. No code fences. "
                    "Preserve ALL numbers exactly and never invent missing info."
                )
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        
        # Call LLM (json_mode=False for markdown output)
        try:
            markdown_output = self.llm_service.chat_completion(
                messages=messages,
                temperature=0.1,  # Low temperature for consistent formatting
                json_mode=False,  # CRITICAL: We want Markdown, not JSON
                max_retries=2
            )
            
            return markdown_output.strip()
            
        except Exception as e:
            logger.error(f"DeepSeek API call failed: {e}")
            raise
    
    def _split_chunks(self, markdown_text: str) -> List[str]:
        """
        Split markdown text by <---CHUNK---> delimiter.
        Validates chunk sizes and merges tiny chunks.
        
        Args:
            markdown_text: Markdown with chunk delimiters
            
        Returns:
            List of chunk strings (cleaned, validated)
        """
        # Split by delimiter
        raw_chunks = markdown_text.split(self.CHUNK_DELIMITER)
        
        # Clean and validate chunks
        validated_chunks = []
        buffer_chunk = ""
        
        for i, chunk in enumerate(raw_chunks):
            chunk = chunk.strip()
            
            # Skip empty chunks
            if not chunk:
                continue
            
            # If chunk is too small, merge with next
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                buffer_chunk += "\n\n" + chunk if buffer_chunk else chunk
                logger.debug(f"Chunk {i+1} too small ({len(chunk)} chars), buffering for merge")
                continue
            
            # If we have buffered content, prepend it
            if buffer_chunk:
                chunk = buffer_chunk + "\n\n" + chunk
                buffer_chunk = ""
            
            validated_chunks.append(chunk)
        
        # Add any remaining buffered content as final chunk
        if buffer_chunk and len(buffer_chunk) >= self.MIN_CHUNK_LENGTH:
            validated_chunks.append(buffer_chunk)
        elif buffer_chunk and validated_chunks:
            # Merge with last chunk if too small
            validated_chunks[-1] += "\n\n" + buffer_chunk
        
        logger.info(f"Split into {len(validated_chunks)} validated chunks")
        
        return validated_chunks
    
    def _write_chunks(
        self, 
        chunks: List[str], 
        source_file: str, 
        output_folder: Path
    ) -> int:
        """
        Write chunks as individual .md files with SOURCE_FILE header.
        
        Args:
            chunks: List of chunk text
            source_file: Original filename
            output_folder: Output directory
            
        Returns:
            Number of chunks written
        """
        # Sanitize source filename for use in chunk filenames
        sanitized_name = self._sanitize_filename(source_file)
        
        chunks_written = 0
        
        for i, chunk in enumerate(chunks, start=1):
            # Generate chunk filename
            chunk_filename = f"{sanitized_name}__Chunk{i:03d}.md"
            chunk_path = output_folder / chunk_filename
            
            # Prepare chunk content with SOURCE_FILE header
            chunk_content = f"# SOURCE_FILE: Offers/Argumentaires des offres AR FR/Argumentaire offre Idoom Medium Business acquisition et modernisation V.2{source_file}\n\n{chunk}"
            
            # Write to file
            try:
                chunk_path.write_text(chunk_content, encoding='utf-8')
                chunks_written += 1
                logger.debug(f"Wrote chunk {i} to {chunk_filename}")
            except Exception as e:
                logger.error(f"Failed to write chunk {i} to {chunk_filename}: {e}")
        
        return chunks_written
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Sanitize filename for use in chunk filenames.
        Removes extension and invalid characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename suitable for chunk naming
        """
        # Remove extension
        name_without_ext = Path(filename).stem
        
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^\w\-.]', '_', name_without_ext)
        
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        return sanitized

