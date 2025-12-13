import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import fitz  # PyMuPDF
from docx import Document
import markdown
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def extract_text(self, file_path: Path) -> Optional[str]:
        """
        Extract text from PDF, DOCX, or Markdown files.
        """
        try:
            suffix = file_path.suffix.lower()
            
            if suffix == '.pdf':
                return self._extract_from_pdf(file_path)
            elif suffix == '.docx':
                return self._extract_from_docx(file_path)
            elif suffix in ['.md', '.markdown']:
                return self._extract_from_markdown(file_path)
            else:
                logger.warning(f"Unsupported file type: {suffix}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return None
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """
        Extract text from PDF using PyMuPDF.
        """
        text = ""
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            raise
        
        return text
    
    def extract_text_with_pages(self, file_path: Path) -> Optional[List[Dict[str, Any]]]:
        """
        Extract text from PDF with page number tracking.
        Returns list of {"page": page_num, "text": text_content}
        """
        try:
            suffix = file_path.suffix.lower()
            
            if suffix == '.pdf':
                return self._extract_from_pdf_with_pages(file_path)
            else:
                # For non-PDF files, return single entry without page number
                text = self.extract_text(file_path)
                if text:
                    return [{"page": None, "text": text}]
                return None
                
        except Exception as e:
            logger.error(f"Error extracting text with pages from {file_path}: {e}")
            return None
    
    def _extract_from_pdf_with_pages(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with page numbers using PyMuPDF.
        """
        pages_data = []
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():  # Only include pages with content
                    pages_data.append({
                        "page": page_num + 1,  # 1-indexed for user-friendliness
                        "text": text
                    })
            doc.close()
        except Exception as e:
            logger.error(f"Error reading PDF with pages {file_path}: {e}")
            raise
        
        return pages_data
    
    def _extract_from_docx(self, file_path: Path) -> str:
        """
        Extract text from DOCX using python-docx.
        """
        text = ""
        try:
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {e}")
            raise
        
        return text
    
    def _extract_from_markdown(self, file_path: Path) -> str:
        """
        Extract text from Markdown files.
        Converts markdown to HTML then extracts plain text.
        """
        text = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # Convert markdown to HTML
            html = markdown.markdown(md_content)
            
            # Extract plain text from HTML
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text(separator='\n')
            
        except Exception as e:
            logger.error(f"Error reading Markdown {file_path}: {e}")
            raise
        
        return text

document_processor = DocumentProcessor()
