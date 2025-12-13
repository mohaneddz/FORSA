import os
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF
from docx import Document
import logging

logger = logging.getLogger(__name__)

def count_files_with_images(data_folder: str = "data") -> dict:
    """
    Iterate over data folder and count how many files contain one or more images.
    
    Args:
        data_folder: Path to the data folder
        
    Returns:
        Dictionary with statistics about files with images
    """
    stats = {
        "total_files": 0,
        "files_with_images": 0,
        "files_without_images": 0,
        "pdf_files": 0,
        "docx_files": 0,
        "other_files": 0,
        "total_images": 0,
        "files_details": []
    }
    
    data_path = Path(data_folder)
    
    if not data_path.exists():
        logger.error(f"Data folder not found: {data_folder}")
        return stats
    
    # Iterate through all files in data folder and subfolders
    for file_path in data_path.rglob("*"):
        if not file_path.is_file():
            continue
        
        stats["total_files"] += 1
        file_ext = file_path.suffix.lower()
        
        image_count = 0
        
        try:
            if file_ext == ".pdf":
                stats["pdf_files"] += 1
                image_count = count_images_in_pdf(file_path)
            elif file_ext in [".docx", ".doc"]:
                stats["docx_files"] += 1
                image_count = count_images_in_docx(file_path)
            else:
                stats["other_files"] += 1
                continue
            
            if image_count > 0:
                stats["files_with_images"] += 1
                stats["total_images"] += image_count
                stats["files_details"].append({
                    "file": str(file_path),
                    "type": file_ext,
                    "image_count": image_count
                })
            else:
                stats["files_without_images"] += 1
                
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
            continue
    
    return stats


def count_images_in_pdf(pdf_path: Path) -> int:
    """Count images in a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        image_count = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images(full=True)
            image_count += len(images)
        
        doc.close()
        return image_count
    except Exception as e:
        logger.warning(f"Error counting images in PDF {pdf_path}: {e}")
        return 0


def count_images_in_docx(docx_path: Path) -> int:
    """Count images in a DOCX file."""
    try:
        doc = Document(docx_path)
        
        # Count inline shapes (images)
        image_count = 0
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                image_count += 1
        
        return image_count
    except Exception as e:
        logger.warning(f"Error counting images in DOCX {docx_path}: {e}")
        return 0


def print_image_statistics(data_folder: str = "data"):
    """Print formatted statistics about images in files."""
    stats = count_files_with_images(data_folder)
    
    print("=" * 80)
    print("IMAGE STATISTICS")
    print("=" * 80)
    print(f"Total files processed: {stats['total_files']}")
    print(f"  - PDF files: {stats['pdf_files']}")
    print(f"  - DOCX files: {stats['docx_files']}")
    print(f"  - Other files: {stats['other_files']}")
    print()
    print(f"Files with images: {stats['files_with_images']}")
    print(f"Files without images: {stats['files_without_images']}")
    print(f"Total images found: {stats['total_images']}")
    print()
    
    if stats['files_details']:
        print("Files containing images:")
        print("-" * 80)
        for detail in stats['files_details']:
            print(f"  {detail['file']}")
            print(f"    Type: {detail['type']}, Images: {detail['image_count']}")
    
    print("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_image_statistics()
