#!/usr/bin/env python3
"""
QA Micro-Chunks Generator

Creates micro-chunks for Q/A pairs to improve definition retrieval.
These chunks are indexed alongside full-file chunks with metadata tagging.

Usage:
    python create_qa_chunks.py [--input questions.json] [--output gen/]
    
The script:
1. Reads Q/A pairs from a JSON file (common/questions.json by default)
2. Creates micro-chunks in format: "Q: <question>\nA: <answer>"
3. Saves chunks with metadata: chunk_type="qa", original chunk_type="fullfile"
4. Re-indexes all chunks (both qa and fullfile) with proper metadata
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from schemas import DocumentChunk
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_qa_pairs(file_path: Path) -> List[Dict[str, Any]]:
    """Load Q/A pairs from JSON file"""
    if not file_path.exists():
        logger.error(f"Questions file not found: {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle nested structure: {"category": {"q_id": "question text"}}
    qa_pairs = []
    
    if isinstance(data, dict):
        # Check if it's the nested structure from common/questions.json
        for category, questions in data.items():
            if isinstance(questions, dict):
                for q_id, question_text in questions.items():
                    qa_pairs.append({
                        "category": category,
                        "question_id": q_id,
                        "question": question_text,
                        "answer": None  # Will be filled if we have answers
                    })
            else:
                # Simple list format
                qa_pairs.append({
                    "category": "general",
                    "question_id": str(len(qa_pairs) + 1),
                    "question": questions,
                    "answer": None
                })
    
    logger.info(f"Loaded {len(qa_pairs)} Q/A pairs from {file_path}")
    return qa_pairs


def extract_qa_from_chunks(gen_dir: Path) -> List[Dict[str, Any]]:
    """
    Extract Q/A pairs from existing markdown chunks.
    Looks for patterns like "Q: ... A: ..." or "السؤال: ... الجواب: ..."
    """
    import re
    
    qa_pairs = []
    chunk_files = sorted(gen_dir.glob("*.md"))
    
    # Patterns to detect Q/A pairs in both languages
    patterns = [
        (r'Q:\s*(.+?)\s*A:\s*(.+?)(?=\n\n|\n?Q:|$)', 'en'),  # English
        (r'السؤال:\s*(.+?)\s*(?:الجواب|الإجابة):\s*(.+?)(?=\n\n|السؤال:|$)', 'ar'),  # Arabic
        (r'س:\s*(.+?)\s*ج:\s*(.+?)(?=\n\n|س:|$)', 'ar'),  # Arabic abbreviated
    ]
    
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try each pattern
            for pattern, lang in patterns:
                matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
                
                for question, answer in matches:
                    qa_pairs.append({
                        "category": "extracted",
                        "question_id": f"{chunk_file.stem}_{len(qa_pairs)+1}",
                        "question": question.strip(),
                        "answer": answer.strip(),
                        "source_file": chunk_file.name,
                        "language": lang
                    })
        
        except Exception as e:
            logger.warning(f"Failed to extract from {chunk_file}: {e}")
    
    logger.info(f"Extracted {len(qa_pairs)} Q/A pairs from chunks")
    return qa_pairs


def create_qa_chunk_content(qa_pair: Dict[str, Any]) -> str:
    """
    Create chunk content from Q/A pair.
    Format: Q: <question>\nA: <answer>
    """
    question = qa_pair.get("question", "").strip()
    answer = qa_pair.get("answer", "").strip()
    
    if not question:
        return ""
    
    # If we don't have an answer, create a question-only chunk
    if not answer or answer.lower() in ["none", "null", "n/a"]:
        return f"Q: {question}"
    
    return f"Q: {question}\n\nA: {answer}"


def create_qa_chunks(qa_pairs: List[Dict[str, Any]], output_dir: Path) -> int:
    """
    Create QA micro-chunk files with proper metadata.
    Returns the number of chunks created.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    chunks_created = 0
    
    for qa_pair in qa_pairs:
        content = create_qa_chunk_content(qa_pair)
        
        if not content or len(content) < 20:
            continue
        
        # Create metadata
        metadata = {
            "chunk_type": "qa",
            "category": qa_pair.get("category", "general"),
            "question_id": qa_pair.get("question_id", "unknown"),
            "source_file": qa_pair.get("source_file", "qa_pairs"),
            "language": qa_pair.get("language", "unknown"),
            "type": "qa_micro_chunk"
        }
        
        # Create filename
        safe_id = qa_pair.get("question_id", str(chunks_created)).replace("/", "_")
        filename = f"QA_{safe_id}.md"
        output_path = output_dir / filename
        
        # Write chunk with metadata comment
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"<!-- {json.dumps(metadata, ensure_ascii=False)} -->\n")
            f.write(content)
        
        chunks_created += 1
    
    logger.info(f"✓ Created {chunks_created} QA micro-chunks in {output_dir}")
    return chunks_created


def reindex_all_chunks(gen_dir: Path):
    """
    Rebuild FAISS index with both fullfile and qa chunks.
    Tags existing chunks with chunk_type="fullfile" if not tagged.
    """
    from services.vector_store import vector_store
    from services.embedding_service import embedding_service
    
    logger.info("\nRebuilding FAISS index with all chunks (fullfile + qa)...")
    
    chunk_files = sorted(gen_dir.glob("*.md"))
    
    if not chunk_files:
        logger.error("No chunk files found!")
        return
    
    # Clear existing index
    vector_store.clear_index()
    
    all_chunks = []
    all_texts = []
    qa_count = 0
    fullfile_count = 0
    
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata from first line (if JSON comment)
            lines = content.split('\n', 1)
            metadata = {}
            chunk_content = content
            
            if lines[0].strip().startswith('<!--') and lines[0].strip().endswith('-->'):
                try:
                    metadata_str = lines[0].strip()[4:-3].strip()
                    metadata = json.loads(metadata_str)
                    chunk_content = lines[1] if len(lines) > 1 else ""
                except:
                    pass
            
            # Tag with chunk_type if not present
            if "chunk_type" not in metadata:
                metadata["chunk_type"] = "fullfile"
            
            if metadata.get("chunk_type") == "qa":
                qa_count += 1
            else:
                fullfile_count += 1
            
            chunk = DocumentChunk(
                content=chunk_content.strip(),
                metadata=metadata
            )
            
            all_chunks.append(chunk)
            all_texts.append(chunk.content)
            
        except Exception as e:
            logger.warning(f"Failed to process {chunk_file}: {e}")
    
    logger.info(f"Processing {len(all_chunks)} chunks ({qa_count} qa + {fullfile_count} fullfile)...")
    
    # Generate embeddings in batches
    batch_size = 100
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i+batch_size]
        batch_chunks = all_chunks[i:i+batch_size]
        
        embeddings = embedding_service.generate_embeddings(batch_texts)
        vector_store.add_documents(batch_chunks, embeddings)
        
        logger.info(f"  Indexed {min(i+batch_size, len(all_texts))}/{len(all_texts)} chunks")
    
    logger.info(f"✓ Index rebuilt with {vector_store.count()} total documents")
    logger.info(f"  - {qa_count} QA micro-chunks")
    logger.info(f"  - {fullfile_count} full-file chunks")


def main():
    parser = argparse.ArgumentParser(
        description="Create QA micro-chunks for better definition retrieval"
    )
    parser.add_argument(
        '--input',
        default='common/questions.json',
        help='Input JSON file with Q/A pairs (default: common/questions.json)'
    )
    parser.add_argument(
        '--output',
        default='gen/',
        help='Output directory for chunk files (default: gen/)'
    )
    parser.add_argument(
        '--extract-from-chunks',
        action='store_true',
        help='Extract Q/A pairs from existing chunks instead of JSON file'
    )
    parser.add_argument(
        '--no-reindex',
        action='store_true',
        help='Do not rebuild FAISS index after creating chunks'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    logger.info("=" * 80)
    logger.info("QA Micro-Chunks Generator")
    logger.info("=" * 80)
    
    # Load Q/A pairs
    if args.extract_from_chunks:
        logger.info("\n[1/2] Extracting Q/A pairs from existing chunks...")
        qa_pairs = extract_qa_from_chunks(output_dir)
    else:
        logger.info(f"\n[1/2] Loading Q/A pairs from {args.input}...")
        qa_pairs = load_qa_pairs(Path(args.input))
    
    if not qa_pairs:
        logger.error("No Q/A pairs found. Exiting.")
        return
    
    # Create QA chunks
    logger.info(f"\n[2/2] Creating QA micro-chunks...")
    chunks_created = create_qa_chunks(qa_pairs, output_dir)
    
    if chunks_created == 0:
        logger.warning("No chunks were created!")
        return
    
    # Reindex
    if not args.no_reindex:
        logger.info("\n[3/3] Rebuilding FAISS index...")
        reindex_all_chunks(output_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ QA micro-chunks creation complete!")
    logger.info("=" * 80)
    logger.info(f"\nCreated {chunks_created} QA chunks in {output_dir}")
    
    if not args.no_reindex:
        logger.info("Index has been rebuilt with both QA and full-file chunks")
    else:
        logger.info("Remember to rebuild the index: python rebuild_index.py --keep-chunks")


if __name__ == "__main__":
    main()
