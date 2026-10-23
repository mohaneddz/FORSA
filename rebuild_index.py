#!/usr/bin/env python3
"""
Rebuild FAISS Index Script

This script completely rebuilds the FAISS index and embeddings.
Use when:
- Embedding model changes
- EMBEDDING_DIMENSION changes
- Index appears corrupted
- Similarity metric changes

Usage:
    python rebuild_index.py [--keep-chunks]
    
Arguments:
    --keep-chunks: Keep generated markdown chunks, only rebuild index/embeddings
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def delete_faiss_index():
    """Delete FAISS index files"""
    files_to_delete = [
        "db/faiss_index.bin",
        "db/faiss_index.bin.meta"
    ]
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"✓ Deleted: {file_path}")
        else:
            logger.info(f"  (not found): {file_path}")


def delete_cached_embeddings():
    """Delete cached embedding files"""
    cache_patterns = [
        "cache/raw/*.pkl",
        "cache/*.npy",
        "cache/*.pkl"
    ]
    
    deleted_count = 0
    for pattern in cache_patterns:
        for file_path in Path(".").glob(pattern):
            file_path.unlink()
            deleted_count += 1
            logger.info(f"✓ Deleted cache: {file_path}")
    
    if deleted_count == 0:
        logger.info("  (no cached embeddings found)")


def delete_generated_chunks():
    """Delete generated markdown chunks"""
    gen_dir = Path("gen")
    
    if gen_dir.exists() and gen_dir.is_dir():
        file_count = len(list(gen_dir.glob("*.md")))
        shutil.rmtree(gen_dir)
        gen_dir.mkdir(exist_ok=True)
        logger.info(f"✓ Deleted {file_count} generated chunk files from gen/")
    else:
        logger.info("  (gen/ directory not found or empty)")


def rebuild_index_from_chunks():
    """Rebuild FAISS index from existing chunks"""
    logger.info("\nRebuilding FAISS index from gen/ chunks...")

    # Import here to avoid loading models unnecessarily
    from services.index_builder import rebuild_index

    try:
        count = rebuild_index(Path("gen"))
    except FileNotFoundError as e:
        logger.error(f"ERROR: {e}")
        return False

    logger.info(f"✓ Index rebuilt with {count} documents")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild FAISS index and embeddings"
    )
    parser.add_argument(
        '--keep-chunks',
        action='store_true',
        help='Keep generated chunks, only rebuild index/embeddings'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("FAISS Index Rebuild Script")
    logger.info("=" * 80)
    
    # Step 1: Delete FAISS index
    logger.info("\n[1/4] Deleting FAISS index files...")
    delete_faiss_index()
    
    # Step 2: Delete cached embeddings
    logger.info("\n[2/4] Deleting cached embeddings...")
    delete_cached_embeddings()
    
    # Step 3: Delete generated chunks (optional)
    if not args.keep_chunks:
        logger.info("\n[3/4] Deleting generated chunks...")
        delete_generated_chunks()
        logger.info("\nTo regenerate chunks, run: python preprocess.py")
    else:
        logger.info("\n[3/4] Keeping generated chunks (--keep-chunks flag)")
    
    # Step 4: Rebuild index if chunks exist
    logger.info("\n[4/4] Rebuilding index from chunks...")
    if args.keep_chunks:
        success = rebuild_index_from_chunks()
        if not success:
            logger.error("\nFailed to rebuild index. Please check the logs above.")
            sys.exit(1)
    else:
        logger.info("Skipped (no chunks available)")
        logger.info("Next step: python preprocess.py")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ Rebuild complete!")
    logger.info("=" * 80)
    
    if not args.keep_chunks:
        logger.info("\nNext steps:")
        logger.info("  1. Run preprocessing: python preprocess.py")
        logger.info("  2. Start the server: python main.py")
    else:
        logger.info("\nIndex is ready. Start the server: python main.py")


if __name__ == "__main__":
    main()
