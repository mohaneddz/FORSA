"""
Example: Integrate Gemini-preprocessed chunks into the RAG pipeline.

This script demonstrates how to:
1. Load chunks from Gemini preprocessing
2. Convert them to embeddings
3. Index them in FAISS
4. Query the system
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from services.embedding_service import embedding_service
from services.vector_store import vector_store
from schemas import DocumentChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_chunks_from_json(json_file: Path) -> List[Dict[str, Any]]:
    """Load chunks from a Gemini-preprocessed JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    doc_metadata = data.get('doc', {})
    chunks = data.get('chunks', [])
    
    # Combine doc metadata with chunk metadata
    enriched_chunks = []
    for chunk in chunks:
        enriched_chunk = {
            'text': chunk.get('text_md', ''),
            'metadata': {
                'file': doc_metadata.get('file_name'),
                'path': doc_metadata.get('file_path'),
                'language': doc_metadata.get('language'),
                'doc_date': doc_metadata.get('doc_date'),
                'version': doc_metadata.get('version'),
                'chunk_id': chunk.get('chunk_id'),
                'type': chunk.get('type'),
                'section_title': chunk.get('section_title'),
                'page': chunk.get('page'),
                'keywords': chunk.get('keywords', [])
            }
        }
        enriched_chunks.append(enriched_chunk)
    
    return enriched_chunks


def load_all_chunks_from_folder(chunks_folder: Path) -> List[Dict[str, Any]]:
    """Load all chunks from a folder of JSON files."""
    all_chunks = []
    json_files = list(chunks_folder.glob("*_chunks.json"))
    
    logger.info(f"Found {len(json_files)} chunk files in {chunks_folder}")
    
    for json_file in json_files:
        logger.info(f"Loading chunks from: {json_file.name}")
        chunks = load_chunks_from_json(json_file)
        all_chunks.extend(chunks)
        logger.info(f"  Loaded {len(chunks)} chunks")
    
    logger.info(f"Total chunks loaded: {len(all_chunks)}")
    return all_chunks


def index_chunks_in_faiss(chunks: List[Dict[str, Any]]):
    """Generate embeddings and index chunks in FAISS."""
    logger.info("Generating embeddings for chunks...")
    
    # Extract text from chunks
    texts = [chunk['text'] for chunk in chunks]
    
    # Generate embeddings
    embeddings = embedding_service.generate_embeddings(texts)
    logger.info(f"Generated {len(embeddings)} embeddings")
    
    # Create DocumentChunk objects
    document_chunks = []
    for chunk in chunks:
        document_chunks.append(DocumentChunk(
            content=chunk['text'],
            metadata=chunk['metadata']
        ))
    
    # Add to FAISS vector store
    logger.info("Indexing in FAISS...")
    vector_store.add_documents(document_chunks, embeddings)
    
    logger.info(f"Successfully indexed {len(document_chunks)} chunks")
    logger.info(f"Total documents in vector store: {vector_store.count()}")


def query_example(query: str, top_k: int = 5):
    """Example query against the indexed chunks."""
    from services.rag_service import rag_service
    
    logger.info(f"\nQuerying: {query}")
    results = rag_service.retrieve_and_rerank(query, top_k=top_k)
    
    logger.info(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"--- Result {i} (Score: {result.score:.3f}) ---")
        print(f"File: {result.chunk.metadata.get('file')}")
        print(f"Page: {result.chunk.metadata.get('page', 'N/A')}")
        print(f"Type: {result.chunk.metadata.get('type')}")
        print(f"Section: {result.chunk.metadata.get('section_title', 'N/A')}")
        print(f"Keywords: {', '.join(result.chunk.metadata.get('keywords', []))}")
        print(f"Content: {result.chunk.content[:200]}...")
        print()


def main():
    """Main workflow example."""
    print("=" * 60)
    print("Gemini Chunks → FAISS Integration Example")
    print("=" * 60)
    
    # Step 1: Load chunks
    chunks_folder = Path("data/chunks")
    
    if not chunks_folder.exists():
        print(f"\n❌ Chunks folder not found: {chunks_folder}")
        print("\nPlease run preprocessing first:")
        print("  python preprocess.py")
        return
    
    print("\n📂 Loading chunks from JSON files...")
    chunks = load_all_chunks_from_folder(chunks_folder)
    
    if not chunks:
        print("\n❌ No chunks found. Run preprocessing first:")
        print("  python preprocess.py")
        return
    
    # Display chunk statistics
    print("\n📊 Chunk Statistics:")
    chunk_types = {}
    languages = {}
    
    for chunk in chunks:
        chunk_type = chunk['metadata'].get('type', 'unknown')
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        lang = chunk['metadata'].get('language', 'unknown')
        languages[lang] = languages.get(lang, 0) + 1
    
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Chunk types: {dict(chunk_types)}")
    print(f"  Languages: {dict(languages)}")
    
    # Step 2: Index in FAISS
    print("\n🔍 Indexing chunks in FAISS...")
    index_chunks_in_faiss(chunks)
    
    # Step 3: Example queries
    print("\n💬 Running example queries...")
    
    queries = [
        "Quel est le tarif de l'offre IDOOM ADSL 4 Mbps?",
        "Quels sont les documents requis pour une convention?",
        "Comment créer un nouveau client dans NGBSS?"
    ]
    
    for query in queries:
        query_example(query, top_k=3)
        print("-" * 60)
    
    print("\n✅ Integration complete!")
    print("\nNext steps:")
    print("1. Test with your own queries")
    print("2. Use the /api/chat/ endpoint for Q&A")
    print("3. Review chunk quality and adjust preprocessing if needed")


if __name__ == "__main__":
    main()
