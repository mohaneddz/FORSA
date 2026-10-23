"""
Builds the FAISS index from the chunk-per-file Markdown in gen/.

Each gen/*.md file is one chunk whose first line is
    # SOURCE_FILE: <path of the original document>
Used by rebuild_index.py and POST /api/embeddings/gen, both of which replace
the whole index (appending used to leave every chunk in the index 3 times).
"""
import logging
import re
from pathlib import Path
from typing import List

from schemas import DocumentChunk

logger = logging.getLogger(__name__)

MIN_CHUNK_CHARS = 80
SOURCE_HEADER = re.compile(r"^#\s*SOURCE_FILE:\s*(.+?)\s*$", flags=re.MULTILINE)
# preprocess.py glued this folder name onto every source path
BROKEN_PATH_PREFIX = "Argumentaire offre Idoom Medium Business acquisition et modernisation V.2"


def clean_source_name(source_path: str) -> str:
    name = source_path.replace("\\", "/").rsplit("/", 1)[-1]
    if name.startswith(BROKEN_PATH_PREFIX) and len(name) > len(BROKEN_PATH_PREFIX):
        name = name[len(BROKEN_PATH_PREFIX):]
    return name.strip(" -")


def detect_language(text: str) -> str:
    has_ar = bool(re.search(r"[؀-ۿ]", text))
    has_fr = bool(re.search(r"[A-Za-zÀ-ÿ]", text))
    if has_ar and has_fr:
        return "mixed"
    return "ar" if has_ar else "fr"


def load_markdown_chunks(gen_dir: Path) -> List[DocumentChunk]:
    """Read gen/*.md into chunks, dropping near-empty files and exact duplicates."""
    chunks: List[DocumentChunk] = []
    seen_texts = set()
    for md_file in sorted(Path(gen_dir).glob("*.md")):
        if md_file.name == "README.md":
            continue
        raw = md_file.read_text(encoding="utf-8").strip()
        match = SOURCE_HEADER.search(raw)
        source_name = clean_source_name(match.group(1)) if match else md_file.stem
        text = SOURCE_HEADER.sub("", raw, count=1).strip()
        if len(text) < MIN_CHUNK_CHARS:
            continue
        dedupe_key = re.sub(r"\s+", " ", text).lower()
        if dedupe_key in seen_texts:
            continue
        seen_texts.add(dedupe_key)
        chunks.append(DocumentChunk(
            content=text,
            metadata={
                "file": source_name,
                "doc": re.sub(r"__Chunk\d+$", "", md_file.stem),
                "chunk_id": md_file.stem,
                "language": detect_language(text),
            },
        ))
    return chunks


def rebuild_index(gen_dir: Path = Path("gen"), batch_size: int = 32) -> int:
    """Replace the FAISS index with embeddings of every chunk in gen_dir. Returns the chunk count."""
    import numpy as np
    from services.embedding_service import embedding_service
    from services.vector_store import vector_store

    chunks = load_markdown_chunks(gen_dir)
    if not chunks:
        raise FileNotFoundError(f"No usable Markdown chunks in {gen_dir}/ (run preprocess.py first)")
    logger.info(f"Embedding {len(chunks)} unique chunks from {gen_dir}/")

    embeddings = np.vstack([
        embedding_service.generate_embeddings([c.content for c in chunks[i:i + batch_size]])
        for i in range(0, len(chunks), batch_size)
    ])
    vector_store.clear_index()
    vector_store.add_documents(chunks, embeddings)
    return len(chunks)
