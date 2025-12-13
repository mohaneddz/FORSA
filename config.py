import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.local")

class Config:
    DEEPSEEK_TOKEN = os.getenv("DEEPSEEK_TOKEN")
    DEEPSEEK_API_URL = os.getenv(
        "DEEPSEEK_API_URL",
        "https://api.modelarts-maas.com/v2/chat/completions"
    )
    MODEL_NAME = "deepseek-v3.1"

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = "gemini-2.5-flash"

    EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
    EMBEDDING_DIMENSION = 1024
    FAISS_INDEX_PATH = "db/faiss_index.bin"

    CHUNK_SIZE = None
    CHUNK_OVERLAP = 0

    SIMILARITY_METRIC = "cosine"
    INDEX_TYPE = "Flat"

    IVF_NLIST = 100
    IVF_NPROBE = 10

    HNSW_M = 32
    HNSW_EF_CONSTRUCTION = 40
    HNSW_EF_SEARCH = 16

    # Hybrid retrieval configuration
    TOP_K_SEMANTIC = 20
    TOP_K_LEXICAL = 20
    HYBRID_W_SEM = 0.65
    HYBRID_W_LEX = 0.35
    LEXICAL_SCORE_FLOOR = 0.0  

    SIMILARITY_THRESHOLD = 0
    MIN_SOURCES = 1
    MAX_SOURCES = 6

    CACHE_ENABLED = True
    CACHE_EXPIRATION_HOURS = 24

    DEBUG_LOG_RETRIEVAL = False
    ASSERT_EMBEDDING_DIM = True
    
    # Answer Fixer Settings
    ENABLE_ANSWER_FIXER = False  # Enable strict RAG answer post-processing
    ANSWER_FIXER_TEMPERATURE = 0.1  # Low temperature for strict, deterministic corrections

    # LLM Service Settings
    ENABLE_LLM_WARMUP = False  # Enable LLM service warmup on application startup

    # Reranker settings
    USE_RERANKER = False  # Set to False to disable reranking
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

config = Config()
