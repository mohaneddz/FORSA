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
    MODELS_FP16 = True  # run embedding + reranker in fp16 when CUDA is available
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

    # LLM Service Settings
    ENABLE_LLM_WARMUP = False  # Enable LLM service warmup on application startup
    LLM_MAX_CONNECTIONS = 20  # pooled HTTP connections to the LLM API

    # Reranker settings
    USE_RERANKER = True  # Set to False to disable reranking
    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
    RERANK_CANDIDATES = 10  # hybrid candidates scored per question
    # Sigmoid reranker score a chunk needs to reach the LLM. On eval/questions.json
    # off-topic questions top out below 0.01 and every in-scope question keeps
    # its source at any value from 0.01 to 0.3 (see eval/threshold_sweep.py).
    RERANK_MIN_SCORE = 0.05
    RERANK_BATCH_SIZE = 32
    RERANK_MAX_LENGTH = 512
    SAME_LANGUAGE_BONUS = 0.05  # ranking bonus for chunks written in the question's language

config = Config()
