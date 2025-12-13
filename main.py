from fastapi import FastAPI
from api import chat, embeddings, cache
import logging
from logger_config import setup_logging

# Setup logging
setup_logging()

app = FastAPI(title="Heavy RAG LOCAL System")

# Include Routers
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(embeddings.router, prefix="/api/embeddings", tags=["Embeddings"])
app.include_router(cache.router, prefix="/api/cache", tags=["Cache"])

@app.get("/")
async def root():
    return {"message": "Heavy RAG LOCAL System is running"}

@app.on_event("startup")
async def startup_event():
    """
    Startup event handler - runs when the application starts.
    Warms up the DeepSeek service to reduce first-request latency (if enabled).
    """
    from config import config
    logger = logging.getLogger(__name__)
    logger.info("🚀 Application startup - initializing services...")
    
    # Warmup DeepSeek LLM service (if enabled)
    if config.ENABLE_LLM_WARMUP:
        try:
            from services.llm_service import llm_service
            llm_service.warmup()
        except Exception as e:
            logger.warning(f"Failed to warmup LLM service: {e}")
    else:
        logger.info("LLM warmup disabled in config")
    
    logger.info("✓ Application startup complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app")