from fastapi import APIRouter, HTTPException
from services.vector_store import vector_store
from utils.logging_utils import log_execution_time

router = APIRouter()

@router.post("/clear")
@log_execution_time
async def clear_cache():
    """
    Clear the FAISS index and metadata.
    """
    try:
        vector_store.clear_index()
        return {"status": "success", "message": "Index cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reload")
@log_execution_time
async def reload_cache():
    """
    Reload the FAISS index from disk.
    """
    try:
        vector_store.load_index()
        return {"status": "success", "message": "Index reloaded.", "total_documents": vector_store.index.ntotal if vector_store.index else 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
