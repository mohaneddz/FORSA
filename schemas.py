from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

# --- Input Schemas ---

class StructuredChatRequest(BaseModel):
    """Structured request format for GhostTruth"""
    equipe: str = Field(..., description="Team name, e.g., GhostTruth")
    question: Dict[str, Dict[str, str]] = Field(..., description="Nested structure: {Category_ID: {Question_ID: Question_Text}}")

# --- Internal/Other Schemas ---

class DocumentChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class SearchResult(BaseModel):
    chunk: DocumentChunk
    score: float

class StructuredChatResponse(BaseModel):
    """Structured response format for GhostTruth"""
    equipe: str = Field(..., description="Team name")
    reponses: Dict[str, Dict[str, str]] = Field(..., description="Nested structure: {Category_ID: {Question_ID: Answer}}")
