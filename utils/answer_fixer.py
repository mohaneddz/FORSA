"""
STRICT RAG ANSWER FIXER (LLM-based)

This module implements an LLM-based post-processing fixer for RAG-generated answers
to ensure strict compliance with retrieval evidence using a specialized prompt.

The fixer uses a dedicated LLM call with strict instructions to:
- Remove unsupported information
- Enforce scope control
- Prevent chunk dumping
- Ensure proper formatting
- Handle conflicts in evidence

Usage:
    from utils.answer_fixer import fix_rag_answer
    
    corrected = fix_rag_answer(
        question_text="ما معنى الرفع Upload؟",
        retrieved_sources=[
            {"file": "doc.pdf", "score": 0.95, "evidence": "الرفع Upload هو سرعة إرسال البيانات"},
            {"file": "doc.pdf", "score": 0.92, "evidence": "الرفع يقاس بالميغابت في الثانية"}
        ],
        generated_answer="الرفع Upload هو سرعة إرسال البيانات. على سبيل المثال..."
    )
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def format_sources_for_prompt(sources: List[Dict[str, Any]]) -> str:
    """
    Format sources list into a readable format for the LLM prompt.
    
    Args:
        sources: List of dicts with 'file', 'score', and 'evidence' keys
        
    Returns:
        Formatted string representation of sources
    """
    if not sources:
        return "No sources provided."
    
    formatted_lines = []
    for idx, source in enumerate(sources, 1):
        file_name = source.get("file", "unknown")
        score = source.get("score", 0.0)
        evidence = source.get("evidence", "").strip()
        
        formatted_lines.append(f"[{idx}] file: {file_name}, score: {score:.4f}")
        formatted_lines.append(f"    evidence: {evidence}")
    
    return "\n".join(formatted_lines)


def fix_rag_answer(
    question_text: str,
    retrieved_sources: List[Dict[str, Any]],
    generated_answer: str = None
) -> str:
    """
    Fix a RAG answer using LLM-based validation to ensure strict compliance with evidence.
    
    This function sends the question, draft answer, and sources to an LLM with strict
    instructions to validate and correct the answer according to RAG rules.
    
    Args:
        question_text: The original question
        retrieved_sources: List of dicts with keys: 'file', 'score', 'evidence'
        generated_answer: The draft answer to fix. If None or empty, returns unavailable message.
        
    Returns:
        Corrected answer string
    """
    from services.llm_service import llm_service
    from prompts import ANSWER_FIXER_SYSTEM_PROMPT, ANSWER_FIXER_USER_TEMPLATE
    from config import config
    
    UNAVAILABLE_MESSAGE = "Information non disponible dans les documents fournis."
    
    # Rule 1: Empty sources → unavailable message
    if not retrieved_sources or len(retrieved_sources) == 0:
        logger.debug("No sources provided - returning unavailable message")
        return UNAVAILABLE_MESSAGE
    
    # Extract evidence texts to validate
    evidence_texts = [src.get("evidence", "") for src in retrieved_sources if src.get("evidence")]
    
    if not evidence_texts or all(not e.strip() for e in evidence_texts):
        logger.debug("No evidence text in sources - returning unavailable message")
        return UNAVAILABLE_MESSAGE
    
    # If no draft answer provided, return unavailable
    if not generated_answer or not generated_answer.strip():
        logger.debug("No draft answer provided - returning unavailable message")
        return UNAVAILABLE_MESSAGE
    
    # Format sources for the prompt
    formatted_sources = format_sources_for_prompt(retrieved_sources)
    
    # Construct the user prompt
    user_prompt = ANSWER_FIXER_USER_TEMPLATE.format(
        question=question_text,
        draft_answer=generated_answer,
        sources=formatted_sources
    )
    
    # Prepare messages for LLM
    messages = [
        {"role": "system", "content": ANSWER_FIXER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    # Get answer fixer temperature from config (default to 0.1 for strict adherence)
    temperature = getattr(config, 'ANSWER_FIXER_TEMPERATURE', 0.1)
    
    try:
        logger.info(f"Applying LLM-based answer fixer to question: {question_text[:100]}...")
        
        # Call LLM with low temperature for deterministic output
        # No JSON mode - we want plain text output
        fixed_answer = llm_service.chat_completion(
            messages=messages,
            temperature=temperature,
            json_mode=False
        )
        
        # Clean up the response
        fixed_answer = fixed_answer.strip()
        
        # Log if the answer was modified
        if fixed_answer != generated_answer:
            logger.info("Answer fixer made corrections")
            logger.debug(f"Original length: {len(generated_answer)}, Fixed length: {len(fixed_answer)}")
        else:
            logger.debug("Answer fixer: no changes needed")
        
        return fixed_answer
        
    except Exception as e:
        logger.error(f"Error in LLM-based answer fixer: {e}", exc_info=True)
        # Fallback: return original answer if fixer fails
        logger.warning("Falling back to original answer due to fixer error")
        return generated_answer


# Legacy class kept for backward compatibility (delegates to new LLM-based implementation)
class AnswerFixer:
    """
    Legacy wrapper class for backward compatibility.
    Delegates all operations to the new LLM-based fix_rag_answer function.
    """
    
    UNAVAILABLE_MESSAGE = "Information non disponible dans les documents fournis."
    
    @staticmethod
    def fix_rag_answer(
        question_text: str,
        retrieved_sources: List[Dict[str, Any]],
        generated_answer: str = None
    ) -> str:
        """Legacy method that delegates to the new LLM-based implementation."""
        return fix_rag_answer(question_text, retrieved_sources, generated_answer)
