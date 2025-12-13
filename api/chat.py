from fastapi import APIRouter, HTTPException
from typing import Dict, Tuple
from schemas import StructuredChatRequest, StructuredChatResponse
from services.rag_service import rag_service
from services.llm_service import llm_service
from utils.logging_utils import log_execution_time
from utils.answer_fixer import fix_rag_answer
import logging
import time
import json
import asyncio

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=StructuredChatResponse)
@log_execution_time
async def structured_chat_endpoint(payload: StructuredChatRequest):
    """
    GhostTruth structured RAG endpoint with JSON-enforced responses.
    Enforces strict grounding and bilingual support.
    Processes all questions in PARALLEL for maximum performance using pure async operations.
    """
    try:
        team_name = payload.equipe
        questions_map = payload.question
        
        responses_map: Dict[str, Dict[str, str]] = {}

        # Collect all questions to process
        all_tasks = []
        task_metadata = []  # Store (category_id, question_id) for each task
        
        for category_id, questions in questions_map.items():
            for question_id, question_text in questions.items():
                # Create async task for each question
                all_tasks.append(
                    process_single_question(question_text)
                )
                task_metadata.append((category_id, question_id))
                logger.debug(f"Queued Question - Category: {category_id}, ID: {question_id}")
        
        # Process ALL questions in parallel using asyncio.gather
        total_questions = len(all_tasks)
        logger.info(f"Processing {total_questions} questions in PARALLEL (pure async)...")
        start_time = time.time()
        
        # asyncio.gather runs all tasks concurrently
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / total_questions if total_questions > 0 else 0
        logger.info(f"✓ All {total_questions} questions processed in {elapsed:.2f}s (avg: {avg_time:.2f}s per question)")
        
        # Organize results into response structure
        for idx, ((category_id, question_id), result) in enumerate(zip(task_metadata, results)):
            # Handle exceptions from gather
            if isinstance(result, Exception):
                logger.error(f"Question {category_id}.{question_id} failed: {result}")
                answer = "Erreur: Le traitement de cette question a échoué."
            else:
                answer = result
            
            # Initialize category if needed
            if category_id not in responses_map:
                responses_map[category_id] = {}
            
            # Store results
            responses_map[category_id][question_id] = answer

        return StructuredChatResponse(
            equipe=team_name,
            reponses=responses_map
        )

    except Exception as e:
        logger.error(f"Error in structured chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def process_single_question(question_text: str) -> str:
    """
    Process a single question: retrieve context and generate answer.
    Uses pure async operations for maximum parallelism without thread pool overhead.
    Applies strict RAG validation rules with scope-based chunk filtering.
    Returns answer string.
    """
    from config import config
    from utils.validation_utils import (
        extract_question_scope,
        filter_chunks_by_scope,
        validate_answer_grounding
    )
    
    question_start_time = time.time()
    question_preview = question_text[:80] + "..." if len(question_text) > 80 else question_text
    
    logger.info(f"[START] Processing question: {question_preview}")
    
    try:
        # 0. Extract question scope for pre-filtering (CPU-bound, but fast)
        scope = extract_question_scope(question_text)
        logger.debug(f"Detected scope: {scope}")
        
        # 1. Retrieve relevant context using async retrieval
        start_time = time.time()
        relevant_chunks = await retrieve_chunks_async(question_text, config.TOP_K_SEMANTIC)
        retrieval_time = time.time() - start_time
        logger.debug(f"Retrieval: {retrieval_time:.2f}s, {len(relevant_chunks)} chunks")
        
        # 1.5. Apply scope-based filtering to remove irrelevant chunks (CPU-bound, but fast)
        filtered_chunks = filter_chunks_by_scope(relevant_chunks, scope)
        
        if len(filtered_chunks) == 0 and len(relevant_chunks) > 0:
            logger.warning(f"All {len(relevant_chunks)} chunks filtered out by scope - may indicate scope mismatch")
            # Use original chunks if filtering removed everything (prevents false negatives)
            filtered_chunks = relevant_chunks
        
        logger.info(f"Scope filtering: {len(relevant_chunks)} -> {len(filtered_chunks)} chunks")
        
        # 2. Format Context with file/page headers (CPU-bound, but fast)
        retrieved_chunks = rag_service.format_context_blocks(filtered_chunks)
        
        # 3. Construct Prompt
        from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
        
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=retrieved_chunks,
            question_text=question_text
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        # 4. Call LLM with async HTTP request
        start_llm = time.time()
        llm_response = await llm_service.chat_completion_async(messages, temperature=0.3, json_mode=True)
        llm_time = time.time() - start_llm
        logger.debug(f"LLM call: {llm_time:.2f}s")
        
        # 5. Parse JSON response
        try:
            response_data = json.loads(llm_response)
            answer = response_data.get("answer", "")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            answer = "Erreur: Réponse invalide du système."
        
        # 5.25. Apply answer fixer if enabled
        if config.ENABLE_ANSWER_FIXER:
            # Convert chunks to source format for fixer
            source_list = []
            for chunk in filtered_chunks:
                evidence_text = chunk.chunk.content  # Extract evidence from chunk
                source_list.append({
                    "file": chunk.chunk.metadata.get("file", "unknown"),
                    "score": chunk.score,
                    "evidence": evidence_text
                })
            
            original_answer = answer
            answer = await fix_rag_answer_async(question_text, source_list, answer)
            
            if answer != original_answer:
                logger.info(f"Answer fixer applied corrections")
                logger.debug(f"Original: {original_answer[:100]}...")
                logger.debug(f"Fixed: {answer[:100]}...")
        
        # 5.5. Validate answer grounding (CPU-bound, but fast)
        validation_result = validate_answer_grounding(answer, filtered_chunks)
        
        if not validation_result["is_valid"]:
            logger.warning(f"Answer validation issues: {validation_result['issues']}")
        
        if validation_result.get("warnings"):
            logger.info(f"Answer validation warnings: {validation_result['warnings']}")
        
        # Log validation metrics
        logger.debug(f"Validation: valid={validation_result['is_valid']}, "
                    f"unavailable={validation_result.get('is_unavailable', False)}, "
                    f"issues={len(validation_result.get('issues', []))}, "
                    f"warnings={len(validation_result.get('warnings', []))}")
        
        return answer
        
    except Exception as e:
        question_duration = time.time() - question_start_time
        logger.error(f"[FAILED] Question failed after {question_duration:.2f}s: {question_preview} - Error: {e}", exc_info=True)
        raise  # Re-raise to be caught by gather
    finally:
        question_duration = time.time() - question_start_time
        logger.info(f"[END] Question completed in {question_duration:.2f}s: {question_preview}")


async def retrieve_chunks_async(question_text: str, top_k: int):
    """
    Async wrapper for retrieval to avoid blocking the event loop.
    Uses asyncio.to_thread for CPU-bound operations.
    """
    return await asyncio.to_thread(
        rag_service.retrieve_and_rerank,
        question_text,
        top_k
    )


async def fix_rag_answer_async(question_text: str, source_list: list, answer: str) -> str:
    """
    Async wrapper for answer fixer to avoid blocking the event loop.
    """
    return await asyncio.to_thread(
        fix_rag_answer,
        question_text,
        source_list,
        answer
    )
