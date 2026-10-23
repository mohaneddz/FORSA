"""
Validation utilities for strict RAG answer quality control.
"""
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def validate_answer_grounding(answer: str, chunks: List[Any]) -> Dict[str, Any]:
    """
    Validate that the answer is properly grounded in the retrieved chunks.
    
    Args:
        answer: The generated answer text
        chunks: List of SearchResult or dict objects with chunk content
        
    Returns:
        Dict with validation results:
        - is_valid: bool
        - issues: List[str] of detected issues
        - warnings: List[str] of potential concerns
    """
    issues = []
    warnings = []
    
    # Check if answer is the unavailability message
    unavailable_patterns = [
        "information non disponible dans les documents fournis",
        "information non disponible",
        "non disponible dans les documents"
    ]
    
    is_unavailable = any(pattern in answer.lower() for pattern in unavailable_patterns)
    
    if is_unavailable:
        # Valid response when no information available
        return {
            "is_valid": True,
            "is_unavailable": True,
            "issues": [],
            "warnings": []
        }
    
    # Extract chunk content
    chunk_texts = []
    for chunk in chunks:
        if hasattr(chunk, 'chunk'):
            chunk_texts.append(chunk.chunk.content.lower())
        elif isinstance(chunk, dict):
            chunk_texts.append(chunk.get('text', '').lower())
    
    combined_chunks = ' '.join(chunk_texts)
    
    # Check for common assumption patterns (indicators of hallucination)
    assumption_patterns = [
        r'\bpeut-être\b',
        r'\bprobablement\b',
        r'\bgénéralement\b',
        r'\ben général\b',
        r'\bnormalement\b',
        r'\bselon les pratiques\b',
        r'\bil est possible\b',
        r'\bje pense\b',
        r'\bje suppose\b',
        r'\bon peut supposer\b',
    ]
    
    for pattern in assumption_patterns:
        if re.search(pattern, answer.lower()):
            warnings.append(f"Assumption language detected: {pattern}")
    
    # Check for mixing indicators (multiple offers/modes mentioned)
    mixing_patterns = [
        r'\bet aussi\b.*\boffre\b',
        r'\bou bien\b.*\bmode\b',
        r'\bainsi que\b.*\bvariant',
        r'\bégalement\b.*\boffre\b',
    ]
    
    for pattern in mixing_patterns:
        if re.search(pattern, answer.lower()):
            warnings.append(f"Potential context mixing: {pattern}")
    
    # Check for numbers/values in answer
    # Verify they appear in chunks (exact match)
    number_pattern = r'\b\d+(?:[.,]\d+)?\s*(?:DA|Mbps|Gbps|GB|Mo|%|mois|jours?)\b'
    answer_numbers = re.findall(number_pattern, answer)
    
    for num in answer_numbers:
        if num.lower() not in combined_chunks:
            issues.append(f"Number/value not found in chunks: {num}")
    
    # Basic validation
    is_valid = len(issues) == 0
    
    return {
        "is_valid": is_valid,
        "is_unavailable": False,
        "issues": issues,
        "warnings": warnings
    }


def extract_question_scope(question: str) -> Dict[str, Optional[str]]:
    """
    Extract scope indicators from the question to help with chunk filtering.
    Now with enhanced detection using query service.
    
    Args:
        question: User's question text
        
    Returns:
        Dict with detected scope elements:
        - offer_name: Detected offer name
        - mode: Detected mode (Equal, Non-Equal, etc.)
        - language: Detected language (ar/fr)
        - document_families: Relevant document categories
        - key_terms: Important keywords for filtering
    """
    from services.query_service import query_service
    
    scope = {
        "offer_name": None,
        "mode": None,
        "language": None,
        "document_families": [],
        "key_terms": []
    }
    
    question_lower = question.lower()
    
    # Detect specific offers (expanded list)
    offer_keywords = {
        "idoom": "Idoom",
        "fawri": "Fawri",
        "weekend boost": "Weekend Boost",
        "boost": "Boost",
        "4g lte": "4G LTE",
        "adsl": "ADSL",
        "fibre": "Fibre",
    }
    
    for keyword, offer in offer_keywords.items():
        if keyword in question_lower:
            scope["offer_name"] = offer
            break
    
    # Detect modes with stricter patterns
    mode_keywords = {
        # Non-Equal patterns (more specific first)
        "non-equal arrangement": "Non-Equal",
        "non equal arrangement": "Non-Equal",
        "non-égal": "Non-Equal",
        "non-egal": "Non-Equal",
        "non equal": "Non-Equal",
        "inégal": "Non-Equal",
        "inegal": "Non-Equal",
        "غير متساوي": "Non-Equal",
        
        # Equal patterns (checked after Non-Equal to avoid substring match)
        "equal arrangement": "Equal",
        "égal arrangement": "Equal",
        "egal arrangement": "Equal",
        "arrangement égal": "Equal",
        "arrangement egal": "Equal",
        "equal": "Equal",
        "égal": "Equal",
        "egal": "Equal",
        "équitable": "Equal",
        "متساوي": "Equal",
    }
    
    for keyword, mode in mode_keywords.items():
        if keyword in question_lower:
            scope["mode"] = mode
            logger.debug(f"Mode detected: {mode} from keyword: {keyword}")
            break
    
    # Detect language
    arabic_pattern = r'[\u0600-\u06FF]'
    if re.search(arabic_pattern, question):
        scope["language"] = "ar"
    else:
        scope["language"] = "fr"
    
    # Use query service for document family and key term detection
    try:
        query_analysis = query_service.enhance_query_for_retrieval(question)
        scope["document_families"] = query_analysis.get("document_families", [])
        scope["key_terms"] = query_analysis.get("key_terms", [])
    except Exception as e:
        logger.warning(f"Query service analysis failed: {e}")
    
    return scope


def filter_chunks_by_scope(chunks: List[Any], scope: Dict[str, Optional[str]]) -> List[Any]:
    """
    Filter chunks based on detected question scope.
    Removes chunks from irrelevant offers/modes/documents.
    Now includes document family filtering to reduce retrieval noise.
    
    Args:
        chunks: List of SearchResult or dict objects
        scope: Dict from extract_question_scope
        
    Returns:
        Filtered list of chunks matching the scope
    """
    if not any(scope.values()):
        # No specific scope detected, return all chunks
        return chunks
    
    filtered = []
    
    for chunk in chunks:
        # Extract metadata
        if hasattr(chunk, 'chunk'):
            metadata = chunk.chunk.metadata or {}
            content = chunk.chunk.content.lower()
        elif isinstance(chunk, dict):
            metadata = chunk.get('metadata', {})
            content = chunk.get('text', '').lower()
        else:
            continue
        
        source_file = metadata.get('file', '').lower()
        source_path = metadata.get('source', source_file).lower()
        chunk_language = metadata.get('language', '').lower()
        
        # Apply filters
        keep = True
        filter_reason = None
        
        # 1. Filter by document family (reject NGBSS guide if asking about offers)
        if scope.get('document_families'):
            doc_families = scope['document_families']
            
            # Check if chunk matches any of the target families
            is_relevant_family = False
            for family in doc_families:
                if family == "payment_arrangement":
                    if any(kw in source_path for kw in ["arrangement", "paiement", "modalite"]):
                        is_relevant_family = True
                        break
                elif family == "offers":
                    if any(kw in source_path for kw in ["offre", "idoom", "fawri", "boost"]):
                        is_relevant_family = True
                        break
                elif family == "ngbss":
                    if "ngbss" in source_path or "guide" in source_path:
                        is_relevant_family = True
                        break
            
            # Filter out if from wrong family (e.g., NGBSS guide when asking about offers)
            if not is_relevant_family:
                # Check if it's a noise document
                if "ngbss" in source_path or "guide" in source_path:
                    keep = False
                    filter_reason = "document_family_mismatch(NGBSS_noise)"
        
        # 2. Filter by offer name
        if scope.get('offer_name') and keep:
            offer = scope['offer_name'].lower()
            # Chunk text counts too: "Argumentaire Offre Gamers V5 2025 FR.docx" is an Idoom Fibre offer
            if offer not in source_file and offer not in source_path and offer not in content:
                keep = False
                filter_reason = f"offer_mismatch({scope['offer_name']})"
        
        # 3. Filter by mode (STRICT matching to prevent Equal/Non-Equal confusion)
        if scope.get('mode') and keep:
            mode = scope['mode']
            
            if mode == "Non-Equal":
                # Must contain non-equal indicators
                non_equal_patterns = ["non-equal", "non equal", "non-egal", "non-égal", "inegal", "inégal"]
                has_non_equal = any(pattern in source_path for pattern in non_equal_patterns)
                
                # Must NOT contain only equal indicators
                equal_only_patterns = ["equal arrangement", "egal arrangement", "égal arrangement"]
                has_equal_only = any(pattern in source_path for pattern in equal_only_patterns)
                
                if not has_non_equal or has_equal_only:
                    # Also check content for mode markers
                    if "non-equal" not in content and "non equal" not in content:
                        keep = False
                        filter_reason = "mode_mismatch(expected_Non-Equal)"
            
            elif mode == "Equal":
                # Must contain equal indicators but NOT non-equal
                equal_patterns = ["equal", "egal", "égal", "équitable"]
                non_equal_patterns = ["non-equal", "non equal", "non-egal", "non-égal"]
                
                has_equal = any(pattern in source_path for pattern in equal_patterns)
                has_non_equal = any(pattern in source_path for pattern in non_equal_patterns)
                
                if has_non_equal or not has_equal:
                    # Check content too
                    if "non-equal" in content or "non equal" in content:
                        keep = False
                        filter_reason = "mode_mismatch(expected_Equal)"
        
        # 4. Filter by language (soft filter - prefer but don't exclude)
        if scope.get('language') and keep and chunk_language:
            if scope['language'] != chunk_language:
                logger.debug(f"Language mismatch (soft): {chunk_language} vs {scope['language']}")
                # Don't exclude, just log
        
        # 5. Filter by key terms (soft filter - boost relevance)
        if scope.get('key_terms') and keep:
            key_terms = scope['key_terms']
            content_for_match = source_path + " " + content
            
            # Count how many key terms match
            matches = sum(1 for term in key_terms if term in content_for_match)
            
            if matches == 0 and len(key_terms) >= 3:
                # No key term matches and we have strong signal
                logger.debug(f"Low key term match: 0/{len(key_terms)} for chunk")
                # Don't exclude yet, but flag for lower priority
        
        if keep:
            filtered.append(chunk)
        else:
            logger.debug(f"Filtered chunk ({filter_reason}): {source_file}")
    
    logger.info(f"Scope filtering: {len(chunks)} -> {len(filtered)} chunks")
    return filtered
