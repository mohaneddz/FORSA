"""
Query processing and normalization service for improved retrieval.
Handles query expansion, language normalization, and synonym mapping.
"""
import re
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class QueryService:
    """Handles query preprocessing for better retrieval performance."""
    
    def __init__(self):
        # French-Arabic synonym mappings for common terms
        self.bilingual_synonyms = {
            # Payment terms
            "paiement": ["دفع", "paiement", "payment", "versement"],
            "arrangement": ["ترتيب", "arrangement", "modalité", "plan"],
            "acompte": ["دفعة مقدمة", "acompte", "avance", "down payment"],
            "mensualité": ["قسط شهري", "mensualité", "mensualite", "tranche"],
            
            # Offer terms
            "offre": ["عرض", "offre", "offer", "proposition"],
            "abonnement": ["اشتراك", "abonnement", "subscription"],
            
            # Technical terms
            "débit": ["سرعة", "débit", "debit", "vitesse", "speed"],
            "internet": ["انترنت", "internet", "connexion"],
            "ligne": ["خط", "ligne", "line"],
            
            # Conditions
            "conditions": ["شروط", "conditions", "critères", "criteres"],
            "éligibilité": ["الأهلية", "éligibilité", "eligibilite", "conditions"],
            "minimum": ["الحد الأدنى", "minimum", "min"],
            "maximum": ["الحد الأقصى", "maximum", "max"],
            
            # Modes
            "égal": ["متساوي", "égal", "egal", "equal", "équitable"],
            "non-égal": ["غير متساوي", "non-égal", "non-egal", "non equal", "inégal"],
        }
        
        # Document family patterns
        self.document_families = {
            "payment_arrangement": [
                "arrangement", "paiement", "modalité", "mensualité",
                "acompte", "tranche", "versement", "échéance", "echeance"
            ],
            "offers": [
                "offre", "idoom", "fawri", "boost", "weekend",
                "internet", "débit", "debit", "forfait"
            ],
            "ngbss": [
                "ngbss", "guide", "système", "systeme", "billing",
                "facturation", "technique"
            ],
            "convention": [
                "convention", "contrat", "accord", "engagement"
            ]
        }
        
        # Question type patterns
        self.question_patterns = {
            "list": [
                r"quels?\s+sont",
                r"liste[rz]?",
                r"énumérer",
                r"enumerer",
                r"all\s+\w+",
                r"tous\s+les",
                r"toutes\s+les",
                r"ما\s+هي",
            ],
            "definition": [
                r"c'est\s+quoi",
                r"qu'est-ce\s+que",
                r"définition",
                r"definition",
                r"signifie",
                r"veut\s+dire",
                r"ما\s+معنى",
                r"ما\s+تعريف",
            ],
            "procedure": [
                r"comment",
                r"étapes",
                r"etapes",
                r"procédure",
                r"procedure",
                r"process",
                r"كيف",
                r"خطوات",
            ],
            "constraint": [
                r"minimum",
                r"maximum",
                r"conditions?",
                r"critères?",
                r"criteres?",
                r"règles?",
                r"regles?",
                r"الحد",
                r"شروط",
            ]
        }
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize query text for better matching.
        
        Args:
            query: Raw user question
            
        Returns:
            Normalized query string
        """
        # Lowercase for French/English
        normalized = query.strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Normalize common variations
        normalized = normalized.replace("'", "'")
        normalized = normalized.replace("é", "e").replace("è", "e").replace("ê", "e")
        normalized = normalized.replace("à", "a").replace("â", "a")
        
        return normalized
    
    def expand_query(self, query: str) -> List[str]:
        """
        Generate query variations with synonyms for better recall.
        
        Args:
            query: Original query
            
        Returns:
            List of query variations including original and expanded versions
        """
        variations = [query]
        normalized = self.normalize_query(query)
        
        # Add normalized version if different
        if normalized.lower() != query.lower():
            variations.append(normalized)
        
        # Add versions with key synonym replacements
        for key, synonyms in self.bilingual_synonyms.items():
            if key in normalized.lower():
                # Try replacing with each synonym
                for syn in synonyms[:2]:  # Limit to top 2 synonyms
                    if syn != key:
                        expanded = normalized.replace(key, syn)
                        if expanded not in variations:
                            variations.append(expanded)
        
        logger.debug(f"Query expansion: {len(variations)} variations")
        return variations[:3]  # Limit to 3 variations max
    
    def detect_question_type(self, query: str) -> Optional[str]:
        """
        Detect the type of question being asked.
        
        Args:
            query: User question
            
        Returns:
            Question type: 'list', 'definition', 'procedure', 'constraint', or None
        """
        query_lower = query.lower()
        
        for qtype, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    logger.debug(f"Detected question type: {qtype}")
                    return qtype
        
        return None
    
    def detect_document_family(self, query: str) -> List[str]:
        """
        Detect which document families are relevant to the query.
        
        Args:
            query: User question
            
        Returns:
            List of relevant document family names
        """
        query_lower = self.normalize_query(query).lower()
        relevant_families = []
        
        for family, keywords in self.document_families.items():
            for keyword in keywords:
                if keyword in query_lower:
                    if family not in relevant_families:
                        relevant_families.append(family)
                    break
        
        if relevant_families:
            logger.info(f"Detected document families: {relevant_families}")
        else:
            logger.debug("No specific document family detected")
        
        return relevant_families
    
    def extract_key_terms(self, query: str) -> List[str]:
        """
        Extract important keywords from query for filtering.
        
        Args:
            query: User question
            
        Returns:
            List of key terms
        """
        # Remove common question words
        stopwords = {
            "quel", "quelle", "quels", "quelles", "est", "sont", "c'est",
            "qu'est-ce", "que", "qui", "comment", "pourquoi", "quand",
            "le", "la", "les", "un", "une", "des", "du", "de",
            "et", "ou", "dans", "pour", "avec", "sans",
            "ما", "هي", "هو", "في", "من", "إلى", "على"
        }
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [w for w in words if len(w) > 2 and w not in stopwords]
        
        # Keep only most important terms (nouns, specific terms)
        return key_terms[:5]  # Top 5 keywords max
    
    def enhance_query_for_retrieval(self, query: str) -> Dict[str, any]:
        """
        Comprehensive query analysis for improved retrieval.
        
        Args:
            query: Raw user question
            
        Returns:
            Dict containing:
            - normalized: Normalized query
            - variations: Query variations for multiple retrieval attempts
            - question_type: Detected question type
            - document_families: Relevant document families
            - key_terms: Important keywords
            - metadata_filters: Suggested metadata filters
        """
        normalized = self.normalize_query(query)
        variations = self.expand_query(query)
        question_type = self.detect_question_type(query)
        doc_families = self.detect_document_family(query)
        key_terms = self.extract_key_terms(query)
        
        # Build metadata filters based on detected families
        metadata_filters = {}
        if doc_families:
            metadata_filters["document_family"] = doc_families
        
        result = {
            "original": query,
            "normalized": normalized,
            "variations": variations,
            "question_type": question_type,
            "document_families": doc_families,
            "key_terms": key_terms,
            "metadata_filters": metadata_filters
        }
        
        logger.debug(f"Query enhancement complete: {result}")
        return result


# Global instance
query_service = QueryService()
