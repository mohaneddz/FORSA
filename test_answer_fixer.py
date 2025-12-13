"""
Test Cases for Strict RAG Answer Fixer

This file validates that the answer fixer correctly applies all strict RAG rules.
"""

import pytest
from utils.answer_fixer import fix_rag_answer, AnswerFixer


class TestRule1EmptySources:
    """Test Rule 1: Empty sources → unavailable message"""
    
    def test_empty_sources_list(self):
        result = fix_rag_answer(
            question_text="Quel est le prix?",
            retrieved_sources=[]
        )
        assert result == "Information non disponible dans les documents fournis."
    
    def test_none_sources(self):
        result = fix_rag_answer(
            question_text="Quel est le prix?",
            retrieved_sources=None
        )
        assert result == "Information non disponible dans les documents fournis."
    
    def test_sources_without_evidence(self):
        result = fix_rag_answer(
            question_text="Quel est le prix?",
            retrieved_sources=[
                {"file": "doc.pdf", "evidence": ""},
                {"file": "doc2.pdf", "evidence": "   "}
            ]
        )
        assert result == "Information non disponible dans les documents fournis."


class TestRule2OnlyUseEvidence:
    """Test Rule 2: Answer ONLY using words/phrases from evidence"""
    
    def test_remove_example_not_in_evidence(self):
        sources = [
            {"file": "doc.pdf", "evidence": "L'offre coûte 1990 DA par mois."}
        ]
        answer = "L'offre coûte 1990 DA par mois. Par exemple, vous pouvez payer mensuellement."
        
        result = fix_rag_answer("Quel est le prix?", sources, answer)
        
        # "Par exemple" should be removed since it's not in evidence
        assert "par exemple" not in result.lower()
        assert "L'offre coûte 1990 DA" in result
    
    def test_remove_arabic_examples(self):
        sources = [
            {"file": "doc.pdf", "evidence": "الرفع هو سرعة إرسال البيانات"}
        ]
        answer = "الرفع هو سرعة إرسال البيانات. مثل البريد الإلكتروني والصور."
        
        result = fix_rag_answer("ما معنى الرفع؟", sources, answer)
        
        # "مثل" (like/for example) should be removed
        assert "مثل" not in result
        assert "الرفع هو سرعة إرسال البيانات" in result
    
    def test_copy_paste_from_evidence(self):
        sources = [
            {"file": "doc.pdf", "evidence": "L'offre Idoom 4 Mega coûte 1990 DA par mois avec un engagement de 12 mois."}
        ]
        
        # No answer provided, should construct from evidence
        result = fix_rag_answer("Quel est le prix de l'offre Idoom 4 Mega?", sources)
        
        # Should be verbatim or very close to evidence
        assert "1990 DA" in result
        assert "12 mois" in result or "engagement" in result


class TestRule3AnswerOnlyWhatAsked:
    """Test Rule 3: Answer ONLY what question asks"""
    
    def test_dont_include_extra_modes(self):
        # Question asks for "modes" not sub-modes
        sources = [
            {"file": "doc.pdf", "evidence": "Les modes de paiement sont: Cash et TPE."},
            {"file": "doc.pdf", "evidence": "Le TPE accepte les cartes CIB et DAHABIA."}
        ]
        
        # If answer includes sub-modes when only modes were asked, should be filtered
        # This test documents expected behavior
        result = fix_rag_answer("Quels sont les modes de paiement?", sources)
        
        # Should mention main modes
        assert "Cash" in result or "TPE" in result
    
    def test_dont_merge_different_offers(self):
        # Question asks about ONE offer, don't merge multiple
        sources = [
            {"file": "Idoom.pdf", "evidence": "L'offre Idoom 4 Mega coûte 1990 DA."}
        ]
        
        result = fix_rag_answer("Quel est le prix de l'offre Idoom 4 Mega?", sources)
        
        # Should only mention Idoom 4 Mega
        assert "1990 DA" in result


class TestRule4RemoveIncompleteLines:
    """Test Rule 4: Remove incomplete lines"""
    
    def test_remove_ellipsis(self):
        sources = [
            {"file": "doc.pdf", "evidence": "L'offre coûte 1990 DA par mois."}
        ]
        answer = "L'offre coûte 1990 DA par mois. Ensuite..."
        
        result = fix_rag_answer("Quel est le prix?", sources, answer)
        
        # "Ensuite..." should be removed
        assert "ensuite" not in result.lower()
        assert "..." not in result
        assert "1990 DA" in result
    
    def test_remove_sentence_ending_with_ensuite(self):
        sources = [
            {"file": "doc.pdf", "evidence": "L'installation comprend 3 étapes."}
        ]
        answer = "L'installation comprend 3 étapes. La première est la vérification. Ensuite"
        
        result = fix_rag_answer("Quelles sont les étapes?", sources, answer)
        
        # Incomplete "Ensuite" should be removed
        assert not result.endswith("Ensuite")
        assert "3 étapes" in result
    
    def test_remove_arabic_incomplete_marker(self):
        sources = [
            {"file": "doc.pdf", "evidence": "التركيب يتضمن 3 خطوات"}
        ]
        answer = "التركيب يتضمن 3 خطوات. الخطوة الأولى هي التحقق. ثم"
        
        result = fix_rag_answer("ما هي الخطوات؟", sources, answer)
        
        # "ثم" (then) at the end should be removed
        assert not result.endswith("ثم")


class TestRule5ConsistencyCheck:
    """Test Rule 5: Consistency check against evidence"""
    
    def test_replace_unavailable_when_evidence_exists(self):
        sources = [
            {"file": "doc.pdf", "evidence": "L'offre coûte 1990 DA par mois."}
        ]
        answer = "Information non disponible dans les documents fournis."
        
        result = fix_rag_answer("Quel est le prix?", sources, answer)
        
        # Should NOT be unavailable since evidence exists
        assert "non disponible" not in result.lower()
        assert "1990 DA" in result
    
    def test_no_contradiction_with_evidence(self):
        sources = [
            {"file": "doc.pdf", "evidence": "L'offre coûte 1990 DA par mois."}
        ]
        answer = "L'offre coûte 2500 DA par mois."  # Wrong number!
        
        result = fix_rag_answer("Quel est le prix?", sources, answer)
        
        # Should be reconstructed from evidence, not contain wrong number
        assert "1990" in result
        assert "2500" not in result


class TestRule6Deduplicate:
    """Test Rule 6: Deduplicate repeated information"""
    
    def test_remove_duplicate_sentences(self):
        sources = [
            {"file": "doc.pdf", "evidence": "L'offre coûte 1990 DA par mois."},
            {"file": "doc2.pdf", "evidence": "L'offre coûte 1990 DA par mois."}
        ]
        
        result = fix_rag_answer("Quel est le prix?", sources)
        
        # Should only appear once
        count = result.count("1990 DA")
        assert count == 1, f"Expected '1990 DA' to appear once, but appeared {count} times"
    
    def test_keep_different_information(self):
        sources = [
            {"file": "doc.pdf", "evidence": "L'offre coûte 1990 DA par mois."},
            {"file": "doc.pdf", "evidence": "Le débit est de 4 Mbps."}
        ]
        
        result = fix_rag_answer("Donnez les détails de l'offre.", sources)
        
        # Both distinct facts should be present
        assert "1990 DA" in result
        assert "4 Mbps" in result or "débit" in result


class TestRule7LanguageConsistency:
    """Test Rule 7: Keep same language as evidence"""
    
    def test_arabic_evidence_arabic_answer(self):
        sources = [
            {"file": "doc.pdf", "evidence": "الرفع هو سرعة إرسال البيانات"}
        ]
        
        result = fix_rag_answer("ما معنى الرفع؟", sources)
        
        # Answer should be in Arabic
        assert "الرفع" in result
        assert "Upload" not in result or "Upload" in sources[0]["evidence"]  # Allow if in evidence
    
    def test_french_evidence_french_answer(self):
        sources = [
            {"file": "doc.pdf", "evidence": "L'upload est la vitesse d'envoi des données"}
        ]
        
        result = fix_rag_answer("Qu'est-ce que l'upload?", sources)
        
        # Answer should be in French
        assert "upload" in result.lower() or "vitesse" in result.lower()
    
    def test_mixed_evidence_prefers_matching_language(self):
        sources = [
            {"file": "doc_ar.pdf", "evidence": "الرفع هو سرعة إرسال البيانات"},
            {"file": "doc_fr.pdf", "evidence": "L'upload est la vitesse d'envoi"}
        ]
        
        # Arabic question should prefer Arabic evidence
        result = fix_rag_answer("ما معنى الرفع؟", sources)
        # This is aspirational - basic implementation may combine both
        # Full implementation would use language detection


class TestComplexScenarios:
    """Test complex real-world scenarios"""
    
    def test_procedure_all_steps_present(self):
        sources = [
            {"file": "guide.pdf", "evidence": "Étape 1: Vérifier l'éligibilité. Étape 2: Remplir le formulaire. Étape 3: Confirmer la demande."}
        ]
        
        result = fix_rag_answer("Quelles sont les étapes d'installation?", sources)
        
        # All steps should be present
        assert "Étape 1" in result or "Vérifier" in result
        assert "Étape 2" in result or "Remplir" in result
        assert "Étape 3" in result or "Confirmer" in result
    
    def test_list_all_items_preserved(self):
        sources = [
            {"file": "doc.pdf", "evidence": "Les offres disponibles sont: Idoom 4 Mega, Idoom 8 Mega, Idoom 20 Mega, Idoom 50 Mega."}
        ]
        
        result = fix_rag_answer("Quelles sont les offres disponibles?", sources)
        
        # All offers should be mentioned
        assert "4 Mega" in result
        assert "8 Mega" in result
        assert "20 Mega" in result
        assert "50 Mega" in result
    
    def test_numbers_exact_format(self):
        sources = [
            {"file": "doc.pdf", "evidence": "Le débit est ≥ 4 Mbps avec 50% de réduction."}
        ]
        
        result = fix_rag_answer("Quel est le débit?", sources)
        
        # Numbers and symbols should be exact
        assert "4" in result
        assert "Mbps" in result
        # "50%" might be included if it's relevant to the answer


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_very_long_evidence(self):
        long_text = "L'offre Idoom. " * 100
        sources = [
            {"file": "doc.pdf", "evidence": long_text}
        ]
        
        result = fix_rag_answer("Qu'est-ce que l'offre?", sources)
        
        # Should handle long text without error
        assert result is not None
        assert len(result) > 0
    
    def test_special_characters_preserved(self):
        sources = [
            {"file": "doc.pdf", "evidence": "Le prix est 1.990,00 DA/mois (TTC)."}
        ]
        
        result = fix_rag_answer("Quel est le prix?", sources)
        
        # Special characters should be preserved
        assert "1" in result and "990" in result
        assert "DA" in result
    
    def test_empty_question(self):
        sources = [
            {"file": "doc.pdf", "evidence": "Information test."}
        ]
        
        result = fix_rag_answer("", sources)
        
        # Should handle gracefully
        assert result is not None


class TestIntegrationWithCurrentSystem:
    """Test integration with existing RAG system structures"""
    
    def test_source_evidence_format(self):
        """Test with SourceEvidence-like structure"""
        sources = [
            {
                "file": "Offres_Idoom.pdf",
                "score": 0.85,
                "evidence": "L'offre Idoom 4 Mega coûte 1990 DA par mois."
            }
        ]
        
        result = fix_rag_answer("Quel est le prix?", sources)
        
        assert "1990 DA" in result
    
    def test_multiple_sources_same_info(self):
        """Test deduplication across multiple source files"""
        sources = [
            {"file": "doc1.pdf", "score": 0.9, "evidence": "L'offre coûte 1990 DA."},
            {"file": "doc2.pdf", "score": 0.85, "evidence": "L'offre coûte 1990 DA par mois."},
            {"file": "doc3.pdf", "score": 0.8, "evidence": "Le prix est de 1990 DA."}
        ]
        
        result = fix_rag_answer("Quel est le prix?", sources)
        
        # Should deduplicate
        count = result.count("1990")
        assert count <= 2  # Allow some variation but not all three


if __name__ == "__main__":
    # Run tests with pytest
    # pytest test_answer_fixer.py -v
    
    # Or run simple demonstration
    print("=" * 80)
    print("STRICT RAG ANSWER FIXER - DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Example 1: Remove incomplete sentences
    print("Example 1: Remove incomplete sentences")
    sources = [{"file": "doc.pdf", "evidence": "L'offre coûte 1990 DA par mois."}]
    answer = "L'offre coûte 1990 DA par mois. Ensuite..."
    fixed = fix_rag_answer("Quel est le prix?", sources, answer)
    print(f"Original: {answer}")
    print(f"Fixed: {fixed}")
    print()
    
    # Example 2: Remove forbidden additions
    print("Example 2: Remove forbidden additions")
    sources = [{"file": "doc.pdf", "evidence": "الرفع هو سرعة إرسال البيانات"}]
    answer = "الرفع هو سرعة إرسال البيانات. مثل البريد الإلكتروني."
    fixed = fix_rag_answer("ما معنى الرفع؟", sources, answer)
    print(f"Original: {answer}")
    print(f"Fixed: {fixed}")
    print()
    
    # Example 3: Fix unavailable when evidence exists
    print("Example 3: Fix unavailable when evidence exists")
    sources = [{"file": "doc.pdf", "evidence": "L'offre coûte 1990 DA par mois."}]
    answer = "Information non disponible dans les documents fournis."
    fixed = fix_rag_answer("Quel est le prix?", sources, answer)
    print(f"Original: {answer}")
    print(f"Fixed: {fixed}")
    print()
    
    # Example 4: Empty sources
    print("Example 4: Empty sources")
    sources = []
    fixed = fix_rag_answer("Quel est le prix?", sources)
    print(f"Fixed: {fixed}")
    print()
    
    print("=" * 80)
    print("Run 'pytest test_answer_fixer.py -v' for full test suite")
    print("=" * 80)
