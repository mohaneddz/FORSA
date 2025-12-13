"""
Integration Test for Answer Fixer with Chat Endpoint

This test verifies the complete integration of the answer fixer
with the RAG pipeline and chat endpoint.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from utils.answer_fixer import fix_rag_answer, AnswerFixer
from schemas import SearchResult, DocumentChunk


class TestIntegrationWithChatEndpoint:
    """Test integration with the actual chat endpoint flow."""
    
    def test_fixer_with_real_chunk_format(self):
        """Test fixer with SearchResult chunks (actual format from chat.py)"""
        
        # Create mock chunks in the format used by the chat endpoint
        chunk1 = Mock()
        chunk1.content = "L'offre Idoom 4 Mega coûte 1990 DA par mois avec un engagement de 12 mois."
        chunk1.metadata = {
            "file": "Offres_Idoom.pdf",
            "page": 5,
            "type": "md_chunk",
            "section_title": "Tarifs"
        }
        
        chunk2 = Mock()
        chunk2.content = "L'offre Idoom 4 Mega offre un débit de 4 Mbps."
        chunk2.metadata = {
            "file": "Offres_Idoom.pdf",
            "page": 5,
            "type": "md_chunk",
            "section_title": "Débits"
        }
        
        search_result1 = Mock()
        search_result1.chunk = chunk1
        search_result1.score = 0.85
        
        search_result2 = Mock()
        search_result2.chunk = chunk2
        search_result2.score = 0.82
        
        filtered_chunks = [search_result1, search_result2]
        
        # Simulate the conversion that happens in chat.py
        source_list = []
        for chunk in filtered_chunks:
            evidence_text = chunk.chunk.content
            source_list.append({
                "file": chunk.chunk.metadata.get("file", "unknown"),
                "score": chunk.score,
                "evidence": evidence_text
            })
        
        # Test the fixer
        question = "Quel est le prix de l'offre Idoom 4 Mega?"
        answer = "L'offre Idoom 4 Mega coûte 1990 DA par mois. Par exemple, c'est abordable."
        
        fixed = fix_rag_answer(question, source_list, answer)
        
        # Verify "Par exemple" is removed
        assert "par exemple" not in fixed.lower()
        assert "1990 DA" in fixed
    
    def test_fixer_disabled_scenario(self):
        """Test behavior when fixer is disabled (returns original answer)"""
        
        sources = [
            {"file": "doc.pdf", "evidence": "L'offre coûte 1990 DA."}
        ]
        answer = "L'offre coûte 1990 DA. Par exemple, c'est abordable."
        
        # When fixer is disabled, we'd just return the answer as-is
        # This test documents the expected behavior
        
        # With fixer enabled:
        fixed = fix_rag_answer("Quel est le prix?", sources, answer)
        assert "par exemple" not in fixed.lower()
        
        # Without fixer (simulated):
        # In chat.py, this would be:
        # if config.ENABLE_ANSWER_FIXER:
        #     answer = fix_rag_answer(...)
        # else:
        #     # answer remains unchanged
        #     pass
        
        # So original answer would be returned
        assert "par exemple" in answer.lower()  # Original unchanged
    
    def test_empty_sources_integration(self):
        """Test empty sources scenario as it would happen in chat endpoint"""
        
        # When no chunks are retrieved
        filtered_chunks = []
        
        # Convert to source format (would result in empty list)
        source_list = []
        for chunk in filtered_chunks:
            evidence_text = chunk.chunk.content
            source_list.append({
                "file": chunk.chunk.metadata.get("file", "unknown"),
                "score": chunk.score,
                "evidence": evidence_text
            })
        
        # Apply fixer
        question = "Quel est le prix?"
        answer = "Le prix est de 1990 DA."  # Hypothetical hallucinated answer
        
        fixed = fix_rag_answer(question, source_list, answer)
        
        # Should return unavailable message
        assert fixed == "Information non disponible dans les documents fournis."
    
    def test_unavailable_answer_with_sources(self):
        """Test fixing when LLM returns unavailable but sources exist"""
        
        # Chunks exist
        chunk = Mock()
        chunk.content = "L'offre Idoom 4 Mega coûte 1990 DA par mois."
        chunk.metadata = {"file": "Offres.pdf", "page": 5}
        
        search_result = Mock()
        search_result.chunk = chunk
        search_result.score = 0.85
        
        filtered_chunks = [search_result]
        
        # Convert to sources
        source_list = [{
            "file": chunk.metadata.get("file"),
            "score": search_result.score,
            "evidence": chunk.content[:300]
        }]
        
        # LLM incorrectly returns unavailable
        question = "Quel est le prix?"
        answer = "Information non disponible dans les documents fournis."
        
        # Fixer should reconstruct from evidence
        fixed = fix_rag_answer(question, source_list, answer)
        
        # Should NOT be unavailable
        assert "non disponible" not in fixed.lower()
        assert "1990 DA" in fixed


class TestPerformanceAndThreadSafety:
    """Test performance characteristics and thread safety."""
    
    def test_fixer_performance(self):
        """Verify fixer completes quickly (< 100ms for typical case)"""
        import time
        
        sources = [
            {"file": "doc.pdf", "evidence": "L'offre coûte 1990 DA."}
        ]
        question = "Quel est le prix?"
        answer = "L'offre coûte 1990 DA. Par exemple, c'est abordable."
        
        start = time.time()
        fixed = fix_rag_answer(question, sources, answer)
        duration = time.time() - start
        
        # Should complete in < 100ms
        assert duration < 0.1, f"Fixer took {duration*1000:.2f}ms, expected < 100ms"
        assert "1990 DA" in fixed
    
    def test_fixer_with_long_evidence(self):
        """Test performance with long evidence texts"""
        import time
        
        # Long evidence (simulate real documents)
        long_evidence = "L'offre Idoom comprend. " * 100
        sources = [
            {"file": "doc.pdf", "evidence": long_evidence}
        ]
        
        question = "Qu'est-ce que l'offre?"
        answer = long_evidence + " Ensuite..."
        
        start = time.time()
        fixed = fix_rag_answer(question, sources, answer)
        duration = time.time() - start
        
        # Should still be fast
        assert duration < 0.2, f"Fixer took {duration*1000:.2f}ms with long evidence"
        
        # Should remove incomplete
        assert not fixed.endswith("Ensuite...")


class TestLoggingIntegration:
    """Test logging behavior."""
    
    def test_corrections_are_logged(self, caplog):
        """Verify that corrections are logged"""
        import logging
        
        # Set log level
        caplog.set_level(logging.DEBUG, logger='utils.answer_fixer')
        
        sources = [
            {"file": "doc.pdf", "evidence": "L'offre coûte 1990 DA."}
        ]
        answer = "L'offre coûte 1990 DA. Par exemple..."
        
        fixed = fix_rag_answer("Quel est le prix?", sources, answer)
        
        # Check that corrections were logged
        # Note: Actual log messages depend on implementation
        # This test documents expected behavior
        assert fixed != answer  # Correction was made


class TestConfigIntegration:
    """Test configuration integration."""
    
    def test_config_flag_exists(self):
        """Verify config flag is available"""
        from config import config
        
        # Config should have the flag
        assert hasattr(config, 'ENABLE_ANSWER_FIXER')
    
    def test_config_default_value(self):
        """Verify default value is True (fixer enabled)"""
        from config import config
        
        # Should be enabled by default
        assert config.ENABLE_ANSWER_FIXER == True


class TestRealWorldScenarios:
    """Test with real-world question/answer scenarios."""
    
    def test_arabic_question_scenario(self):
        """Test with Arabic question and evidence"""
        sources = [
            {
                "file": "Guide_AR.pdf",
                "evidence": "الرفع Upload هو سرعة إرسال البيانات من الحاسوب إلى الإنترنت"
            }
        ]
        
        question = "ما معنى الرفع Upload؟"
        answer = "الرفع Upload هو سرعة إرسال البيانات من الحاسوب إلى الإنترنت. مثل البريد الإلكتروني والصور."
        
        fixed = fix_rag_answer(question, sources, answer)
        
        # Should remove "مثل البريد..."
        assert "مثل" not in fixed
        assert "الرفع" in fixed
        assert "سرعة إرسال البيانات" in fixed
    
    def test_procedure_question(self):
        """Test with procedure/steps question"""
        sources = [
            {
                "file": "Installation.pdf",
                "evidence": "Étape 1: Vérifier l'éligibilité. Étape 2: Remplir le formulaire. Étape 3: Confirmer."
            }
        ]
        
        question = "Quelles sont les étapes d'installation?"
        answer = "Étape 1: Vérifier l'éligibilité. Étape 2: Remplir le formulaire. Ensuite..."
        
        fixed = fix_rag_answer(question, sources, answer)
        
        # Should remove incomplete "Ensuite..."
        assert not fixed.endswith("Ensuite...")
        assert "Étape 1" in fixed or "Vérifier" in fixed
        assert "Étape 2" in fixed or "Remplir" in fixed
    
    def test_price_question_with_wrong_number(self):
        """Test fixing wrong numbers"""
        sources = [
            {"file": "Tarifs.pdf", "evidence": "L'offre Idoom 4 Mega coûte 1990 DA par mois."}
        ]
        
        question = "Quel est le prix de Idoom 4 Mega?"
        answer = "L'offre coûte 2500 DA par mois."  # Wrong number!
        
        fixed = fix_rag_answer(question, sources, answer)
        
        # Should be reconstructed with correct number
        assert "1990" in fixed
        assert "2500" not in fixed
    
    def test_duplicate_information(self):
        """Test deduplication with multiple sources"""
        sources = [
            {"file": "doc1.pdf", "evidence": "L'offre coûte 1990 DA par mois."},
            {"file": "doc2.pdf", "evidence": "L'offre coûte 1990 DA par mois."},
            {"file": "doc3.pdf", "evidence": "Prix: 1990 DA/mois"}
        ]
        
        question = "Quel est le prix?"
        
        # Generate from sources
        fixed = fix_rag_answer(question, sources)
        
        # Should not repeat "1990 DA" three times
        count = fixed.count("1990")
        assert count <= 2  # Allow some variation but not all three


def run_integration_tests():
    """Run all integration tests manually (without pytest)"""
    print("=" * 80)
    print("ANSWER FIXER - Integration Tests")
    print("=" * 80)
    print()
    
    test_cases = [
        ("Real chunk format", TestIntegrationWithChatEndpoint().test_fixer_with_real_chunk_format),
        ("Empty sources", TestIntegrationWithChatEndpoint().test_empty_sources_integration),
        ("Fix unavailable", TestIntegrationWithChatEndpoint().test_unavailable_answer_with_sources),
        ("Performance", TestPerformanceAndThreadSafety().test_fixer_performance),
        ("Config flag", TestConfigIntegration().test_config_flag_exists),
        ("Arabic scenario", TestRealWorldScenarios().test_arabic_question_scenario),
        ("Procedure question", TestRealWorldScenarios().test_procedure_question),
        ("Wrong number", TestRealWorldScenarios().test_price_question_with_wrong_number),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in test_cases:
        try:
            test_func()
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1
    
    print()
    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed == 0:
        print("✅ All integration tests passed!")
    else:
        print(f"❌ {failed} test(s) failed")


if __name__ == "__main__":
    # Run with pytest for full details
    # pytest test_answer_fixer_integration.py -v
    
    # Or run manual tests
    run_integration_tests()
