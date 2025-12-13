"""
Test script for the new LLM-based Answer Fixer

This script demonstrates the new answer fixer with a simple example.
"""

from utils.answer_fixer import fix_rag_answer

def test_llm_answer_fixer():
    """Test the LLM-based answer fixer with a simple example."""
    
    print("=" * 80)
    print("Testing LLM-based Answer Fixer")
    print("=" * 80)
    
    # Example question
    question = "Quel est le prix de l'offre IDOOM ADSL 8 Mega?"
    
    # Example sources (retrieved chunks)
    sources = [
        {
            "file": "Offre_IDOOM_ADSL.pdf",
            "score": 0.95,
            "evidence": "L'offre IDOOM ADSL 8 Mega coûte 1990 DA par mois."
        },
        {
            "file": "Offre_IDOOM_ADSL.pdf",
            "score": 0.87,
            "evidence": "Le prix mensuel est de 1990 DA incluant la TVA."
        }
    ]
    
    # Example draft answer (with extra information not in sources)
    draft_answer = """
    L'offre IDOOM ADSL 8 Mega coûte 1990 DA par mois. 
    Par exemple, vous pouvez l'activer en 24 heures et elle offre une vitesse stable.
    Elle est disponible dans toutes les wilayas.
    """
    
    print(f"\nQuestion: {question}")
    print(f"\nDraft Answer:\n{draft_answer}")
    print(f"\nSources:")
    for i, src in enumerate(sources, 1):
        print(f"  [{i}] {src['file']} (score: {src['score']:.2f})")
        print(f"      {src['evidence']}")
    
    print("\n" + "-" * 80)
    print("Calling Answer Fixer...")
    print("-" * 80 + "\n")
    
    try:
        fixed_answer = fix_rag_answer(
            question_text=question,
            retrieved_sources=sources,
            generated_answer=draft_answer
        )
        
        print(f"Fixed Answer:\n{fixed_answer}")
        print("\n" + "=" * 80)
        print("✓ Test completed successfully!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_llm_answer_fixer()
