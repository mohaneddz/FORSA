"""
Test Cases for Strict RAG Rules Implementation

This file contains test scenarios to validate the strict RAG implementation.
Run these tests to ensure the system follows all 10 rules correctly.
"""

import json

# Test data representing different scenarios

# Test Case 1: Answer found in context (should return answer with sources)
test_case_1 = {
    "name": "Answer Found - Price Information",
    "context": [
        {
            "file": "Offres_Idoom.pdf",
            "page": 5,
            "content": "L'offre Idoom 4 Mega coûte 1990 DA par mois avec un engagement de 12 mois."
        }
    ],
    "question": "Quel est le prix de l'offre Idoom 4 Mega?",
    "expected_answer": "L'offre Idoom 4 Mega coûte 1990 DA par mois avec un engagement de 12 mois.",
    "expected_sources_count": ">0",
    "rule_tested": "Rule 2: When context contains the answer"
}

# Test Case 2: Answer NOT found in context (should return "Information non disponible" with empty sources)
test_case_2 = {
    "name": "Answer NOT Found - Missing Information",
    "context": [
        {
            "file": "Offres_Idoom.pdf",
            "page": 5,
            "content": "L'offre Idoom 4 Mega coûte 1990 DA par mois avec un engagement de 12 mois."
        }
    ],
    "question": "Quel est le prix de l'offre Idoom 100 Mega?",
    "expected_answer": "Information non disponible dans les documents fournis.",
    "expected_sources_count": "0",
    "rule_tested": "Rule 3: When to answer 'Information non disponible' + Rule 4: Empty sources"
}

# Test Case 3: Definition question (should return complete definition)
test_case_3 = {
    "name": "Definition - Complete and Unmodified",
    "context": [
        {
            "file": "Glossaire_Telecom.pdf",
            "page": 2,
            "content": "**Q: Qu'est-ce que l'ADSL?**\n\nA: L'ADSL (Asymmetric Digital Subscriber Line) est une technologie de transmission numérique qui permet de transmettre des données à haut débit sur une ligne téléphonique standard. Elle offre des débits asymétriques, c'est-à-dire plus rapides en téléchargement qu'en envoi."
        }
    ],
    "question": "C'est quoi l'ADSL?",
    "expected_answer": "L'ADSL (Asymmetric Digital Subscriber Line) est une technologie de transmission numérique qui permet de transmettre des données à haut débit sur une ligne téléphonique standard. Elle offre des débits asymétriques, c'est-à-dire plus rapides en téléchargement qu'en envoi.",
    "expected_sources_count": ">0",
    "rule_tested": "Rule 6: For definitions - copy directly, do not simplify"
}

# Test Case 4: List question (should return ALL items)
test_case_4 = {
    "name": "List - All Items Preserved",
    "context": [
        {
            "file": "Eligibilite.pdf",
            "page": 3,
            "content": "Les catégories éligibles à cette offre sont:\n- Entreprises\n- Professionnels\n- Particuliers\n- Administrations publiques\n- Associations"
        }
    ],
    "question": "Quelles sont les catégories éligibles?",
    "expected_answer": "Les catégories éligibles à cette offre sont: Entreprises, Professionnels, Particuliers, Administrations publiques, Associations.",
    "expected_sources_count": ">0",
    "rule_tested": "Rule 7: For list-based questions - return ALL items"
}

# Test Case 5: Arabic content (should answer in Arabic)
test_case_5 = {
    "name": "Language - Arabic Context = Arabic Answer",
    "context": [
        {
            "file": "عروض_ايدوم.pdf",
            "page": 2,
            "content": "عرض ايدوم 4 ميغا يكلف 1990 دينار جزائري شهريا مع التزام لمدة 12 شهرا."
        }
    ],
    "question": "ما هو سعر عرض ايدوم 4 ميغا؟",  # What is the price of Idoom 4 Mega?
    "expected_answer": "عرض ايدوم 4 ميغا يكلف 1990 دينار جزائري شهريا مع التزام لمدة 12 شهرا.",
    "expected_sources_count": ">0",
    "rule_tested": "Rule 8: Language rule - Arabic context → Arabic answer"
}

# Test Case 6: Numbers (should be exact, no rounding)
test_case_6 = {
    "name": "Numbers - Exact Values",
    "context": [
        {
            "file": "Tarifs_Detailles.pdf",
            "page": 7,
            "content": "Le débit de téléchargement est de 4.5 Mbps et le débit d'envoi est de 0.512 Mbps. Le prix mensuel est de 1990.50 DA."
        }
    ],
    "question": "Quels sont les débits de cette offre?",
    "expected_answer": "Le débit de téléchargement est de 4.5 Mbps et le débit d'envoi est de 0.512 Mbps.",
    "expected_sources_count": ">0",
    "rule_tested": "Rule 10 (Numbers): Extract exact numbers with units, never round"
}

# Test Case 7: Procedure (should include all steps)
test_case_7 = {
    "name": "Procedure - All Steps Included",
    "context": [
        {
            "file": "Guide_NGBSS.pdf",
            "page": 15,
            "content": "Procédure d'activation d'une ligne dans NGBSS:\n1. Accéder au menu 'Inventaire'\n2. Sélectionner 'Distribuer une ressource'\n3. Choisir la méthode: Rang / Liste / Fichier\n4. Saisir le numéro de ligne\n5. Cliquer sur 'Soumettre'\n6. Vérifier le statut dans 'Historique'"
        }
    ],
    "question": "Comment activer une ligne dans NGBSS?",
    "expected_answer": "Procédure d'activation d'une ligne dans NGBSS:\n1. Accéder au menu 'Inventaire'\n2. Sélectionner 'Distribuer une ressource'\n3. Choisir la méthode: Rang / Liste / Fichier\n4. Saisir le numéro de ligne\n5. Cliquer sur 'Soumettre'\n6. Vérifier le statut dans 'Historique'",
    "expected_sources_count": ">0",
    "rule_tested": "Rule 10 (Procedures): Provide clear numbered steps, keep all together"
}

# Test Case 8: Hallucination prevention (should NOT fabricate)
test_case_8 = {
    "name": "No Hallucination - Partial Information",
    "context": [
        {
            "file": "Offres_Idoom.pdf",
            "page": 5,
            "content": "L'offre Idoom 4 Mega offre un débit de 4 Mbps en téléchargement."
        }
    ],
    "question": "Quel est le prix et le débit de l'offre Idoom 4 Mega?",
    "expected_answer": "L'offre Idoom 4 Mega offre un débit de 4 Mbps en téléchargement. (Note: Price information is not available in context, so should say 'Information non disponible' OR only mention what's available)",
    "expected_sources_count": "depends",
    "rule_tested": "Rule 5: Never hallucinate - only state what's in context"
}

# Test Case 9: Scope rules - General vs Specific
test_case_9 = {
    "name": "Scope - Use General Unless Specified",
    "context": [
        {
            "file": "Offre_General.pdf",
            "page": 1,
            "content": "**Q: J'ai un problème, qui dois-je contacter ?**\n\nA: Pour toute information et/ou réclamation, appelez le 12."
        },
        {
            "file": "Offre_Locataire.pdf",
            "page": 1,
            "content": "**Q: J'ai un problème, qui dois-je contacter ?** (LOCATAIRE)\n\nA: Pour les locataires, contactez d'abord votre propriétaire."
        }
    ],
    "question": "J'ai un problème, qui dois-je contacter ?",
    "expected_answer": "Pour toute information et/ou réclamation, appelez le 12.",
    "expected_sources_count": ">0",
    "rule_tested": "Scope rules: Use general scope unless specifically asked"
}

# Test Case 10: List completeness
test_case_10 = {
    "name": "List - Return ALL Items",
    "context": [
        {
            "file": "Offres.pdf",
            "page": 2,
            "content": "Les débits disponibles sont: 4 Mbps, 8 Mbps, 10 Mbps, 20 Mbps, 50 Mbps, 100 Mbps."
        }
    ],
    "question": "Quels sont les débits disponibles?",
    "expected_answer": "Les débits disponibles sont: 4 Mbps, 8 Mbps, 10 Mbps, 20 Mbps, 50 Mbps, 100 Mbps.",
    "expected_sources_count": ">0",
    "rule_tested": "List rules: Return ALL items, do not omit or merge"
}

# Test Case 11: Definition - Do not expand
test_case_11 = {
    "name": "Definition - Copy As-Is, No Expansion",
    "context": [
        {
            "file": "Glossaire.pdf",
            "page": 1,
            "content": "**Q: C'est quoi la fibre optique?**\n\nA: La fibre optique est une technologie qui utilise des câbles de verre pour transmettre des données à très haute vitesse."
        }
    ],
    "question": "C'est quoi la fibre optique?",
    "expected_answer": "La fibre optique est une technologie qui utilise des câbles de verre pour transmettre des données à très haute vitesse.",
    "expected_sources_count": ">0",
    "rule_tested": "Definition rules: Copy as-is, no expansion or examples"
}

# Test Case 12: Source alignment
test_case_12 = {
    "name": "Source Alignment - Every Fact Supported",
    "context": [
        {
            "file": "Tarifs.pdf",
            "page": 3,
            "content": "L'offre 4 Mega coûte 1990 DA par mois."
        },
        {
            "file": "Debits.pdf",
            "page": 5,
            "content": "L'offre 4 Mega offre un débit de 4 Mbps en téléchargement."
        }
    ],
    "question": "Donnez-moi les détails de l'offre 4 Mega.",
    "expected_answer": "L'offre 4 Mega coûte 1990 DA par mois et offre un débit de 4 Mbps en téléchargement.",
    "expected_sources_count": ">0",
    "rule_tested": "Source alignment: Every fact must be supported by sources"
}

# Compile all test cases
all_test_cases = [
    test_case_1,
    test_case_2,
    test_case_3,
    test_case_4,
    test_case_5,
    test_case_6,
    test_case_7,
    test_case_8,
    test_case_9,
    test_case_10,
    test_case_11,
    test_case_12
]

def print_test_cases():
    """Print all test cases in a readable format"""
    print("=" * 80)
    print("STRICT RAG RULES - TEST CASES")
    print("=" * 80)
    print()
    
    for i, test in enumerate(all_test_cases, 1):
        print(f"Test Case {i}: {test['name']}")
        print(f"Rule Tested: {test['rule_tested']}")
        print(f"Question: {test['question']}")
        print(f"Expected Answer: {test['expected_answer']}")
        print(f"Expected Sources: {test['expected_sources_count']}")
        print()
        print("Context:")
        for ctx in test['context']:
            print(f"  File: {ctx['file']} (Page {ctx['page']})")
            print(f"  Content: {ctx['content'][:100]}...")
        print("-" * 80)
        print()

def validate_response(response_json: str, test_case: dict) -> dict:
    """
    Validate a response against a test case
    
    Args:
        response_json: JSON response from the system
        test_case: Test case dictionary
    
    Returns:
        Dictionary with validation results
    """
    try:
        response = json.loads(response_json)
        answer = response.get("answer", "")
        
        # Check if answer matches expected (or contains key elements)
        answer_valid = test_case["expected_answer"].lower() in answer.lower() or answer.lower() in test_case["expected_answer"].lower()
        
        # Check JSON format
        json_valid = isinstance(response, dict) and "answer" in response
        
        # Check for markdown violations
        no_markdown = not ("```" in response_json or "```json" in response_json.lower())
        
        return {
            "test_name": test_case["name"],
            "json_valid": json_valid,
            "answer_valid": answer_valid,
            "no_markdown": no_markdown,
            "answer": answer,
            "passed": json_valid and answer_valid and no_markdown
        }
        
    except json.JSONDecodeError:
        return {
            "test_name": test_case["name"],
            "json_valid": False,
            "answer_valid": False,
            "no_markdown": False,
            "answer": None,
            "passed": False,
            "error": "Invalid JSON"
        }

if __name__ == "__main__":
    print_test_cases()
    print("\nTo run actual tests, integrate with your API endpoint.")
    print("Example:")
    print("""
    import requests
    
    for test in all_test_cases:
        # Format request
        payload = {
            "equipe": "GhostTruth",
            "question": {"CAT001": {"Q001": test["question"]}},
            "include_sources": True
        }
        
        # Send request
        response = requests.post("http://localhost:8000/api/chat/", json=payload)
        result = response.json()
        
        # Validate
        validation = validate_response(
            json.dumps(result["reponses"]["CAT001"]["Q001"]), 
            test
        )
        print(f"Test: {validation['test_name']} - {'PASSED' if validation['passed'] else 'FAILED'}")
    """)
