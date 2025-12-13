"""
Answer Fixer CLI Tool

Quick command-line interface for testing the strict RAG answer fixer.

Usage:
    python answer_fixer_cli.py --interactive
    python answer_fixer_cli.py --test-examples
    python answer_fixer_cli.py --fix "question" "answer" --evidence "ev1" "ev2"
"""

import argparse
import sys
from typing import List
from utils.answer_fixer import fix_rag_answer


def interactive_mode():
    """Interactive mode for testing the fixer."""
    print("=" * 80)
    print("STRICT RAG ANSWER FIXER - Interactive Mode")
    print("=" * 80)
    print()
    print("Enter question, answer, and evidence to see the fixer in action.")
    print("Type 'quit' or 'exit' to quit.")
    print()
    
    while True:
        print("-" * 80)
        
        # Get question
        question = input("Question: ").strip()
        if question.lower() in ['quit', 'exit']:
            break
        
        if not question:
            print("Question cannot be empty. Try again.")
            continue
        
        # Get answer
        answer = input("Generated Answer: ").strip()
        if answer.lower() in ['quit', 'exit']:
            break
        
        # Get evidence (multiple lines)
        print("Evidence (enter each evidence line, press Enter twice when done):")
        evidence_lines = []
        while True:
            line = input("  > ").strip()
            if line.lower() in ['quit', 'exit']:
                sys.exit(0)
            if not line:
                break
            evidence_lines.append(line)
        
        if not evidence_lines:
            print("At least one evidence line is required. Try again.")
            continue
        
        # Build sources
        sources = [
            {"file": f"source_{i+1}.pdf", "evidence": ev}
            for i, ev in enumerate(evidence_lines)
        ]
        
        # Apply fixer
        print()
        print("Applying fixer...")
        fixed = fix_rag_answer(question, sources, answer if answer else None)
        
        # Display results
        print()
        print("RESULTS:")
        print("-" * 80)
        print(f"Original Answer: {answer if answer else '[Generated from evidence]'}")
        print(f"Fixed Answer:    {fixed}")
        print()
        
        if answer and answer != fixed:
            print("✓ Corrections applied")
        else:
            print("✓ No corrections needed")
        print()


def test_examples():
    """Run predefined test examples."""
    print("=" * 80)
    print("STRICT RAG ANSWER FIXER - Test Examples")
    print("=" * 80)
    print()
    
    examples = [
        {
            "name": "Remove Incomplete Sentence",
            "question": "Quel est le prix?",
            "sources": [{"file": "doc.pdf", "evidence": "L'offre coûte 1990 DA par mois."}],
            "answer": "L'offre coûte 1990 DA par mois. Ensuite..."
        },
        {
            "name": "Remove Forbidden Addition (French)",
            "question": "Qu'est-ce que l'upload?",
            "sources": [{"file": "doc.pdf", "evidence": "L'upload est la vitesse d'envoi des données."}],
            "answer": "L'upload est la vitesse d'envoi des données. Par exemple, pour les emails."
        },
        {
            "name": "Remove Forbidden Addition (Arabic)",
            "question": "ما معنى الرفع؟",
            "sources": [{"file": "doc.pdf", "evidence": "الرفع هو سرعة إرسال البيانات"}],
            "answer": "الرفع هو سرعة إرسال البيانات. مثل البريد الإلكتروني."
        },
        {
            "name": "Fix False Unavailable",
            "question": "Quel est le prix?",
            "sources": [{"file": "doc.pdf", "evidence": "L'offre coûte 1990 DA par mois."}],
            "answer": "Information non disponible dans les documents fournis."
        },
        {
            "name": "Deduplicate Repeated Info",
            "question": "Quel est le prix?",
            "sources": [
                {"file": "doc1.pdf", "evidence": "L'offre coûte 1990 DA."},
                {"file": "doc2.pdf", "evidence": "L'offre coûte 1990 DA par mois."}
            ],
            "answer": None  # Will generate from sources
        },
        {
            "name": "Empty Sources",
            "question": "Quel est le prix?",
            "sources": [],
            "answer": "Le prix est 1990 DA."
        },
        {
            "name": "Fix Wrong Numbers",
            "question": "Quel est le prix?",
            "sources": [{"file": "doc.pdf", "evidence": "L'offre coûte 1990 DA."}],
            "answer": "L'offre coûte 2500 DA."
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"Example {i}: {example['name']}")
        print("-" * 80)
        print(f"Question: {example['question']}")
        print(f"Sources: {len(example['sources'])} sources")
        for j, src in enumerate(example['sources'], 1):
            print(f"  [{j}] {src.get('evidence', '')}")
        print(f"Original: {example['answer'] if example['answer'] else '[Generate from sources]'}")
        
        # Apply fixer
        fixed = fix_rag_answer(
            example['question'],
            example['sources'],
            example['answer']
        )
        
        print(f"Fixed:    {fixed}")
        
        # Show if changes made
        if example['answer'] and example['answer'] != fixed:
            print("✓ Corrections applied")
        elif not example['answer']:
            print("✓ Generated from sources")
        else:
            print("✓ No corrections needed")
        
        print()
    
    print("=" * 80)
    print(f"Completed {len(examples)} examples")
    print("=" * 80)


def fix_command(question: str, answer: str, evidence_list: List[str]):
    """Fix a single answer via command line."""
    sources = [
        {"file": f"source_{i+1}.pdf", "evidence": ev}
        for i, ev in enumerate(evidence_list)
    ]
    
    fixed = fix_rag_answer(question, sources, answer if answer else None)
    
    print("FIXED ANSWER:")
    print(fixed)


def main():
    parser = argparse.ArgumentParser(
        description="Strict RAG Answer Fixer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python answer_fixer_cli.py --interactive
  
  # Test predefined examples
  python answer_fixer_cli.py --test-examples
  
  # Fix a specific answer
  python answer_fixer_cli.py --fix "Quel est le prix?" "Le prix est 1990 DA. Ensuite..." \\
      --evidence "L'offre coûte 1990 DA par mois."
        """
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--test-examples', '-t',
        action='store_true',
        help='Run predefined test examples'
    )
    
    parser.add_argument(
        '--fix', '-f',
        nargs=2,
        metavar=('QUESTION', 'ANSWER'),
        help='Fix a specific question and answer'
    )
    
    parser.add_argument(
        '--evidence', '-e',
        nargs='+',
        metavar='EVIDENCE',
        help='Evidence texts (required with --fix)'
    )
    
    args = parser.parse_args()
    
    # Determine which mode to run
    if args.interactive:
        interactive_mode()
    elif args.test_examples:
        test_examples()
    elif args.fix:
        if not args.evidence:
            print("Error: --evidence is required when using --fix")
            sys.exit(1)
        
        question, answer = args.fix
        fix_command(question, answer, args.evidence)
    else:
        # Default: show help
        parser.print_help()
        print()
        print("Tip: Try 'python answer_fixer_cli.py --test-examples' to see examples")


if __name__ == "__main__":
    main()
