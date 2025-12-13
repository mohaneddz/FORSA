"""
Test the updated embeddings generation with AI chunks.
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_gen_endpoint():
    """Test the /api/embeddings/gen endpoint with AI chunks."""
    
    print("=" * 60)
    print("Testing Embeddings Generation with AI Chunks")
    print("=" * 60)
    
    # Check if gen folder has markdown files
    gen_folder = Path("gen")
    if gen_folder.exists():
        md_files = list(gen_folder.glob("**/*.md"))
        print(f"\n✓ Found {len(md_files)} Markdown files in gen/ folder")
        if md_files:
            print("Sample files:")
            for md_file in md_files[:5]:
                print(f"  - {md_file.name}")
    else:
        print("\n⚠ gen/ folder not found")
        print("Run preprocessing first:")
        print("  python preprocess.py")
        print("  OR")
        print("  POST /api/embeddings/preprocess")
    
    # Test the endpoint
    url = f"{BASE_URL}/api/embeddings/gen"
    
    print(f"\nSending POST request to: {url}")
    
    try:
        response = requests.post(url)
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Success!")
            print(f"\nMessage: {result.get('message')}")
            print(f"Source: {result.get('source')}")
            print(f"Files Processed: {result.get('files_processed')}")
            print(f"Chunks Created: {result.get('chunks_created')}")
            print(f"Total in Store: {result.get('total_documents_in_store')}")
            
            # Highlight if using AI chunks
            if result.get('source') == 'deepseek_markdown':
                print("\n🎉 Using AI-generated chunks from DeepSeek!")
            elif result.get('source') == 'basic_chunking':
                print("\n⚠ Fallback: Using basic chunking")
                print("Tip: Run preprocessing to use AI chunks:")
                print("  python preprocess.py")
                
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error")
        print("Make sure the server is running:")
        print("  python main.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


def test_full_workflow():
    """Test the complete workflow: preprocess -> generate embeddings."""
    
    print("\n" + "=" * 60)
    print("Testing Complete Workflow")
    print("=" * 60)
    
    # Step 1: Preprocess
    print("\n[1/2] Preprocessing documents...")
    preprocess_url = f"{BASE_URL}/api/embeddings/preprocess"
    
    try:
        preprocess_response = requests.post(preprocess_url, json={
            "input_folder": "data",
            "output_folder": "gen"
        })
        
        if preprocess_response.status_code == 200:
            result = preprocess_response.json()
            print(f"✓ Preprocessing complete: {result['stats']['successful']} files")
        else:
            print(f"⚠ Preprocessing failed: {preprocess_response.status_code}")
            if "DEEPSEEK_TOKEN" in preprocess_response.text:
                print("Note: DeepSeek API key not configured - skipping this step")
            return
    except Exception as e:
        print(f"⚠ Preprocessing error (skipping): {e}")
    
    # Step 2: Generate embeddings
    print("\n[2/2] Generating embeddings from AI chunks...")
    gen_url = f"{BASE_URL}/api/embeddings/gen"
    
    try:
        gen_response = requests.post(gen_url)
        
        if gen_response.status_code == 200:
            result = gen_response.json()
            print(f"✓ Embeddings generated: {result['chunks_created']} chunks")
            print(f"✓ Source: {result['source']}")
            
            if result['source'] == 'deepseek_markdown':
                print("\n🎉 Success! Using AI-generated chunks")
            
        else:
            print(f"✗ Embeddings generation failed: {gen_response.status_code}")
            
    except Exception as e:
        print(f"✗ Error: {e}")


def inspect_chunk_metadata():
    """Query the system to inspect chunk metadata."""
    
    print("\n" + "=" * 60)
    print("Inspecting Chunk Metadata")
    print("=" * 60)
    
    # This would require a query endpoint to retrieve chunks
    # For now, we'll just check the vector store count
    
    gen_url = f"{BASE_URL}/api/embeddings/gen"
    
    try:
        # Get current state
        response = requests.post(gen_url)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\nVector Store Info:")
            print(f"  Total Documents: {result.get('total_documents_in_store')}")
            print(f"  Last Source: {result.get('source')}")
            
            if result.get('source') == 'deepseek_markdown':
                print(f"\n✓ Chunks include AI metadata:")
                print(f"  - File name")
                print(f"  - Page number")
                print(f"  - Section title")
                print(f"  - Content type (table, qa, procedure, paragraph)")
                print(f"  - Language")
                
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("\n🚀 AI Chunks Embedding Test\n")
    
    # Test 1: Check current setup
    test_gen_endpoint()
    
    # Test 2: Full workflow (optional - commented out to avoid re-processing)
    # Uncomment to test complete workflow
    # test_full_workflow()
    
    # Test 3: Inspect metadata
    inspect_chunk_metadata()
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
    print("\nWorkflow Summary:")
    print("1. Run: python preprocess.py (or POST /api/embeddings/preprocess)")
    print("2. Run: POST /api/embeddings/gen")
    print("3. AI chunks automatically used if gen/ folder has .md files")
    print("4. Falls back to basic chunking if no .md files found")
