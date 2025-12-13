"""
Test script for Gemini preprocessing functionality.
Run this to verify your Gemini API setup is working correctly.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path='.env.local')

def test_gemini_api():
    """Test Gemini API connectivity."""
    print("=" * 60)
    print("Testing Gemini API Setup")
    print("=" * 60)
    
    # Check if API key is set
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env.local")
        print("\nPlease add your API key to .env.local:")
        print("GEMINI_API_KEY=AIzaSyDT7fQGhYT-yDbBxclE0j3OBkhsPnqKCrw")
        print("\nGet your key from: https://makersuite.google.com/app/apikey")
        return False
    
    print(f"✓ GEMINI_API_KEY found: {api_key[:10]}...")
    
    # Try importing google.generativeai
    try:
        import google.generativeai as genai
        print("✓ google-generativeai package installed")
    except ImportError:
        print("❌ google-generativeai package not installed")
        print("\nInstall it with:")
        print("pip install google-generativeai")
        return False
    
    # Try configuring and creating a model
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("✓ Gemini API configured successfully")
    except Exception as e:
        print(f"❌ Error configuring Gemini API: {e}")
        return False
    
    # Try a simple test generation
    try:
        print("\nTesting API call with simple prompt...")
        response = model.generate_content("Return only the word 'SUCCESS' in JSON format: {\"status\": \"SUCCESS\"}")
        print(f"✓ API call successful")
        print(f"Response: {response.text[:100]}")
    except Exception as e:
        print(f"❌ Error calling Gemini API: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Gemini API is ready to use.")
    print("=" * 60)
    return True


def test_document_processor():
    """Test document processor functionality."""
    print("\n" + "=" * 60)
    print("Testing Document Processor")
    print("=" * 60)
    
    try:
        from services.document_processor import document_processor
        print("✓ Document processor imported successfully")
    except Exception as e:
        print(f"❌ Error importing document processor: {e}")
        return False
    
    # Check if data folder exists
    data_folder = Path("data")
    if not data_folder.exists():
        print(f"⚠ Data folder not found: {data_folder}")
        print("Creating data folder...")
        data_folder.mkdir(exist_ok=True)
    else:
        print(f"✓ Data folder exists: {data_folder}")
    
    # Find test files
    pdf_files = list(data_folder.glob("**/*.pdf"))
    docx_files = list(data_folder.glob("**/*.docx"))
    
    print(f"Found {len(pdf_files)} PDF files")
    print(f"Found {len(docx_files)} DOCX files")
    
    if pdf_files:
        test_file = pdf_files[0]
        print(f"\nTesting extraction on: {test_file.name}")
        try:
            pages_data = document_processor.extract_text_with_pages(test_file)
            if pages_data:
                print(f"✓ Extracted {len(pages_data)} pages")
                print(f"First page preview: {pages_data[0].get('text', '')[:100]}...")
            else:
                print("⚠ No content extracted")
        except Exception as e:
            print(f"❌ Error extracting text: {e}")
            return False
    
    print("\n✅ Document processor is working")
    return True


def test_preprocessing_script():
    """Test the preprocessing script."""
    print("\n" + "=" * 60)
    print("Testing Preprocessing Script")
    print("=" * 60)
    
    try:
        from preprocess import DeepSeekChunker
        print("✓ Preprocessing script imported successfully")
    except Exception as e:
        print(f"❌ Error importing preprocess.py: {e}")
        return False
    
    try:
        chunker = DeepSeekChunker(model_name="deepseek-v3.1")
        print("✓ DeepSeekChunker initialized")
    except Exception as e:
        print(f"❌ Error initializing DeepSeekChunker: {e}")
        return False
    
    print("\n✅ Preprocessing script is ready")
    return True


def main():
    """Run all tests."""
    print("\n🚀 Gemini Preprocessing Test Suite\n")
    
    results = []
    
    # Test 1: Gemini API
    results.append(("Gemini API", test_gemini_api()))
    
    # Test 2: Document Processor
    results.append(("Document Processor", test_document_processor()))
    
    # Test 3: Preprocessing Script
    results.append(("Preprocessing Script", test_preprocessing_script()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to preprocess documents.")
        print("\nNext steps:")
        print("1. Add documents to the data/ folder")
        print("2. Run: python preprocess.py")
        print("3. Check output in data/chunks/")
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
