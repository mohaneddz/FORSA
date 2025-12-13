"""
Test the new preprocessing endpoint.
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_preprocess_endpoint():
    """Test the /api/embeddings/preprocess endpoint."""
    
    print("=" * 60)
    print("Testing Preprocessing Endpoint")
    print("=" * 60)
    
    # Endpoint URL
    url = f"{BASE_URL}/api/embeddings/preprocess"
    
    # Request payload
    payload = {
        "input_folder": "data",
        "output_folder": "gen",
        "model": "deepseek-v3.1",
        "extensions": [".pdf", ".docx"]
    }
    
    print(f"\nSending POST request to: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload)
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Success!")
            print(f"\nMessage: {result.get('message')}")
            print(f"Output Folder: {result.get('output_folder')}")
            
            stats = result.get('stats', {})
            print(f"\nStatistics:")
            print(f"  Total Files: {stats.get('total_files')}")
            print(f"  Successful: {stats.get('successful')}")
            print(f"  Failed: {stats.get('failed')}")
            
            files_processed = stats.get('files_processed', [])
            if files_processed:
                print(f"\nProcessed Files:")
                for file_info in files_processed[:5]:  # Show first 5
                    print(f"  - {file_info.get('file')}")
                    print(f"    Output: {file_info.get('output')}")
                    print(f"    Size: {file_info.get('size')} chars")
                
                if len(files_processed) > 5:
                    print(f"  ... and {len(files_processed) - 5} more")
            
            # Check if output folder exists
            output_folder = Path(result.get('output_folder'))
            if output_folder.exists():
                md_files = list(output_folder.glob("*.md"))
                print(f"\nGenerated Markdown Files: {len(md_files)}")
                if md_files:
                    print("Sample files:")
                    for md_file in md_files[:3]:
                        print(f"  - {md_file.name}")
                        
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error")
        print("Make sure the server is running:")
        print("  python main.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


def test_preprocess_custom():
    """Test with custom parameters."""
    
    print("\n" + "=" * 60)
    print("Testing with Custom Parameters")
    print("=" * 60)
    
    url = f"{BASE_URL}/api/embeddings/preprocess"
    
    payload = {
        "input_folder": "data/Convention",
        "output_folder": "gen/convention",
        "model": "deepseek-v3.1"
    }
    
    print(f"\nProcessing Convention folder...")
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            stats = result.get('stats', {})
            print(f"Processed: {stats.get('successful')} / {stats.get('total_files')} files")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("\n🚀 Gemini Preprocessing Endpoint Test\n")
    
    # Test basic endpoint
    test_preprocess_endpoint()
    
    # Test custom parameters
    # Uncomment if you have a Convention subfolder
    # test_preprocess_custom()
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check gen/ folder for Markdown files")
    print("2. Review processing_stats.json")
    print("3. Use the Markdown files for RAG")
