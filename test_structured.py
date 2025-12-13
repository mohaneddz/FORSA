"""
Test script for GhostTruth structured Q&A endpoint.
Tests the /api/chat/ endpoint with JSON validation.
"""

import requests
import json
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:3000"
ENDPOINT = f"{BASE_URL}/api/chat/"

def load_test_data(file_path: str):
    """Load test questions from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def test_endpoint():
    """Test the structured endpoint"""
    print("=" * 80)
    print("TESTING STRUCTURED ENDPOINT: /api/chat/")
    print("=" * 80)
    
    # Load test data
    test_file = Path("test/algerie_telecom_questions.json")
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    payload = load_test_data(test_file)
    
    print(f"\n📤 Sending request with {len(payload['question'])} categories...")
    print(f"Request payload preview:")
    print(json.dumps(payload, indent=2, ensure_ascii=False)[:500])
    print("...\n")
    
    try:
        response = requests.post(ENDPOINT, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        print("✅ Response received successfully!")
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        
        # Display results
        for category_id, questions in result.get("reponses", {}).items():
            print(f"\n📁 Category: {category_id}")
            print("-" * 80)
            
            for question_id, answer in questions.items():
                print(f"\n  🔹 {question_id}:")
                print(f"     Question: {payload['question'][category_id][question_id]}")
                print(f"     Answer: {answer[:200]}..." if len(answer) > 200 else f"     Answer: {answer}")
        
        # Save full response
        output_file = Path("test/response.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n\n💾 Full response saved to: {output_file}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return None

if __name__ == "__main__":
    print("🚀 GhostTruth Structured Q&A System Test")
    print(f"Ensure the server is running at {BASE_URL}\n")
    
    test_endpoint()
    
    print("\n✨ Test completed!")
