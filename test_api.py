import requests
import json
import time

BASE_URL = "http://localhost:3000/api"

def test_embeddings():
    print("Testing /api/embeddings/add...")
    payload = {
        "documents": [
            "This is a document about the GhostTruth team.",
            "The GhostTruth team specializes in AI and RAG systems.",
            "Offer A includes 50GB of data and unlimited calls.",
            "Offer B is a prepaid plan with 10GB data."
        ],
        "metadatas": [
            {"source": "doc1", "category": "team"},
            {"source": "doc1", "category": "team"},
            {"source": "offer_doc", "category": "offer"},
            {"source": "offer_doc", "category": "offer"}
        ]
    }
    try:
        response = requests.post(f"{BASE_URL}/embeddings/add", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

def test_chat():
    print("\nTesting /api/chat/ask...")
    payload = {
        "equipe": "GhostTruth",
        "question": {
            "Offer_A_Category": {
                "Q1": "What does Offer A include?"
            },
            "Team_Category": {
                "Q2": "What does the GhostTruth team do?"
            }
        }
    }
    try:
        response = requests.post(f"{BASE_URL}/chat/ask", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Wait for server to start
    time.sleep(2)
    test_embeddings()
    test_chat()
