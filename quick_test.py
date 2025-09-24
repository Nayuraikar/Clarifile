#!/usr/bin/env python3
import requests
import json

print("🔍 Quick Connectivity Test")
print("=" * 30)

# Test gateway
try:
    response = requests.get("http://127.0.0.1:4000/drive/health", timeout=5)
    print(f"✅ Gateway: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"❌ Gateway Error: {e}")

# Test parser service
try:
    response = requests.get("http://127.0.0.1:8000/", timeout=5)
    print(f"✅ Parser: {response.status_code}")
except Exception as e:
    print(f"❌ Parser Error: {e}")

# Test analyze endpoint
try:
    test_data = {
        "file": {
            "id": "test_file_id",
            "name": "test_document.txt", 
            "mimeType": "text/plain",
            "parents": []
        },
        "q": "What is this document about?"
    }
    response = requests.post("http://127.0.0.1:4000/drive/analyze", json=test_data, timeout=10)
    print(f"✅ Analyze Endpoint: {response.status_code}")
    if response.status_code != 200:
        print(f"   Error: {response.text}")
    else:
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"❌ Analyze Error: {e}")

print("\n🔍 If you see errors above, the services are not running!")
print("🚀 Start services with:")
print("   cd gateway && node index.js")
print("   cd services/parser && python app.py")
print("   cd services/embed && python app.py")
print("   cd services/indexer && python app.py") 
print("   cd services/dedup && python app.py")
