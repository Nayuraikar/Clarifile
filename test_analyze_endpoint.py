#!/usr/bin/env python3
"""
Test script to debug the analyze endpoint issue
"""
import requests
import json
import time

BASE_URL = "http://localhost:3000"

def test_endpoint(endpoint, method="GET", data=None):
    """Test an endpoint and return the response"""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n=== Testing {method} {url} ===")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error Response: {response.text}")
            
        return response
        
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR: Server is not running")
        return None
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT: Server took too long to respond")
        return None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

def main():
    print("🔍 Testing Clarifile Backend Services...")
    print("=" * 50)
    
    # Test basic health endpoints
    test_endpoint("/drive/health", "GET")
    test_endpoint("/drive/proposals", "GET")
    
    # Test the analyze endpoint with a sample request
    print("\n" + "=" * 50)
    print("🧪 Testing Analyze Endpoint...")
    
    analyze_data = {
        "file": {
            "id": "test_file_id",
            "name": "test_document.txt",
            "mimeType": "text/plain",
            "parents": []
        },
        "q": "What is this document about?"
    }
    
    test_endpoint("/drive/analyze", "POST", analyze_data)
    
    # Test with no question
    print("\n" + "=" * 50)
    print("🧪 Testing Analyze Endpoint (no question)...")
    
    analyze_data_no_q = {
        "file": {
            "id": "test_file_id",
            "name": "test_document.txt",
            "mimeType": "text/plain",
            "parents": []
        }
    }
    
    test_endpoint("/drive/analyze", "POST", analyze_data_no_q)
    
    print("\n" + "=" * 50)
    print("✅ Testing Complete!")

if __name__ == "__main__":
    main()
