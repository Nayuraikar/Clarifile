#!/usr/bin/env python3
"""
test_api_keys.py
Test script to verify Gemini API keys are loaded and working properly.
"""

import os
import sys
import requests
import time

# Add parser service to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'parser'))

def test_api_key_loading():
    """Test if API keys are loaded correctly."""
    print("🔍 Testing API key loading...")
    
    # Test direct file reading
    keys_file = os.path.join(os.path.dirname(__file__), 'services', 'parser', 'gemini_keys.txt')
    print(f"📁 Keys file path: {keys_file}")
    print(f"📁 File exists: {os.path.exists(keys_file)}")
    
    if os.path.exists(keys_file):
        with open(keys_file, 'r') as f:
            keys = [line.strip() for line in f.readlines() if line.strip()]
        print(f"📊 Found {len(keys)} keys in file")
        print(f"📝 First key preview: {keys[0][:20]}..." if keys else "No keys found")
    
    # Test module loading
    try:
        import nlp
        print(f"📊 NLP module loaded API keys: {len(nlp.GEMINI_API_KEYS)}")
        if nlp.GEMINI_API_KEYS:
            print(f"📝 First loaded key preview: {nlp.GEMINI_API_KEYS[0][:20]}...")
        else:
            print("❌ No API keys loaded by nlp module")
    except Exception as e:
        print(f"❌ Error importing nlp module: {e}")

def test_single_api_key():
    """Test a single API key with a simple request."""
    print("\n🧪 Testing single API key...")
    
    keys_file = os.path.join(os.path.dirname(__file__), 'services', 'parser', 'gemini_keys.txt')
    if not os.path.exists(keys_file):
        print("❌ Keys file not found")
        return
    
    with open(keys_file, 'r') as f:
        keys = [line.strip() for line in f.readlines() if line.strip()]
    
    if not keys:
        print("❌ No keys found in file")
        return
    
    # Test first key
    test_key = keys[0]
    print(f"🔑 Testing key: {test_key[:20]}...")
    
    # Simple Gemini API test
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": test_key
    }
    
    payload = {
        "contents": [
            {"parts": [{"text": "Say 'Hello, API test successful!' in exactly those words."}]}
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 50
        }
    }
    
    try:
        print("🔄 Making API request...")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            try:
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                print(f"✅ API Response: {text}")
                return True
            except Exception as e:
                print(f"❌ Error parsing response: {e}")
                print(f"📄 Raw response: {result}")
        elif response.status_code == 401:
            print("❌ 401 Unauthorized - API key is invalid")
        elif response.status_code == 403:
            print("❌ 403 Forbidden - API key lacks permissions")
        elif response.status_code == 429:
            print("❌ 429 Rate Limited - Too many requests")
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    return False

def test_multiple_keys():
    """Test multiple API keys to see which ones work."""
    print("\n🧪 Testing multiple API keys...")
    
    keys_file = os.path.join(os.path.dirname(__file__), 'services', 'parser', 'gemini_keys.txt')
    if not os.path.exists(keys_file):
        print("❌ Keys file not found")
        return
    
    with open(keys_file, 'r') as f:
        keys = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"🔑 Testing {min(5, len(keys))} keys...")
    
    working_keys = []
    
    for i, key in enumerate(keys[:5]):  # Test first 5 keys
        print(f"\n🔑 Testing key {i+1}: {key[:20]}...")
        
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": key
        }
        
        payload = {
            "contents": [
                {"parts": [{"text": "Respond with just the number 42."}]}
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 10
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            
            if response.status_code == 200:
                print(f"   ✅ Key {i+1} works!")
                working_keys.append(key)
            elif response.status_code == 401:
                print(f"   ❌ Key {i+1} - 401 Unauthorized")
            elif response.status_code == 403:
                print(f"   ❌ Key {i+1} - 403 Forbidden")
            elif response.status_code == 429:
                print(f"   ⚠️  Key {i+1} - 429 Rate Limited")
            else:
                print(f"   ❌ Key {i+1} - HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Key {i+1} - Exception: {e}")
        
        # Small delay between requests
        time.sleep(1)
    
    print(f"\n📊 Summary: {len(working_keys)}/{min(5, len(keys))} keys are working")
    return working_keys

def test_nlp_module_directly():
    """Test the nlp module's summarize function directly."""
    print("\n🧪 Testing nlp module summarize function...")
    
    try:
        import nlp
        
        test_text = "This is a test invoice for software development services. The total amount due is $1,250.00 with payment terms of Net 30 days."
        
        print(f"📝 Test text: {test_text}")
        print("🔄 Calling nlp.summarize_with_gemini...")
        
        result = nlp.summarize_with_gemini(test_text, max_tokens=100)
        
        if result:
            print(f"✅ Summarization successful: {result}")
            return True
        else:
            print("❌ Summarization returned empty result")
            return False
            
    except Exception as e:
        print(f"❌ Error testing nlp module: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all API key tests."""
    print("🚀 GEMINI API KEY TESTING")
    print("="*50)
    
    # Test 1: API key loading
    test_api_key_loading()
    
    # Test 2: Single key test
    single_key_works = test_single_api_key()
    
    # Test 3: Multiple keys test
    working_keys = test_multiple_keys()
    
    # Test 4: NLP module test
    nlp_works = test_nlp_module_directly()
    
    print("\n" + "="*50)
    print("🎯 TEST SUMMARY")
    print("="*50)
    
    if single_key_works:
        print("✅ At least one API key is working")
    else:
        print("❌ No API keys are working")
    
    if working_keys:
        print(f"✅ {len(working_keys)} API keys are functional")
    else:
        print("❌ No API keys passed the test")
    
    if nlp_works:
        print("✅ NLP module summarization is working")
    else:
        print("❌ NLP module summarization failed")
    
    print("\n🔧 Recommendations:")
    if not single_key_works:
        print("1. Check if your API keys are valid and have Gemini API access enabled")
        print("2. Verify the keys are correctly formatted (should start with 'AIzaSy')")
        print("3. Check if the Gemini API is enabled in your Google Cloud Console")
        print("4. Ensure you have sufficient quota/credits")
    else:
        print("1. API keys are working - the issue might be elsewhere")
        print("2. Check for rate limiting or quota issues")
        print("3. Verify the nlp module is importing the keys correctly")

if __name__ == "__main__":
    main()
