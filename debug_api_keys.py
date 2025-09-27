#!/usr/bin/env python3
"""
debug_api_keys.py
Quick diagnostic for API key issues.
"""

import os
import sys

# Add parser service to path
parser_path = os.path.join(os.path.dirname(__file__), 'services', 'parser')
sys.path.insert(0, parser_path)

print("🔍 DEBUGGING API KEY LOADING")
print("="*40)

# Check file existence
keys_file = os.path.join(parser_path, 'gemini_keys.txt')
print(f"📁 Keys file: {keys_file}")
print(f"📁 Exists: {os.path.exists(keys_file)}")

if os.path.exists(keys_file):
    with open(keys_file, 'r') as f:
        content = f.read()
    
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    print(f"📊 Total lines: {len(lines)}")
    print(f"📝 First few keys:")
    for i, key in enumerate(lines[:3]):
        print(f"   {i+1}: {key[:25]}...")

# Test nlp module import
print(f"\n🔍 Testing nlp module import...")
try:
    import nlp
    print(f"✅ nlp module imported successfully")
    print(f"📊 API keys loaded: {len(nlp.GEMINI_API_KEYS)}")
    
    if nlp.GEMINI_API_KEYS:
        print(f"📝 First key: {nlp.GEMINI_API_KEYS[0][:25]}...")
        
        # Check for duplicates
        unique_keys = set(nlp.GEMINI_API_KEYS)
        print(f"📊 Unique keys: {len(unique_keys)}")
        if len(unique_keys) != len(nlp.GEMINI_API_KEYS):
            print(f"⚠️  Found {len(nlp.GEMINI_API_KEYS) - len(unique_keys)} duplicate keys")
    else:
        print("❌ No API keys loaded!")
        
        # Debug why keys weren't loaded
        print(f"🔍 Environment GEMINI_API_KEY: {'Set' if os.getenv('GEMINI_API_KEY') else 'Not set'}")
        print(f"🔍 Environment GEMINI_API_KEYS: {'Set' if os.getenv('GEMINI_API_KEYS') else 'Not set'}")
        
except Exception as e:
    print(f"❌ Error importing nlp: {e}")
    import traceback
    traceback.print_exc()

# Test a simple API call
print(f"\n🧪 Testing simple API call...")
try:
    import requests
    
    # Read first key directly from file
    with open(keys_file, 'r') as f:
        first_key = f.readline().strip()
    
    print(f"🔑 Using key: {first_key[:25]}...")
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": first_key
    }
    
    payload = {
        "contents": [{"parts": [{"text": "Say hello"}]}],
        "generationConfig": {"maxOutputTokens": 10}
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    print(f"📊 Response status: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ API call successful!")
    else:
        print(f"❌ API call failed: {response.text[:200]}")
        
except Exception as e:
    print(f"❌ API test failed: {e}")

print(f"\n🎯 DIAGNOSIS COMPLETE")
