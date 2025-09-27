#!/usr/bin/env python3
"""
validate_api_keys.py
Quickly validate which Gemini API keys are working.
"""

import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def test_single_key(key_info):
    """Test a single API key."""
    index, key = key_info
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": key
    }
    
    payload = {
        "contents": [{"parts": [{"text": "Respond with just 'OK'"}]}],
        "generationConfig": {"maxOutputTokens": 5}
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            return {"index": index, "key": key[:25], "status": "✅ WORKING", "error": None}
        elif response.status_code == 401:
            return {"index": index, "key": key[:25], "status": "❌ INVALID", "error": "401 Unauthorized"}
        elif response.status_code == 403:
            return {"index": index, "key": key[:25], "status": "❌ FORBIDDEN", "error": "403 Forbidden"}
        elif response.status_code == 429:
            return {"index": index, "key": key[:25], "status": "⚠️ RATE_LIMITED", "error": "429 Rate Limited"}
        else:
            return {"index": index, "key": key[:25], "status": "❌ ERROR", "error": f"HTTP {response.status_code}"}
            
    except requests.exceptions.Timeout:
        return {"index": index, "key": key[:25], "status": "⏰ TIMEOUT", "error": "Request timeout"}
    except Exception as e:
        return {"index": index, "key": key[:25], "status": "❌ EXCEPTION", "error": str(e)}

def main():
    """Validate all API keys."""
    print("🔍 GEMINI API KEY VALIDATION")
    print("="*50)
    
    # Read keys
    keys_file = os.path.join(os.path.dirname(__file__), 'services', 'parser', 'gemini_keys.txt')
    
    if not os.path.exists(keys_file):
        print(f"❌ Keys file not found: {keys_file}")
        return
    
    with open(keys_file, 'r') as f:
        keys = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"📊 Found {len(keys)} API keys to test")
    print("🔄 Testing keys (this may take a moment)...")
    
    # Test keys in parallel (but with some rate limiting)
    results = []
    
    # Test in smaller batches to avoid overwhelming the API
    batch_size = 5
    for i in range(0, len(keys), batch_size):
        batch = keys[i:i+batch_size]
        batch_with_index = [(i+j, key) for j, key in enumerate(batch)]
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(test_single_key, key_info) for key_info in batch_with_index]
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"   Key {result['index']+1:2d}: {result['status']} - {result['key']}...")
        
        # Small delay between batches
        if i + batch_size < len(keys):
            time.sleep(2)
    
    # Summary
    print("\n" + "="*50)
    print("📊 VALIDATION SUMMARY")
    print("="*50)
    
    working = [r for r in results if "WORKING" in r['status']]
    invalid = [r for r in results if "INVALID" in r['status']]
    forbidden = [r for r in results if "FORBIDDEN" in r['status']]
    rate_limited = [r for r in results if "RATE_LIMITED" in r['status']]
    errors = [r for r in results if "ERROR" in r['status'] or "EXCEPTION" in r['status'] or "TIMEOUT" in r['status']]
    
    print(f"✅ Working keys: {len(working)}")
    print(f"❌ Invalid keys: {len(invalid)}")
    print(f"❌ Forbidden keys: {len(forbidden)}")
    print(f"⚠️ Rate limited keys: {len(rate_limited)}")
    print(f"❌ Error/timeout keys: {len(errors)}")
    
    if working:
        print(f"\n🎉 Good news! {len(working)} keys are working:")
        for result in working[:5]:  # Show first 5 working keys
            print(f"   Key {result['index']+1}: {result['key']}...")
    else:
        print(f"\n⚠️ No working keys found!")
        print("\n🔧 Troubleshooting suggestions:")
        print("1. Check if Gemini API is enabled in Google Cloud Console")
        print("2. Verify API keys have the correct permissions")
        print("3. Check if you have sufficient quota/credits")
        print("4. Ensure keys are not expired")
        
        if invalid:
            print(f"5. {len(invalid)} keys are invalid - they may be malformed or deleted")
        if forbidden:
            print(f"6. {len(forbidden)} keys are forbidden - check API permissions")
        if rate_limited:
            print(f"7. {len(rate_limited)} keys are rate limited - try again later")
    
    # Create a working keys file
    if working:
        working_keys_file = os.path.join(os.path.dirname(__file__), 'services', 'parser', 'working_gemini_keys.txt')
        with open(working_keys_file, 'w') as f:
            for result in working:
                # Get the full key from the original list
                full_key = keys[result['index']]
                f.write(full_key + '\n')
        
        print(f"\n💾 Created working keys file: working_gemini_keys.txt")
        print(f"   Contains {len(working)} verified working keys")

if __name__ == "__main__":
    main()
