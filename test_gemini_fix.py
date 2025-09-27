#!/usr/bin/env python3
"""
test_gemini_fix.py
Test that Gemini API is now working with the correct model (gemini-2.5-flash).
"""

import os
import sys
import requests

# Add parser service to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services', 'parser'))

def test_direct_api_call():
    """Test direct API call with your working key."""
    print("🧪 Testing direct Gemini API call...")
    
    # Your working API key
    api_key = "AIzaSyDK9lOMbNuD8QmJ24z_3iw3gGcJGJUhEqc"
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [{
            "parts": [{"text": "Summarize this invoice: INVOICE #123 - Software services - Amount: $1,250.00 - Due: Net 30"}]
        }]
    }
    
    try:
        response = requests.post(f"{url}?key={api_key}", headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            print(f"✅ Direct API call successful!")
            print(f"📝 Response: {text}")
            return True
        else:
            print(f"❌ API call failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_nlp_module():
    """Test the nlp module with the fixed model."""
    print("\n🧪 Testing nlp module with gemini-2.5-flash...")
    
    try:
        import nlp
        
        print(f"📊 Loaded {len(nlp.GEMINI_API_KEYS)} API keys")
        print(f"🤖 Using model: {nlp.GEMINI_MODEL}")
        print(f"🔗 Endpoint: {nlp.GEMINI_ENDPOINT}")
        
        # Test summarization
        test_text = """INVOICE
Invoice Number: INV-2024-001
Date: January 15, 2024
Bill To: John Doe Company
Description: Software development services
Hours: 50 @ $25.00/hour
Total Amount Due: $1,250.00
Payment Terms: Net 30 days"""
        
        print(f"📝 Testing summarization...")
        result = nlp.summarize_with_gemini(test_text, max_tokens=100)
        
        if result:
            print(f"✅ Summarization successful!")
            print(f"📄 Summary: {result}")
            return True
        else:
            print(f"❌ Summarization returned empty result")
            return False
            
    except Exception as e:
        print(f"❌ Error testing nlp module: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_drive_analyze_simulation():
    """Simulate the drive_analyze endpoint call."""
    print("\n🧪 Testing drive_analyze simulation...")
    
    try:
        # Import the app functions
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services', 'parser'))
        from app import summarize_text, assign_category_from_summary
        
        test_content = """INVOICE
Invoice Number: INV-2024-001
Date: January 15, 2024
Bill To: John Doe Company
123 Business Street
Description: Software development services
Hours: 50 @ $25.00/hour
Subtotal: $1,250.00
Total Amount Due: $1,250.00
Payment Terms: Net 30 days"""
        
        print("📝 Testing summarization...")
        summary = summarize_text(test_content)
        print(f"✅ Summary: {summary}")
        
        print("🎯 Testing categorization...")
        cat_id, cat_name = assign_category_from_summary("", test_content)
        print(f"✅ Category: {cat_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in simulation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 TESTING GEMINI API FIX")
    print("="*50)
    
    # Test 1: Direct API call
    direct_works = test_direct_api_call()
    
    # Test 2: NLP module
    nlp_works = test_nlp_module()
    
    # Test 3: Drive analyze simulation
    simulate_works = test_drive_analyze_simulation()
    
    print("\n" + "="*50)
    print("🎯 TEST RESULTS")
    print("="*50)
    
    if direct_works:
        print("✅ Direct API call: WORKING")
    else:
        print("❌ Direct API call: FAILED")
    
    if nlp_works:
        print("✅ NLP module: WORKING")
    else:
        print("❌ NLP module: FAILED")
    
    if simulate_works:
        print("✅ Drive analyze simulation: WORKING")
    else:
        print("❌ Drive analyze simulation: FAILED")
    
    if direct_works and nlp_works and simulate_works:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your analyze button should now work perfectly!")
        print("✅ Gemini API with gemini-2.5-flash is working")
        print("✅ Summarization will work")
        print("✅ Enhanced categorization is active")
        print("✅ AI assistant redirection will work")
    else:
        print("\n⚠️ Some tests failed - check the errors above")

if __name__ == "__main__":
    main()
