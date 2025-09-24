#!/usr/bin/env python3
"""
Test the analyze endpoint with real Drive files
"""
import requests
import json

def test_with_real_files():
    print("🔍 Testing Analyze with Real Drive Files")
    print("=" * 50)
    
    # Get available drive proposals
    try:
        response = requests.get("http://127.0.0.1:4000/drive/proposals", timeout=10)
        if response.status_code == 200:
            proposals = response.json()
            print(f"✅ Found {len(proposals)} Drive files:")
            
            for i, file in enumerate(proposals[:3]):  # Show first 3 files
                print(f"   {i+1}. {file['name']} (ID: {file['id']})")
            
            if proposals:
                # Test with the first file
                test_file = proposals[0]
                print(f"\n🧪 Testing with file: {test_file['name']}")
                
                analyze_data = {
                    "file": {
                        "id": test_file['id'],
                        "name": test_file['name'],
                        "mimeType": test_file.get('mimeType', 'text/plain'),
                        "parents": test_file.get('parents', [])
                    },
                    "q": "What is this document about?"
                }
                
                try:
                    response = requests.post(
                        "http://127.0.0.1:4000/drive/analyze", 
                        json=analyze_data, 
                        timeout=30
                    )
                    
                    print(f"📡 Status: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        print("✅ SUCCESS! Analyze endpoint working!")
                        print(f"📋 Summary: {result.get('summary', 'No summary')[:100]}...")
                        if result.get('qa'):
                            print(f"💬 Answer: {result['qa'].get('answer', 'No answer')[:100]}...")
                        return True
                    else:
                        print(f"❌ Error: {response.text}")
                        return False
                        
                except Exception as e:
                    print(f"❌ Request failed: {e}")
                    return False
            else:
                print("❌ No drive files found")
                return False
                
        else:
            print(f"❌ Could not get drive proposals: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ui_functionality():
    print("\n🎯 Testing UI Functionality")
    print("=" * 50)
    print("Now that the backend is working, try this in the UI:")
    print("1. Open your Clarifile app")
    print("2. Go to the Drive tab")
    print("3. Click 'Analyze' on any file")
    print("4. You should see:")
    print("   - Loading spinner on the button")
    print("   - 'Analyzing file...' notification")
    print("   - 'File selected for AI analysis!' notification")
    print("   - File highlighted with blue border")
    print("5. Go to AI Assistant tab and ask questions about the file")

if __name__ == "__main__":
    success = test_with_real_files()
    test_ui_functionality()
    
    if success:
        print("\n🎉 The Analyze button should now work perfectly!")
    else:
        print("\n⚠️  There might still be issues with the Drive file processing")
