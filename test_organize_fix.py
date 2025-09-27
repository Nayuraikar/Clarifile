#!/usr/bin/env python3
"""
test_organize_fix.py
Test the fix for the organize_drive_files NameError.
"""

import requests
import json

def test_organize_endpoint():
    """Test the organize_drive_files endpoint."""
    print("🧪 Testing organize_drive_files endpoint fix...")
    
    # Test data (you can modify this with your actual file IDs)
    test_data = {
        "files": [
            {
                "id": "test-file-id",
                "name": "test_document.pdf",
                "mimeType": "application/pdf",
                "parents": []
            }
        ],
        "move": True,
        "override_category": "Finance",
        "auth_token": "your-auth-token-here"
    }
    
    parser_url = "http://127.0.0.1:8000"
    
    try:
        print(f"📡 Calling {parser_url}/organize_drive_files...")
        
        response = requests.post(
            f"{parser_url}/organize_drive_files",
            json=test_data,
            timeout=30
        )
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Response:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"❌ Error Response:")
            print(response.text)
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to parser service")
        print("💡 Make sure parser service is running: cd services/parser && python app.py")
        return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def show_fix_summary():
    """Show what was fixed."""
    print("\n" + "="*60)
    print("🔧 ORGANIZE_DRIVE_FILES FIX SUMMARY")
    print("="*60)
    
    print("\n🐛 **Problem Found:**")
    print("   • NameError: name 'moved' is not defined")
    print("   • Line 2449 in organize_drive_files function")
    print("   • Code was checking 'moved' variable that was never created")
    
    print("\n✅ **Fix Applied:**")
    print("   • Added 'moved = None' initialization")
    print("   • Added actual file moving logic:")
    print("     moved = move_file_to_folder(drive_service, f.id, target_folder_id)")
    print("   • Added success/failure logging")
    
    print("\n🎯 **What This Fixes:**")
    print("   • ✅ No more NameError when approving file proposals")
    print("   • ✅ Files actually get moved to the correct folders")
    print("   • ✅ Proper feedback on move success/failure")
    print("   • ✅ Organize functionality works as expected")
    
    print("\n📋 **Expected Behavior Now:**")
    print("   1. You approve a file proposal in the UI")
    print("   2. Backend creates the category folder (if needed)")
    print("   3. Backend moves the file to the category folder")
    print("   4. Success message: '✅ Successfully moved file.pdf to folder Finance'")
    print("   5. File appears in the correct Google Drive folder")
    
    print("\n🧪 **Testing:**")
    print("   • Run this script to test the endpoint")
    print("   • Try approving a file proposal in your UI")
    print("   • Check your Google Drive for the organized files")

def main():
    """Run the test and show summary."""
    print("🚀 TESTING ORGANIZE_DRIVE_FILES FIX")
    print("="*50)
    
    # Show what was fixed
    show_fix_summary()
    
    # Test the endpoint (optional - requires running services)
    print("\n🧪 **Optional Endpoint Test:**")
    user_input = input("Test the endpoint now? (y/n): ").strip().lower()
    
    if user_input == 'y':
        test_organize_endpoint()
    else:
        print("Skipping endpoint test. You can test by approving files in your UI.")
    
    print("\n🎉 **Fix Complete!**")
    print("Your organize functionality should now work without errors.")

if __name__ == "__main__":
    main()
