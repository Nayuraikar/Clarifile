#!/usr/bin/env python3
"""
test_simple_categories.py
Simple test for categories functionality.
"""

import requests
import json

def test_parser_health():
    """Test if parser service is running."""
    print("🧪 Testing parser service health...")
    
    try:
        response = requests.get("http://127.0.0.1:8000/", timeout=5)
        print(f"✅ Parser service is running (status: {response.status_code})")
        return True
    except Exception as e:
        print(f"❌ Parser service not reachable: {e}")
        print("💡 Start it with: cd services/parser && python app.py")
        return False

def test_basic_categories():
    """Test basic categories endpoint."""
    print("\n🧪 Testing /categories endpoint...")
    
    try:
        response = requests.get("http://127.0.0.1:8000/categories", timeout=10)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            categories = response.json()
            print(f"✅ Categories response: {len(categories)} categories")
            
            if categories:
                print("📋 Categories found:")
                for cat in categories:
                    print(f"   • {cat.get('name', 'Unknown')}: {cat.get('total_count', 0)} files")
            else:
                print("📋 No categories found (this is normal if no files have been processed)")
            
            return True
        else:
            print(f"❌ Error response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_enhanced_categories():
    """Test enhanced categories endpoint."""
    print("\n🧪 Testing /enhanced_categories endpoint...")
    
    try:
        response = requests.get("http://127.0.0.1:8000/enhanced_categories", timeout=10)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            categories = response.json()
            print(f"✅ Enhanced categories response: {len(categories)} categories")
            
            if categories:
                print("📋 Enhanced categories found:")
                for cat in categories:
                    drive_status = "📁 Drive folder" if cat.get('has_drive_folder') else "📋 DB only"
                    print(f"   • {cat.get('name', 'Unknown')}: {cat.get('total_count', 0)} files ({drive_status})")
            else:
                print("📋 No categories found (this is normal if no files have been processed)")
            
            return True
        else:
            print(f"❌ Error response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def show_solution():
    """Show what the solution provides."""
    print("\n" + "="*60)
    print("🎯 CATEGORIES SOLUTION SUMMARY")
    print("="*60)
    
    print("\n✅ **What Was Fixed:**")
    print("• Enhanced /categories endpoint to show approved + proposed categories")
    print("• Added /enhanced_categories endpoint with Google Drive folder info")
    print("• Fixed function name error (build_drive_service → get_drive_service)")
    print("• Added proper error handling and fallbacks")
    
    print("\n📊 **Categories Now Include:**")
    print("• Database approved categories (final_label)")
    print("• Database proposed categories (proposed_category)")
    print("• Google Drive folders (when auth token provided)")
    print("• File counts for each category type")
    
    print("\n🎯 **Expected UI Behavior:**")
    print("• Categories tab shows ALL categories (not just approved)")
    print("• Each category shows approved + proposed file counts")
    print("• Categories with Drive folders show folder information")
    print("• Categories sorted by total activity")
    
    print("\n🔗 **Available Endpoints:**")
    print("• GET /categories - Basic categories with counts")
    print("• GET /enhanced_categories - Categories + Drive folder info")
    print("• Both available through gateway at port 4000")

def main():
    """Run the test."""
    print("🚀 TESTING CATEGORIES SOLUTION")
    print("="*50)
    
    # Test parser service
    if not test_parser_health():
        return
    
    # Test endpoints
    basic_works = test_basic_categories()
    enhanced_works = test_enhanced_categories()
    
    # Show solution
    show_solution()
    
    print("\n🎯 **RESULT:**")
    if basic_works and enhanced_works:
        print("✅ Categories endpoints are working!")
        print("✅ Your categories tab should now show all categories")
        print("✅ Try refreshing categories in your UI")
    elif basic_works:
        print("✅ Basic categories working")
        print("⚠️ Enhanced categories may need Drive token")
        print("💡 Your categories tab should still show more categories now")
    else:
        print("❌ Categories endpoints need debugging")
        print("💡 Check parser service logs for errors")

if __name__ == "__main__":
    main()
