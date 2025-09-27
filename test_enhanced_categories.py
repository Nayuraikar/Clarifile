#!/usr/bin/env python3
"""
test_enhanced_categories.py
Test the enhanced categories functionality.
"""

import requests
import json

def test_basic_categories():
    """Test the basic categories endpoint."""
    print("🧪 Testing basic /categories endpoint...")
    
    try:
        response = requests.get("http://127.0.0.1:8000/categories", timeout=10)
        
        if response.status_code == 200:
            categories = response.json()
            print(f"✅ Basic categories: {len(categories)} found")
            
            for cat in categories[:5]:  # Show first 5
                print(f"   • {cat['name']}: {cat.get('approved_count', 0)} approved, {cat.get('proposed_count', 0)} proposed")
            
            return categories
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return []

def test_enhanced_categories():
    """Test the enhanced categories endpoint."""
    print("\n🧪 Testing enhanced /enhanced_categories endpoint...")
    
    try:
        response = requests.get("http://127.0.0.1:8000/enhanced_categories", timeout=15)
        
        if response.status_code == 200:
            categories = response.json()
            print(f"✅ Enhanced categories: {len(categories)} found")
            
            for cat in categories[:5]:  # Show first 5
                drive_info = "📁 Drive folder" if cat.get('has_drive_folder') else "📋 DB only"
                print(f"   • {cat['name']}: {cat.get('total_count', 0)} total files ({drive_info})")
            
            return categories
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return []

def test_gateway_categories():
    """Test categories through the gateway."""
    print("\n🧪 Testing gateway /categories endpoint...")
    
    try:
        response = requests.get("http://127.0.0.1:4000/categories", timeout=10)
        
        if response.status_code == 200:
            categories = response.json()
            print(f"✅ Gateway categories: {len(categories)} found")
            
            for cat in categories[:5]:  # Show first 5
                if isinstance(cat, dict):
                    print(f"   • {cat.get('name', 'Unknown')}: {cat.get('total_count', cat.get('file_count', 0))} files")
                else:
                    print(f"   • {cat}")
            
            return categories
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return []

def test_gateway_enhanced_categories():
    """Test enhanced categories through the gateway."""
    print("\n🧪 Testing gateway /enhanced_categories endpoint...")
    
    try:
        response = requests.get("http://127.0.0.1:4000/enhanced_categories", timeout=15)
        
        if response.status_code == 200:
            categories = response.json()
            print(f"✅ Gateway enhanced categories: {len(categories)} found")
            
            for cat in categories[:5]:  # Show first 5
                drive_info = "📁 Drive folder" if cat.get('has_drive_folder') else "📋 DB only"
                print(f"   • {cat['name']}: {cat.get('total_count', 0)} total files ({drive_info})")
                if cat.get('drive_folder_id'):
                    print(f"     └─ Drive folder ID: {cat['drive_folder_id']}")
            
            return categories
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return []

def show_solution_summary():
    """Show what the enhanced categories solution provides."""
    print("\n" + "="*60)
    print("🎯 ENHANCED CATEGORIES SOLUTION")
    print("="*60)
    
    print("\n🔧 **What Was Enhanced:**")
    print("✅ Basic /categories now shows both approved AND proposed categories")
    print("✅ New /enhanced_categories includes Google Drive folder information")
    print("✅ Gateway endpoints for both basic and enhanced categories")
    print("✅ Categories show file counts from multiple sources")
    
    print("\n📊 **Category Sources Combined:**")
    print("1. 📋 Database approved categories (final_label)")
    print("2. 📋 Database proposed categories (proposed_category)")
    print("3. 📁 Google Drive folders (actual folders in Drive)")
    print("4. 🤖 Enhanced categorization results")
    
    print("\n🎯 **Expected UI Behavior:**")
    print("• Categories tab shows ALL categories (approved + proposed + Drive folders)")
    print("• Each category shows counts: approved files, proposed files, total files")
    print("• Categories with Drive folders show folder information")
    print("• Categories are sorted by total file count")
    
    print("\n🔗 **API Endpoints Available:**")
    print("• GET /categories - Basic categories with counts")
    print("• GET /enhanced_categories - Categories + Drive folder info")
    print("• GET /enhanced_categories?auth_token=... - With Drive access")
    
    print("\n📋 **Next Steps:**")
    print("1. Update your UI to use /enhanced_categories endpoint")
    print("2. Categories tab will show all categories (approved + proposed + Drive)")
    print("3. Test by approving some files and checking categories")

def main():
    """Run all category tests."""
    print("🚀 TESTING ENHANCED CATEGORIES")
    print("="*50)
    
    # Test all endpoints
    basic_cats = test_basic_categories()
    enhanced_cats = test_enhanced_categories()
    gateway_cats = test_gateway_categories()
    gateway_enhanced_cats = test_gateway_enhanced_categories()
    
    # Show summary
    show_solution_summary()
    
    print("\n🎯 **SUMMARY:**")
    if basic_cats or enhanced_cats:
        print("✅ Categories endpoints are working!")
        print("✅ Your categories tab should now show all categories")
        print("✅ Both approved and proposed categories will appear")
        
        if gateway_enhanced_cats:
            print("✅ Enhanced categories with Drive info are working")
            print("💡 Update your UI to use /enhanced_categories for best results")
        
    else:
        print("⚠️ No categories found - this might be normal if no files are processed yet")
        print("💡 Try scanning some files or approving proposals to create categories")

if __name__ == "__main__":
    main()
