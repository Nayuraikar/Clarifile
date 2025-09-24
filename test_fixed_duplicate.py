#!/usr/bin/env python3
"""
Quick test to check if the dedup service is working after database schema fixes
"""

import requests
import json

def test_dedup_service():
    """Test the dedup service directly"""
    print("=== TESTING DEDUP SERVICE ===")
    
    try:
        # Test the duplicates endpoint
        response = requests.get('http://127.0.0.1:8004/duplicates', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Dedup service is responding")
            print(f"📊 Found {data.get('summary', {}).get('duplicate_groups_found', 0)} duplicate groups")
            print(f"📁 Processed {data.get('summary', {}).get('total_files_processed', 0)} files")
            
            # Show duplicate groups
            duplicates = data.get('duplicates', [])
            if duplicates:
                print("\n🔍 Duplicate Groups:")
                for group in duplicates:
                    print(f"  Group {group['group_id']}: {group['file_count']} files")
                    for file_info in group['files'][:2]:  # Show first 2 files
                        print(f"    - {file_info['name']} (ID: {file_info['id']})")
            else:
                print("ℹ️  No duplicates found")
                
            return True
        else:
            print(f"❌ Dedup service error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to dedup service (port 8004)")
        print("💡 Try starting it with: python -m uvicorn services.dedup.app:app --host 0.0.0.0 --port 8004")
        return False
    except Exception as e:
        print(f"❌ Error testing dedup service: {e}")
        return False

def test_gateway():
    """Test the gateway endpoints"""
    print("\n=== TESTING GATEWAY ===")
    
    try:
        # Test the duplicates endpoint
        response = requests.get('http://127.0.0.1:4000/duplicates', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Gateway is responding")
            print(f"📊 Found {data.get('summary', {}).get('duplicate_groups_found', 0)} duplicate groups")
            print(f"📁 Processed {data.get('summary', {}).get('total_files_processed', 0)} files")
            
            # Show duplicate groups
            duplicates = data.get('duplicates', [])
            if duplicates:
                print("\n🔍 Duplicate Groups:")
                for group in duplicates:
                    print(f"  Group {group['group_id']}: {group['file_count']} files")
                    for file_info in group['files'][:2]:  # Show first 2 files
                        print(f"    - {file_info['name']} (ID: {file_info['id']})")
            else:
                print("ℹ️  No duplicates found")
                
            return True
        else:
            print(f"❌ Gateway error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to gateway (port 4000)")
        print("💡 Try starting it with: node gateway/index.js")
        return False
    except Exception as e:
        print(f"❌ Error testing gateway: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Duplicate Functionality After Schema Fix")
    print("=" * 60)
    
    # Test dedup service directly
    dedup_ok = test_dedup_service()
    
    # Test gateway
    gateway_ok = test_gateway()
    
    if dedup_ok and gateway_ok:
        print("\n✅ All tests passed! Duplicate functionality should be working.")
    else:
        print("\n❌ Some services are not responding. Make sure they're running.")
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")
