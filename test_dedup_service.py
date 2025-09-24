#!/usr/bin/env python3
"""
Test the deduplication service
"""

import requests
import json
import sqlite3
import os

def test_database():
    """Check database contents"""
    print("=== DATABASE CHECK ===")
    DB = 'metadata_db/clarifile.db'
    if not os.path.exists(DB):
        print("❌ Database not found")
        return False
    
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM files')
    count = cur.fetchone()
    print(f"📄 Files in database: {count[0]}")
    
    if count[0] > 0:
        cur.execute('SELECT id, file, size FROM files')
        files = cur.fetchall()
        print("Files:")
        for f in files:
            print(f"  {f[0]}: {f[1]} ({f[2]} bytes)")
    conn.close()
    return True

def test_dedup_service():
    """Test the deduplication service"""
    print("\n=== DEDUPLICATION SERVICE TEST ===")
    
    try:
        # Test the duplicates endpoint
        response = requests.get('http://127.0.0.1:8004/duplicates', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Service is responding")
            print(f"📊 Found {data.get('summary', {}).get('duplicate_groups_found', 0)} duplicate groups")
            print(f"📁 Processed {data.get('summary', {}).get('total_files_processed', 0)} files")
            
            # Show duplicate groups
            duplicates = data.get('duplicates', [])
            if duplicates:
                print("\n🔍 Duplicate Groups:")
                for group in duplicates:
                    print(f"  Group {group['group_id']}: {group['file_count']} files")
                    for file_info in group['files']:
                        print(f"    - {file_info['name']} (ID: {file_info['id']})")
            else:
                print("ℹ️  No duplicates found")
                
            return True
        else:
            print(f"❌ Service error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to deduplication service")
        print("💡 Try starting it with: python -m uvicorn services.dedup.app:app --host 0.0.0.0 --port 8004")
        return False
    except Exception as e:
        print(f"❌ Error testing service: {e}")
        return False

def test_debug_endpoint():
    """Test the debug endpoint"""
    print("\n=== DEBUG ENDPOINT TEST ===")
    
    try:
        response = requests.get('http://127.0.0.1:8004/duplicates/debug', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Debug endpoint working")
            debug_info = data.get('debug_info', [])
            print(f"📊 Debug info for {len(debug_info)} files")
            
            for info in debug_info:
                print(f"  {info['name']}: exists={info['exists']}, size={info['size']}")
                
            return True
        else:
            print(f"❌ Debug endpoint error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing debug endpoint: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Deduplication Service")
    print("=" * 50)
    
    # Check database
    db_ok = test_database()
    
    if db_ok:
        print("\n" + "=" * 50)
        
        # Test deduplication service
        service_ok = test_dedup_service()
        
        if service_ok:
            print("\n" + "=" * 50)
            test_debug_endpoint()
    
    print("\n" + "=" * 50)
    print("✅ Test complete!")
