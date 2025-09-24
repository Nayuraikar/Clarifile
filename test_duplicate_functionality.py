#!/usr/bin/env python3
"""
Quick test to check database contents and duplicate functionality
"""

import sqlite3
import os
import requests

def check_database():
    """Check database contents"""
    print("=== DATABASE CHECK ===")
    DB = '/Users/nayana/Desktop/PROGRAMS/clarifile/metadata_db/clarifile.db'
    if not os.path.exists(DB):
        print("❌ Database not found")
        return False
    
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM files')
    count = cur.fetchone()
    print(f"📄 Files in database: {count[0]}")
    
    if count[0] > 0:
        cur.execute('SELECT id, file, size FROM files LIMIT 5')
        files = cur.fetchall()
        print("First 5 files:")
        for f in files:
            print(f"  {f[0]}: {f[1]} ({f[2]} bytes)")
    conn.close()
    return True

def test_gateway():
    """Test the gateway endpoints"""
    print("\n=== GATEWAY TEST ===")
    
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
    print("🧪 Testing Duplicate Functionality")
    print("=" * 50)
    
    # Check database
    db_ok = check_database()
    
    if db_ok:
        print("\n" + "=" * 50)
        
        # Test gateway
        gateway_ok = test_gateway()
        
        if gateway_ok:
            print("\n✅ All tests passed! Duplicate functionality should be working.")
        else:
            print("\n❌ Gateway not responding. Make sure it's running on port 4000.")
    
    print("\n" + "=" * 50)
    print("✅ Test complete!")
