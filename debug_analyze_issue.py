#!/usr/bin/env python3
"""
Comprehensive debugging script for the Analyze button issue
"""
import requests
import json
import subprocess
import time
import os
import signal
import sys

# Service configurations
SERVICES = {
    "parser": {"port": 8000, "command": "python app.py", "dir": "services/parser"},
    "embed": {"port": 8002, "command": "python app.py", "dir": "services/embed"},
    "indexer": {"port": 8003, "command": "python app.py", "dir": "services/indexer"},
    "dedup": {"port": 8004, "command": "python app.py", "dir": "services/dedup"},
    "gateway": {"port": 4000, "command": "node index.js", "dir": "gateway"}
}

def test_service_health(url, name):
    """Test if a service is healthy"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {name}: HEALTHY (Status: {response.status_code})")
            return True
        else:
            print(f"⚠️  {name}: RESPONDING but status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ {name}: DOWN (Connection Error)")
        return False
    except Exception as e:
        print(f"❌ {name}: ERROR - {e}")
        return False

def test_gateway_endpoints():
    """Test all gateway endpoints"""
    base_url = "http://127.0.0.1:4000"
    endpoints = [
        ("GET", "/drive/health"),
        ("GET", "/drive/proposals"),
        ("POST", "/drive/analyze", {
            "file": {"id": "test", "name": "test.txt", "mimeType": "text/plain", "parents": []},
            "q": "What is this about?"
        })
    ]
    
    print("\n🌐 Testing Gateway Endpoints:")
    print("-" * 40)
    
    for endpoint in endpoints:
        method = endpoint[0]
        path = endpoint[1]
        data = endpoint[2] if len(endpoint) > 2 else None
        
        url = f"{base_url}{path}"
        print(f"\n📡 Testing {method} {url}")
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=10)
            else:
                response = requests.post(url, json=data, timeout=30)
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                resp_data = response.json()
                print(f"   Response: {json.dumps(resp_data, indent=2)[:200]}...")
            else:
                print(f"   Error: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"   ❌ TIMEOUT: Request took too long")
        except requests.exceptions.ConnectionError:
            print(f"   ❌ CONNECTION ERROR: Cannot connect to gateway")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")

def check_service_logs():
    """Check if there are any log files with errors"""
    print("\n📋 Checking Service Logs:")
    print("-" * 40)
    
    log_files = [
        "services/parser/parser.log",
        "services/embed/app.log",
        "services/indexer/app.log",
        "services/dedup/app.log"
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            print(f"\n📄 {log_file}:")
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    if content.strip():
                        print(f"   {content[:500]}...")
                    else:
                        print("   (empty)")
            except Exception as e:
                print(f"   Error reading log: {e}")
        else:
            print(f"   📄 {log_file}: (not found)")

def check_database():
    """Check if the database exists and has data"""
    print("\n💾 Checking Database:")
    print("-" * 40)
    
    db_path = "metadata_db/clarifile.db"
    if os.path.exists(db_path):
        print(f"✅ Database exists: {db_path}")
        
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"📋 Tables: {[t[0] for t in tables]}")
            
            # Check files table
            cursor.execute("SELECT COUNT(*) FROM files")
            file_count = cursor.fetchone()[0]
            print(f"📄 Files in database: {file_count}")
            
            # Check files with text
            cursor.execute("SELECT COUNT(*) FROM files WHERE full_text IS NOT NULL AND full_text != ''")
            files_with_text = cursor.fetchone()[0]
            print(f"📝 Files with text: {files_with_text}")
            
            # Check drive proposals
            cursor.execute("SELECT COUNT(*) FROM drive_proposals")
            drive_count = cursor.fetchone()[0]
            print(f"📁 Drive proposals: {drive_count}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Database error: {e}")
    else:
        print(f"❌ Database not found: {db_path}")

def check_drive_token():
    """Check if there's a drive token issue"""
    print("\n🔑 Checking Drive Token Status:")
    print("-" * 40)
    
    try:
        response = requests.get("http://127.0.0.1:4000/drive/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"📊 Drive Health: {health}")
            if not health.get('hasToken', False):
                print("⚠️  NO DRIVE TOKEN - This might be the issue!")
                print("   You need to authenticate with Google Drive first.")
            else:
                print("✅ Drive token is available")
        else:
            print(f"❌ Could not check drive health: {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking drive health: {e}")

def main():
    print("🔍 COMPREHENSIVE DEBUG: Analyze Button Issue")
    print("=" * 60)
    
    # Test all services
    print("\n🔧 Testing Service Health:")
    print("-" * 40)
    
    for service_name, config in SERVICES.items():
        url = f"http://127.0.0.1:{config['port']}"
        test_service_health(url, service_name.upper())
    
    # Test gateway endpoints
    test_gateway_endpoints()
    
    # Check logs
    check_service_logs()
    
    # Check database
    check_database()
    
    # Check drive token
    check_drive_token()
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY & RECOMMENDATIONS:")
    print("=" * 60)
    
    print("\n🚀 If services are not running, start them with:")
    for service_name, config in SERVICES.items():
        print(f"   cd {config['dir']} && {config['command']}")
    
    print("\n🔧 Common Issues:")
    print("   1. Services not running - Start all services first")
    print("   2. No Drive token - Authenticate with Google Drive")
    print("   3. No files in database - Run scan first")
    print("   4. Network issues - Check firewall/port availability")
    
    print("\n📝 Next Steps:")
    print("   1. Start all services if they're not running")
    print("   2. Run: curl -X POST http://127.0.0.1:4000/scan")
    print("   3. Connect Google Drive and organize files")
    print("   4. Try the analyze button again")

if __name__ == "__main__":
    main()
