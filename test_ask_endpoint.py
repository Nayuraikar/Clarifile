#!/usr/bin/env python3
"""
Test script to verify the /ask endpoint functionality
"""

import requests
import sqlite3
import os

def test_ask_endpoint():
    """Test the /ask endpoint with a simple request"""
    
    # Check if database exists and has files
    db_path = "/Users/nayana/Desktop/PROGRAMS/clarifile/metadata_db/clarifile.db"
    
    if not os.path.exists(db_path):
        print("❌ Database file does not exist")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if there are any files with full_text
        cursor.execute("SELECT id, file FROM files WHERE full_text IS NOT NULL AND full_text != ''")
        files = cursor.fetchall()
        
        if not files:
            print("❌ No files with full_text found in database")
            print("Available files:")
            cursor.execute("SELECT id, file, full_text FROM files")
            all_files = cursor.fetchall()
            for file_id, filename, full_text in all_files:
                print(f"  ID: {file_id}, File: {filename}, Has text: {bool(full_text)}")
            conn.close()
            return False
        
        # Get the first file with text
        file_id, filename = files[0]
        print(f"✅ Found file with text: ID {file_id}, Name: {filename}")
        
        # Get the full text
        cursor.execute("SELECT full_text FROM files WHERE id = ?", (file_id,))
        full_text = cursor.fetchone()[0]
        print(f"📄 Text length: {len(full_text)} characters")
        
        conn.close()
        
        # Test the ask endpoint
        base_url = "http://127.0.0.1:4000"
        test_question = "What is this document about?"
        
        print(f"🔄 Testing /ask endpoint with question: '{test_question}'")
        
        try:
            response = requests.get(
                f"{base_url}/ask",
                params={"file_id": file_id, "q": test_question},
                timeout=30
            )
            
            print(f"📡 Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Ask endpoint working!")
                print(f"📋 Response: {data}")
                return True
            else:
                print(f"❌ Ask endpoint returned status {response.status_code}")
                print(f"📄 Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to connect to ask endpoint: {e}")
            return False
            
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return False

def test_direct_parser_endpoint():
    """Test the parser endpoint directly"""
    print("\n🔄 Testing direct parser endpoint...")
    
    try:
        response = requests.get("http://127.0.0.1:8000/", timeout=5)
        print(f"📡 Parser status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Parser service is running")
            return True
        else:
            print(f"❌ Parser returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to parser: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Clarifile Ask Endpoint")
    print("=" * 50)
    
    # Test if parser is running
    parser_ok = test_direct_parser_endpoint()
    
    # Test the ask endpoint
    ask_ok = test_ask_endpoint()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  Parser Service: {'✅ OK' if parser_ok else '❌ FAIL'}")
    print(f"  Ask Endpoint: {'✅ OK' if ask_ok else '❌ FAIL'}")
    
    if not ask_ok:
        print("\n💡 Troubleshooting tips:")
        print("  1. Make sure the parser service is running on port 8000")
        print("  2. Make sure the gateway service is running on port 4000")
        print("  3. Make sure there are files with full_text in the database")
        print("  4. Run a scan first: curl -X POST http://127.0.0.1:4000/scan")
