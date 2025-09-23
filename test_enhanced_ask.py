#!/usr/bin/env python3
"""
Test script to verify the enhanced ask functionality with different file types.
This script tests real-time text extraction for images, audio, and video files.
"""

import requests
import sqlite3
import os
import time
from pathlib import Path

def test_enhanced_ask():
    """Test the enhanced ask endpoint with various file types"""
    
    # Test configuration
    GATEWAY_URL = "http://127.0.0.1:4000"
    PARSER_URL = "http://127.0.0.1:8000"
    DB_PATH = "/Users/nayana/Desktop/PROGRAMS/clarifile/services/parser/metadata_db/clarifile.db"
    
    print("=== Testing Enhanced Ask Functionality ===")
    
    # Check if services are running
    try:
        requests.get(f"{GATEWAY_URL}/health", timeout=5)
        print("✓ Gateway service is running")
    except:
        print("✗ Gateway service is not running. Please start it first.")
        return False
    
    try:
        requests.get(f"{PARSER_URL}/health", timeout=5)
        print("✓ Parser service is running")
    except:
        print("✗ Parser service is not running. Please start it first.")
        return False
    
    # Connect to database
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        print("✓ Database connection successful")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False
    
    # Test files and their expected content
    test_files = [
        {
            "name": "test_document.txt",
            "type": "text",
            "question": "What is artificial intelligence?",
            "expected_keywords": ["AI", "intelligence", "machine learning"]
        },
        {
            "name": "frame_0.jpg",  # Assuming this is an image with text
            "type": "image",
            "question": "What text is visible in this image?",
            "expected_keywords": []  # We don't know what's in the image
        },
        {
            "name": "temp_audio.wav",  # Assuming this is an audio file
            "type": "audio",
            "question": "What is being said in this audio?",
            "expected_keywords": []  # We don't know what's in the audio
        }
    ]
    
    # Check which test files exist
    available_files = []
    for test_file in test_files:
        file_path = f"/Users/nayana/Desktop/PROGRAMS/clarifile/{test_file['name']}"
        if os.path.exists(file_path):
            available_files.append(test_file)
            print(f"✓ Found test file: {test_file['name']} ({test_file['type']})")
        else:
            print(f"✗ Test file not found: {test_file['name']}")
    
    if not available_files:
        print("No test files available. Please create test files first.")
        return False
    
    # Get or create file records in database
    file_ids = {}
    for test_file in available_files:
        file_path = f"/Users/nayana/Desktop/PROGRAMS/clarifile/{test_file['name']}"
        
        # Check if file already exists in database
        cur.execute("SELECT id, full_text FROM files WHERE file_path=?", (file_path,))
        row = cur.fetchone()
        
        if row:
            file_id, full_text = row
            file_ids[test_file['name']] = file_id
            print(f"✓ Found existing record for {test_file['name']} (ID: {file_id})")
            
            # Clear full_text to test real-time extraction
            if full_text:
                cur.execute("UPDATE files SET full_text='' WHERE id=?", (file_id,))
                conn.commit()
                print(f"  Cleared existing text to test real-time extraction")
        else:
            # Create new file record
            try:
                import hashlib
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                
                size = os.path.getsize(file_path)
                cur.execute("""INSERT INTO files 
                             (file_path, file_name, file_hash, size, status, proposed_label, full_text)
                             VALUES (?, ?, ?, ?, ?, ?, ?)""",
                           (file_path, test_file['name'], file_hash, size, "new", "Uncategorized", ""))
                conn.commit()
                
                cur.execute("SELECT id FROM files WHERE file_path=?", (file_path,))
                file_id = cur.fetchone()[0]
                file_ids[test_file['name']] = file_id
                print(f"✓ Created new record for {test_file['name']} (ID: {file_id})")
                
            except Exception as e:
                print(f"✗ Failed to create record for {test_file['name']}: {e}")
                continue
    
    # Test ask functionality for each file
    print("\n=== Testing Ask Endpoint ===")
    
    for test_file in available_files:
        file_name = test_file['name']
        file_id = file_ids.get(file_name)
        
        if not file_id:
            print(f"✗ No file ID for {file_name}")
            continue
        
        print(f"\n--- Testing {file_name} ({test_file['type']}) ---")
        print(f"Question: {test_file['question']}")
        
        try:
            # Make request to ask endpoint
            response = requests.get(
                f"{GATEWAY_URL}/ask",
                params={"file_id": file_id, "q": test_file['question']},
                timeout=60  # Longer timeout for real-time extraction
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('ok'):
                    answer = data.get('answer', '')
                    score = data.get('score', 0)
                    context = data.get('context', '')
                    
                    print(f"✓ Ask successful")
                    print(f"  Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
                    print(f"  Score: {score}")
                    print(f"  Context length: {len(context)} chars")
                    
                    # Check if any expected keywords are in the answer
                    if test_file['expected_keywords']:
                        found_keywords = [kw for kw in test_file['expected_keywords'] 
                                        if kw.lower() in answer.lower()]
                        if found_keywords:
                            print(f"  Found keywords: {found_keywords}")
                        else:
                            print(f"  Expected keywords not found: {test_file['expected_keywords']}")
                    
                else:
                    error = data.get('error', 'Unknown error')
                    print(f"✗ Ask failed: {error}")
                    
            else:
                print(f"✗ HTTP error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"✗ Request timed out for {file_name}")
        except Exception as e:
            print(f"✗ Request failed for {file_name}: {e}")
    
    # Cleanup
    conn.close()
    print("\n=== Test Complete ===")
    return True

if __name__ == "__main__":
    test_enhanced_ask()
