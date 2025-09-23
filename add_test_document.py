#!/usr/bin/env python3
"""
Script to add a test document to the database for testing the ask functionality
"""

import sqlite3
import os
import hashlib

def add_test_document():
    """Add the test document to the database"""
    
    # Paths
    db_path = "/Users/nayana/Desktop/PROGRAMS/clarifile/metadata_db/clarifile.db"
    doc_path = "/Users/nayana/Desktop/PROGRAMS/clarifile/test_document.txt"
    
    # Check if files exist
    if not os.path.exists(db_path):
        print("❌ Database file does not exist")
        return False
    
    if not os.path.exists(doc_path):
        print("❌ Test document does not exist")
        return False
    
    try:
        # Read the document
        with open(doc_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        # Calculate file hash
        with open(doc_path, 'rb') as f:
            file_content = f.read()
        file_hash = hashlib.md5(file_content).hexdigest()
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if database has the required tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        if 'files' not in tables:
            print("❌ 'files' table does not exist in database")
            conn.close()
            return False
        
        # Insert the test document
        cursor.execute("""
            INSERT INTO files (file, full_text, hash, size, category, proposed, final_label, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            "test_document.txt",
            full_text,
            file_hash,
            len(file_content),
            "Test",
            "Test",
            "Test"
        ))
        
        file_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✅ Test document added to database")
        print(f"  File ID: {file_id}")
        print(f"  Filename: test_document.txt")
        print(f"  Text length: {len(full_text)} characters")
        print(f"  File hash: {file_hash}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error adding test document: {e}")
        return False

def check_database_content():
    """Check what's currently in the database"""
    
    db_path = "/Users/nayana/Desktop/PROGRAMS/clarifile/metadata_db/clarifile.db"
    
    if not os.path.exists(db_path):
        print("❌ Database file does not exist")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"📋 Tables in database: {[t[0] for t in tables]}")
        
        # Check files table
        if 'files' in [t[0] for t in tables]:
            cursor.execute("SELECT COUNT(*) FROM files")
            file_count = cursor.fetchone()[0]
            print(f"📄 Files in database: {file_count}")
            
            if file_count > 0:
                cursor.execute("SELECT id, file, category, final_label, length(full_text) as text_len FROM files")
                files = cursor.fetchall()
                print("\n📋 Files list:")
                for file_id, filename, category, final_label, text_len in files:
                    print(f"  ID: {file_id}, File: {filename}, Category: {category}, Final: {final_label}, Text: {text_len} chars")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")

if __name__ == "__main__":
    print("📝 Adding Test Document to Database")
    print("=" * 50)
    
    # Check current database content
    check_database_content()
    
    print("\n" + "=" * 50)
    
    # Add test document
    success = add_test_document()
    
    if success:
        print("\n" + "=" * 50)
        print("✅ Test document added successfully!")
        print("\n💡 Now you can test the ask endpoint:")
        print("  1. Run: python test_ask_endpoint.py")
        print("  2. Or use: curl 'http://127.0.0.1:4000/ask?file_id=1&q=What is this document about?'")
    else:
        print("\n❌ Failed to add test document")
