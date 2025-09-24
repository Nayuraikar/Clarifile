#!/usr/bin/env python3
"""
Simple test to manually add duplicate entries to database for testing deduplication
"""

import sqlite3
import os
import hashlib

def create_test_duplicates():
    """Create test duplicate entries in database"""

    db_path = "metadata_db/clarifile.db"

    if not os.path.exists(db_path):
        print("❌ Database file does not exist")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Read test_1.txt content
        test_file = "storage/sample_files/test_1.txt"
        if not os.path.exists(test_file):
            print("❌ Test file does not exist")
            return False

        with open(test_file, 'rb') as f:
            content = f.read()

        file_hash = hashlib.md5(content).hexdigest()
        file_size = len(content)

        # Add original file
        cursor.execute("""
            INSERT INTO files (file, hash, size, category, proposed, final_label, created_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """, ("test_1.txt", file_hash, file_size, "Test", "Test", "Test"))

        # Add duplicate files with same content hash
        for i in range(3):
            cursor.execute("""
                INSERT INTO files (file, hash, size, category, proposed, final_label, created_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            """, (f"test_1_duplicate_{i}.txt", file_hash, file_size, "Test", "Test", "Test"))

        conn.commit()
        conn.close()

        print("✅ Created test duplicates in database")
        print("💡 Now test the deduplication service:")
        print("  1. Start: python -m uvicorn services.dedup.app:app --host 0.0.0.0 --port 8004")
        print("  2. Test: curl 'http://127.0.0.1:8004/duplicates'")
        return True

    except Exception as e:
        print(f"❌ Error creating test duplicates: {e}")
        return False

if __name__ == "__main__":
    create_test_duplicates()
