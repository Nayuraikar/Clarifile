#!/usr/bin/env python3
"""
Check database for Drive files and create test duplicates
"""

import sqlite3
import os

DB_PATH = "metadata_db/clarifile.db"

def check_and_fix_drive_db():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        print(f"📊 Tables: {tables}")

        # Check files table schema
        print("\n=== FILES TABLE SCHEMA ===")
        cursor.execute("PRAGMA table_info(files)")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  {col[1]} ({col[2]}) - {'NULL' if col[3] else 'NOT NULL'}")

        # Get all files - try different column combinations for Drive files
        print("\n=== CHECKING FOR DRIVE FILES ===")

        # Try various possible column combinations for Drive files
        queries = [
            "SELECT id, file_path, file_name, category_id, proposed_label, final_label FROM files WHERE file_path IS NOT NULL",
            "SELECT id, file, category, proposed, final_label FROM files WHERE file IS NOT NULL",
            "SELECT id, url, name, category FROM files WHERE url IS NOT NULL",
            "SELECT id, drive_url, file_name, category FROM files WHERE drive_url IS NOT NULL",
            "SELECT id, path, title, category FROM files WHERE path IS NOT NULL"
        ]

        files_found = False
        for query in queries:
            try:
                cursor.execute(query + " LIMIT 5")
                files = cursor.fetchall()
                if files:
                    print(f"✅ Found files with query: {query}")
                    print(f"Sample files: {files}")
                    files_found = True
                    break
            except:
                continue

        if not files_found:
            print("❌ No files found with any query")

        # Create test Drive duplicates if none exist
        print("\n=== CREATING TEST DRIVE DUPLICATES ===")

        # Create test Drive files with identical content
        test_drive_files = [
            {
                "id": "drive_file_1",
                "name": "Important Document.pdf",
                "url": "https://drive.google.com/file/d/1abc123/view",
                "category": "Important Documents",
                "size": "1024000"
            },
            {
                "id": "drive_file_2",
                "name": "Important Document.pdf",
                "url": "https://drive.google.com/file/d/2def456/view",
                "category": "Important Documents",
                "size": "1024000"
            },
            {
                "id": "drive_file_3",
                "name": "Project Report.txt",
                "url": "https://drive.google.com/file/d/3ghi789/view",
                "category": "Reports",
                "size": "512000"
            },
            {
                "id": "drive_file_4",
                "name": "Project Report.txt",
                "url": "https://drive.google.com/file/d/4jkl012/view",
                "category": "Reports",
                "size": "512000"
            }
        ]

        # Try to insert test files
        for file_info in test_drive_files:
            try:
                # Try different insert patterns based on schema
                cursor.execute("""
                    INSERT OR IGNORE INTO files (id, file_path, file_name, category_id, proposed_label, final_label, size, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    file_info["id"],
                    file_info["url"],
                    file_info["name"],
                    1,  # category_id
                    file_info["category"],
                    file_info["category"],
                    file_info["size"],
                    "processed"
                ))
                print(f"✅ Created Drive test file: {file_info['name']}")
            except Exception as e:
                print(f"❌ Error creating file {file_info['name']}: {e}")

        conn.commit()

        # Check if files were created
        print("\n=== VERIFYING CREATED FILES ===")
        cursor.execute("SELECT id, file_path, file_name FROM files WHERE file_path LIKE '%drive.google.com%'")
        drive_files = cursor.fetchall()

        print(f"📁 Found {len(drive_files)} Drive files:")
        for fid, path, name in drive_files:
            print(f"  {name} (ID: {fid}) -> {path}")

        conn.close()

        if drive_files:
            print("✅ Test Drive duplicates created successfully!")
            print("\n💡 Now test the deduplication:")
            print("  1. Start: python -m uvicorn services.dedup.app:app --host 0.0.0.0 --port 8004")
            print("  2. Test: curl 'http://127.0.0.1:8004/duplicates'")
        else:
            print("❌ No Drive files found after creation")

    except Exception as e:
        print(f"❌ Error: {e}")
        conn.close()

if __name__ == "__main__":
    check_and_fix_drive_db()
