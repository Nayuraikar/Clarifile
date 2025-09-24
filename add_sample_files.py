#!/usr/bin/env python3
"""
Script to add existing files from storage/sample_files to the database for testing deduplication
"""

import sqlite3
import os
import hashlib
from pathlib import Path

def add_files_to_database():
    """Add existing files from storage/sample_files to the database"""

    # Paths
    db_path = "metadata_db/clarifile.db"
    storage_dir = "storage/sample_files"

    # Check if files exist
    if not os.path.exists(db_path):
        print("❌ Database file does not exist")
        return False

    if not os.path.exists(storage_dir):
        print("❌ Storage directory does not exist")
        return False

    try:
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

        # Get existing files in database
        cursor.execute("SELECT file FROM files")
        existing_files = {row[0] for row in cursor.fetchall()}

        # Find all files in storage directory
        storage_path = Path(storage_dir)
        added_count = 0

        for file_path in storage_path.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                filename = file_path.name

                # Skip if already in database
                if filename in existing_files:
                    print(f"⏭️  Skipping existing file: {filename}")
                    continue

                # Read the file content
                try:
                    with open(file_path, 'rb') as f:
                        file_content = f.read()

                    # Calculate file hash
                    file_hash = hashlib.md5(file_content).hexdigest()

                    # Get file size
                    file_size = len(file_content)

                    # Insert the file
                    cursor.execute("""
                        INSERT INTO files (file, hash, size, category, proposed, final_label, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                    """, (
                        filename,
                        file_hash,
                        file_size,
                        "Sample",
                        "Sample",
                        "Sample"
                    ))

                    added_count += 1
                    print(f"✅ Added file: {filename} ({file_size} bytes)")

                except Exception as e:
                    print(f"❌ Error adding file {filename}: {e}")

        conn.commit()
        conn.close()

        print(f"\n📊 Added {added_count} files to database")
        return True

    except Exception as e:
        print(f"❌ Error adding files to database: {e}")
        return False

def check_database_content():
    """Check what's currently in the database"""

    db_path = "metadata_db/clarifile.db"

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
                cursor.execute("SELECT id, file, category, final_label, size FROM files")
                files = cursor.fetchall()
                print("\n📋 Files list:")
                for file_id, filename, category, final_label, size in files:
                    print(f"  ID: {file_id}, File: {filename}, Category: {category}, Final: {final_label}, Size: {size} bytes")

        conn.close()

    except Exception as e:
        print(f"❌ Error checking database: {e}")

if __name__ == "__main__":
    print("📝 Adding Sample Files to Database")
    print("=" * 50)

    # Check current database content
    check_database_content()

    print("\n" + "=" * 50)

    # Add files
    success = add_files_to_database()

    if success:
        print("\n" + "=" * 50)
        print("✅ Files added successfully!")
        print("\n💡 Now you can test the deduplication:")
        print("  1. Start the dedup service: python -m uvicorn services.dedup.app:app --host 0.0.0.0 --port 8004")
        print("  2. Test: curl 'http://127.0.0.1:8004/duplicates'")
    else:
        print("\n❌ Failed to add files")
