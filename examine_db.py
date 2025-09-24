#!/usr/bin/env python3
import sqlite3
import os
from pathlib import Path

DB_PATH = "metadata_db/clarifile.db"

def examine_database():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    try:
        # Get all tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        print(f"Tables in database: {tables}")

        # Examine files table
        print("\n=== FILES TABLE ===")
        cur.execute("PRAGMA table_info(files)")
        columns = cur.fetchall()
        print("Columns:", [col[1] for col in columns])

        # Get all files
        cur.execute("SELECT id, file_path, file_name, proposed_label, final_label, status FROM files")
        files = cur.fetchall()

        print(f"\nTotal files in database: {len(files)}")

        existing_files = []
        missing_files = []

        for file_row in files:
            file_id, file_path, file_name, proposed_label, final_label, status = file_row
            exists = os.path.exists(file_path)
            size = os.path.getsize(file_path) if exists else 0

            file_info = {
                'id': file_id,
                'path': file_path,
                'name': file_name,
                'proposed_label': proposed_label,
                'final_label': final_label,
                'status': status,
                'exists': exists,
                'size': size
            }

            if exists:
                existing_files.append(file_info)
            else:
                missing_files.append(file_info)

        print(f"\nExisting files: {len(existing_files)}")
        for f in existing_files:
            print(f"  ID {f['id']}: {f['name']} -> {f['path']} ({f['size']} bytes)")

        print(f"\nMissing files: {len(missing_files)}")
        for f in missing_files:
            print(f"  ID {f['id']}: {f['name']} -> {f['path']} (MISSING)")

        # Check if there are any files in storage that aren't in DB
        storage_dir = Path("storage/sample_files")
        if storage_dir.exists():
            print(f"\n=== CHECKING STORAGE DIRECTORY ===")
            db_paths = {f['path'] for f in files}
            storage_files = []

            for file_path in storage_dir.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    rel_path = str(file_path.relative_to(storage_dir.parent.parent))
                    if rel_path not in db_paths:
                        storage_files.append(rel_path)

            if storage_files:
                print(f"Files in storage but not in database: {len(storage_files)}")
                for sf in storage_files:
                    print(f"  {sf}")
            else:
                print("All storage files are in database"
    except Exception as e:
        print(f"Error examining database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    examine_database()
