#!/usr/bin/env python3
"""
Check database for Drive files and categories
"""

import sqlite3
import os

DB_PATH = "metadata_db/clarifile.db"

def check_drive_database():
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

        # Get all files with their categories
        print("\n=== FILES BY CATEGORY ===")

        # Try different possible column combinations
        try:
            cursor.execute("SELECT id, file_path, file_name, category_id, proposed_label, final_label FROM files WHERE file_path IS NOT NULL")
            files = cursor.fetchall()
            if files:
                print(f"✅ Found {len(files)} files")

                # Group by category
                categories = {}
                for fid, path, name, cat_id, proposed, final in files:
                    category = final or proposed or f"Category_{cat_id}" if cat_id else "Uncategorized"
                    if category not in categories:
                        categories[category] = []
                    categories[category].append((fid, name, path))

                # Show files by category
                for category, file_list in categories.items():
                    print(f"\n📂 {category} ({len(file_list)} files):")
                    for fid, name, path in file_list[:5]:  # Show first 5
                        print(f"  {name} (ID: {fid})")
                    if len(file_list) > 5:
                        print(f"  ... and {len(file_list) - 5} more")

                # Check for duplicates within each category
                print("
=== CHECKING FOR DUPLICATES WITHIN CATEGORIES ===")
                duplicates_found = 0
                for category, file_list in categories.items():
                    if len(file_list) <= 1:
                        continue

                    # Check for duplicate names
                    name_counts = {}
                    for fid, name, path in file_list:
                        if name not in name_counts:
                            name_counts[name] = []
                        name_counts[name].append((fid, path))

                    for name, name_files in name_counts.items():
                        if len(name_files) > 1:
                            duplicates_found += 1
                            print(f"\n🔍 DUPLICATE in {category}: {name}")
                            for fid, path in name_files:
                                print(f"  - ID {fid}: {path}")

                if duplicates_found == 0:
                    print("❌ No duplicates found in any category")

            else:
                print("❌ No files found with expected columns")

        except Exception as e:
            print(f"❌ Error querying files: {e}")

        # Try alternative column names
        alternative_queries = [
            "SELECT id, file, category FROM files",
            "SELECT id, path, name, category FROM files",
            "SELECT id, url, title, category FROM files",
            "SELECT id, drive_url, file_name, category FROM files"
        ]

        for query in alternative_queries:
            try:
                cursor.execute(query + " LIMIT 3")
                results = cursor.fetchall()
                if results:
                    print(f"\n✅ Alternative query worked: {query}")
                    print(f"Sample results: {results}")
                    break
            except:
                pass

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_drive_database()
