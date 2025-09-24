#!/usr/bin/env python3
"""
Check the actual database schema and fix the dedup service accordingly
"""

import sqlite3
import os

DB_PATH = "metadata_db/clarifile.db"

def check_and_fix_schema():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    try:
        # Get all tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        print(f"📋 Tables in database: {tables}")

        # Check files table schema
        print("\n=== FILES TABLE SCHEMA ===")
        cur.execute("PRAGMA table_info(files)")
        columns = cur.fetchall()
        print("Columns:")
        for col in columns:
            print(f"  {col[1]} ({col[2]}) - {'NULL' if col[3] else 'NOT NULL'}")

        # Check what columns actually exist
        existing_columns = [col[1] for col in columns]
        print(f"\n✅ Existing columns: {existing_columns}")

        # Check if we can query the table
        print("\n=== TESTING QUERIES ===")
        try:
            # Try to select some basic columns
            query_cols = [col for col in existing_columns if col in ['id', 'file_path', 'file_name', 'proposed', 'proposed_label', 'category', 'final_label']]
            if query_cols:
                cur.execute(f"SELECT {', '.join(query_cols)} FROM files LIMIT 3")
                rows = cur.fetchall()
                print(f"✅ Successfully queried {len(rows)} rows with columns: {query_cols}")
                for row in rows:
                    print(f"  Row: {row}")
            else:
                print("❌ No compatible columns found")
        except Exception as e:
            print(f"❌ Error querying table: {e}")

    except Exception as e:
        print(f"❌ Error examining database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_and_fix_schema()
