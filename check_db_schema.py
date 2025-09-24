#!/usr/bin/env python3
"""
Check the actual database schema
"""

import sqlite3
import os

DB_PATH = "metadata_db/clarifile.db"

def check_schema():
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

        # Check files table schema
        print("\n=== FILES TABLE SCHEMA ===")
        cur.execute("PRAGMA table_info(files)")
        columns = cur.fetchall()
        print("Columns:")
        for col in columns:
            print(f"  {col[1]} ({col[2]}) - {'NULL' if col[3] else 'NOT NULL'}")

        # Try different possible column names
        possible_columns = ['proposed', 'proposed_label', 'proposed_category', 'category', 'label']
        
        print("\n=== CHECKING COLUMN EXISTENCE ===")
        for col in possible_columns:
            try:
                cur.execute(f"SELECT {col} FROM files LIMIT 1")
                result = cur.fetchone()
                print(f"✅ Column '{col}' exists")
                if result:
                    print(f"   Sample value: {result[0]}")
            except sqlite3.OperationalError as e:
                print(f"❌ Column '{col}' does not exist: {e}")

        # Try to select with actual existing columns
        print("\n=== TRYING TO QUERY WITH EXISTING COLUMNS ===")
        try:
            cur.execute("SELECT id, file_path, file_name FROM files LIMIT 3")
            rows = cur.fetchall()
            print(f"✅ Successfully queried files table with {len(rows)} rows")
            for row in rows:
                print(f"  ID {row[0]}: {row[1]} -> {row[2]}")
        except Exception as e:
            print(f"❌ Error querying files table: {e}")

    except Exception as e:
        print(f"Error examining database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_schema()
