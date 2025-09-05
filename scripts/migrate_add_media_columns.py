# scripts/migrate_add_media_columns.py
import sqlite3
import os

# Get project root (parent of scripts/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(BASE_DIR, "metadata_db", "clarifile.db")

conn = sqlite3.connect(DB)
cur = conn.cursor()

# Get existing columns in "files"
cur.execute("PRAGMA table_info(files)")
cols = [r[1] for r in cur.fetchall()]

# Add new columns if they don’t exist
if "transcript" not in cols:
    cur.execute("ALTER TABLE files ADD COLUMN transcript TEXT")

if "tags" not in cols:
    cur.execute("ALTER TABLE files ADD COLUMN tags TEXT")

conn.commit()
print("Migration complete: transcript & tags columns ensured.")
