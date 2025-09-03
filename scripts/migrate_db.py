# scripts/migrate_db.py
import sqlite3, os, sys, json

DB = "metadata_db/clarifile.db"

if not os.path.exists(DB):
    print("ERROR: database not found at", DB)
    sys.exit(1)

conn = sqlite3.connect(DB)
cur = conn.cursor()

def has_column(table, col):
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    return col in cols

changes = []

# Add summary column
if not has_column("files", "summary"):
    cur.execute("ALTER TABLE files ADD COLUMN summary TEXT")
    changes.append("files.summary added")

# Add category_id column
if not has_column("files", "category_id"):
    cur.execute("ALTER TABLE files ADD COLUMN category_id INTEGER")
    changes.append("files.category_id added")

# Create categories table (if missing)
cur.execute("""
CREATE TABLE IF NOT EXISTS categories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT,
  rep_summary TEXT,
  rep_vector TEXT
)
""")
changes.append("categories table ensured")

conn.commit()

# Output status
print("Migration completed. Changes:", changes)

# Show current schema summary (for verification)
cur.execute("PRAGMA table_info(files)")
print("files table columns:", [r[1] for r in cur.fetchall()])

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print("Tables:", tables)

conn.close()