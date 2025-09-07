import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "metadata_db" / "clarifile.db"

DDL_TABLES_AND_INDEXES = """
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS entities (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  type TEXT NOT NULL,
  UNIQUE(name, type)
);

CREATE TABLE IF NOT EXISTS file_entities (
  file_id INTEGER NOT NULL,
  entity_id INTEGER NOT NULL,
  count INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (file_id, entity_id),
  FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
  FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS entity_edges (
  a INTEGER NOT NULL,
  b INTEGER NOT NULL,
  weight INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (a,b),
  FOREIGN KEY (a) REFERENCES entities(id) ON DELETE CASCADE,
  FOREIGN KEY (b) REFERENCES entities(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_file_entities_file ON file_entities(file_id);
CREATE INDEX IF NOT EXISTS idx_file_entities_ent ON file_entities(entity_id);
"""

def column_exists(con, table_name, column_name):
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cur.fetchall()]  # row[1] is the column name
    return column_name in columns

def add_full_text_column(con):
    if not column_exists(con, 'files', 'full_text'):
        con.execute("ALTER TABLE files ADD COLUMN full_text TEXT")
        print("Added column 'full_text' to 'files' table.")

def main():
    con = sqlite3.connect(str(DB_PATH))
    try:
        con.executescript(DDL_TABLES_AND_INDEXES)
        add_full_text_column(con)
        con.commit()
        print("Migration completed successfully.")
    finally:
        con.close()

if __name__ == "__main__":
    main()
