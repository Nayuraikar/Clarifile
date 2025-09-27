"""
FastAPI app for dedup service with exact content comparison:
- PDFs: extract text, normalize it, and hash
- Text files: hash raw bytes (exact content)
- Other/binary files: stream and hash raw bytes (memory-efficient)
"""

import os
import re
import sqlite3
import hashlib
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI
import fitz  # PyMuPDF

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dedup")

# --- Configuration ---
DB_PATH = os.environ.get("CLARIFILE_DB_PATH", "metadata_db/clarifile.db")
HASH_ALGO = "sha256"
STREAM_CHUNK_SIZE = 1024 * 1024  # 1 MB

TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".html", ".css", ".json", ".xml", ".csv", ".log"
}

# --- Utilities ---

def normalize_text_content(content: str) -> str:
    # Remove NULL/control chars, normalize whitespace
    content = re.sub(r"[\x00-\x1F\x7F]", "", content)
    return re.sub(r"\s+", " ", content.strip())

def _hash_bytes(data: bytes) -> str:
    h = hashlib.new(HASH_ALGO)
    h.update(data)
    return h.hexdigest()

def _hash_file_stream(filepath: str) -> str:
    h = hashlib.new(HASH_ALGO)
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(STREAM_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def compute_exact_content_hash(filepath: str) -> Optional[str]:
    if not os.path.exists(filepath):
        logger.warning(f"File not found, skipping: {filepath}")
        return None
    try:
        ext = os.path.splitext(filepath)[1].lower()

        # PDFs: extract text → normalize → hash
        if ext == ".pdf":
            try:
                doc = fitz.open(filepath)
                text_parts: List[str] = []
                for page in doc:
                    text_parts.append(page.get_text() or "")
                doc.close()
                normalized = normalize_text_content("".join(text_parts))
                data = normalized.encode("utf-8")
                return _hash_bytes(data) if data else "empty_file_hash"
            except Exception as e:
                logger.warning(f"PDF parse failed ({filepath}): {e}")
            return None
            
        # Plain text files: hash raw bytes (exact content)
        if ext in TEXT_EXTENSIONS:
            with open(filepath, "rb") as f:
                data = f.read()
            return _hash_bytes(data) if data else "empty_file_hash"

        # Everything else: stream + hash (exact bytes)
        return _hash_file_stream(filepath)

    except Exception as e:
        logger.error(f"Hash error for {filepath}: {e}")
        return None

def get_db_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True) if os.path.dirname(DB_PATH) else None
    return sqlite3.connect(DB_PATH, check_same_thread=False)

# --- FastAPI App ---

app = FastAPI(title="Clarifile Dedup Service")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "dedup"}

@app.get("/duplicates")
def find_duplicates():
    """
    Find duplicate files by exact content:
    - Group by SHA-256 of file content
    - Skip missing files
    - Return only groups with >= 2 unique file paths
    """
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            # Adjust column names/table if yours differ
            cur.execute("SELECT id, file_path, file_name FROM files WHERE file_path IS NOT NULL")
        rows = cur.fetchall()

        if not rows:
            return {"summary": {"duplicate_groups_found": 0, "total_files_processed": 0}, "duplicates": []}

        hash_to_files: Dict[str, List[Dict[str, Any]]] = {}
        total_existing = 0

        for file_id, file_path, file_name in rows:
            if not file_path:
                continue
            # Normalize path to avoid duplicates from different casing or representations
            norm_path = os.path.normcase(os.path.abspath(file_path))
            if not os.path.exists(norm_path):
                continue
            total_existing += 1

            file_hash = compute_exact_content_hash(norm_path)
            if not file_hash:
                continue

            lst = hash_to_files.setdefault(file_hash, [])
            # Avoid duplicates of the same normalized file path within the same hash group
            if not any(f["path"] == norm_path for f in lst):
                lst.append({
                    "id": file_id,
                    "name": (file_name or os.path.basename(norm_path)),
                    "path": norm_path
                })

        duplicates = []
        group_num = 1
        for content_hash, files in hash_to_files.items():
            if len(files) >= 2:
                # Sort deterministically if you want stable ordering
                files_sorted = sorted(files, key=lambda f: (f["name"].lower(), f["path"].lower()))
                duplicates.append({
                    "group_id": f"group_{group_num}",
                    "file_count": len(files_sorted),
                    "files": [{"id": f["id"], "name": f["name"], "path": f["path"]} for f in files_sorted]
                })
                group_num += 1

        return {
            "summary": {
                "duplicate_groups_found": len(duplicates),
                "total_files_processed": total_existing
            },
            "duplicates": duplicates
        }

    except Exception as e:
        logger.exception("Error finding duplicates")
        return {
            "error": str(e),
            "summary": {"duplicate_groups_found": 0, "total_files_processed": 0},
            "duplicates": []
        }

@app.post("/resolve_duplicate")
def resolve_duplicate(payload: Dict[str, Any]):
    """
    Resolve a duplicate pair by keeping one and deleting the other from disk.
    Expected payload:
      {
        "file_a": <id>,
        "file_b": <id>,
        "action": "keep_a" | "keep_b"
      }
    """
    file_a = payload.get("file_a")
    file_b = payload.get("file_b")
    action = payload.get("action")
    
    if action not in ("keep_a", "keep_b") or not file_a or not file_b:
        return {"error": "Invalid payload. Provide file_a, file_b and action in {keep_a|keep_b}."}

    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, file_path, file_name FROM files WHERE id IN (?, ?)", (file_a, file_b))
            rows = cur.fetchall()
            if len(rows) != 2:
                return {"error": "One or both files not found"}

            # Map by id for easy access
            files_map = {r[0]: {"id": r[0], "path": r[1], "name": r[2]} for r in rows}
            a = files_map.get(file_a)
            b = files_map.get(file_b)

            # Normalize paths
            a_path = os.path.normcase(os.path.abspath(a["path"])) if a and a["path"] else None
            b_path = os.path.normcase(os.path.abspath(b["path"])) if b and b["path"] else None
        
        if action == "keep_a":
                # Delete B from disk if exists, and mark DB status
                if b_path and os.path.exists(b_path):
                    try:
                        os.remove(b_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete duplicate file B: {b_path}: {e}")
                # Optionally mark in DB; we keep A, mark B as removed
                cur.execute("UPDATE files SET status=? WHERE id=?", ("removed_duplicate", file_b))
                cur.execute("UPDATE files SET status=? WHERE id=?", ("approved", file_a))
                conn.commit()
                return {"status": "resolved", "kept": a, "deleted": b_path}

        if action == "keep_b":
                if a_path and os.path.exists(a_path):
                    try:
                        os.remove(a_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete duplicate file A: {a_path}: {e}")
                cur.execute("UPDATE files SET status=? WHERE id=?", ("removed_duplicate", file_a))
                cur.execute("UPDATE files SET status=? WHERE id=?", ("approved", file_b))
        conn.commit()
        return {"status": "resolved", "kept": b, "deleted": a_path}
        
        return {"error": "Unsupported action"}
    except Exception as e:
        logger.exception("Error resolving duplicate")
        return {"error": str(e)}