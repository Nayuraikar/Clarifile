# services/dedup/app.py
from fastapi import FastAPI, HTTPException
import sqlite3, os, glob, numpy as np, shutil, hashlib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB = "metadata_db/clarifile.db"
EMB_DIR = "storage/embeddings"
ORGANIZED_DIR = "storage/organized_demo"
THRESHOLD = 0.92  # cosine similarity threshold for near-duplicates

app = FastAPI()

def get_conn():
    return sqlite3.connect(DB, check_same_thread=False)

def compute_file_hash(filepath):
    """Compute SHA-256 hash of file content"""
    try:
        if not os.path.exists(filepath):
            logger.warning(f"File does not exist: {filepath}")
            return None
            
        with open(filepath, "rb") as f:
            content = f.read()
            if not content:  # Handle empty files
                logger.info(f"Empty file: {filepath}")
                return "empty_file_hash"
            return hashlib.sha256(content).hexdigest()
    except (IOError, OSError) as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return None

import re

def normalize_text_content(content):
    """Normalize text content for comparison (remove whitespace, NULLs, control characters)"""
    if isinstance(content, bytes):
        try:
            content = content.decode('utf-8', errors='ignore')
        except:
            return content  # fallback
    
    # Remove NULL bytes and other control characters
    content = re.sub(r'[\x00-\x1F\x7F]', '', content)
    
    # Normalize whitespace and line endings
    normalized = re.sub(r'\s+', ' ', content.strip())
    return normalized


def compute_normalized_text_hash(filepath):
    """Compute hash of normalized text content (ignores whitespace differences)"""
    try:
        if not os.path.exists(filepath):
            return None
            
        with open(filepath, "rb") as f:
            content = f.read()
            if not content:
                return "empty_file_hash"
            
            # Try to normalize if it's text
            try:
                text_content = content.decode('utf-8', errors='ignore')
                normalized = normalize_text_content(text_content)
                return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
            except:
                # If normalization fails, use original content
                return hashlib.sha256(content).hexdigest()
                
    except (IOError, OSError) as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return None

def find_connected_components(nodes, edges):
    """
    Find connected components in an undirected graph.
    nodes: list of node identifiers
    edges: list of (node1, node2) tuples representing edges
    Returns: list of components, where each component is a set of nodes
    """
    # Build adjacency list
    graph = defaultdict(set)
    for node in nodes:
        graph[node] = set()
    for node1, node2 in edges:
        graph[node1].add(node2)
        graph[node2].add(node1)
    
    visited = set()
    components = []
    
    def dfs(node, component):
        if node in visited:
            return
        visited.add(node)
        component.add(node)
        for neighbor in graph[node]:
            dfs(neighbor, component)
    
    for node in nodes:
        if node not in visited:
            component = set()
            dfs(node, component)
            if len(component) > 0:
                components.append(component)
    
    return components

@app.get("/duplicates")
def duplicates():
    """Find duplicate files based on normalized text content."""
    conn = get_conn()
    cur = conn.cursor()

    try:
        cur.execute("SELECT id, file_path, file_name, proposed_label FROM files")
        rows = cur.fetchall()

        if not rows:
            logger.info("No files found in database")
            return {"duplicates": [], "message": "No files found in database"}

        files = {r[0]: {"id": r[0], "path": r[1], "name": r[2], "label": r[3]} for r in rows}
        logger.info(f"Processing {len(files)} files for duplicate detection")

        # Store files by normalized text hash
        by_hash = defaultdict(list)
        processed_files = 0
        hash_errors = 0

        for f in files.values():
            normalized_hash = compute_normalized_text_hash(f["path"])
            if normalized_hash:
                by_hash[normalized_hash].append(f)
                processed_files += 1
            else:
                hash_errors += 1
                logger.warning(f"Could not compute normalized hash for file: {f['path']}")

        # Collect duplicates
        duplicate_groups = []
        gid = 1
        for h, group in by_hash.items():
            if len(group) > 1:
                logger.info(f"Found duplicate group with {len(group)} files: {[f['name'] for f in group]}")
                duplicate_groups.append({
                    "group_id": gid,
                    "hash": h,
                    "files": group,
                    "file_count": len(group)
                })
                gid += 1

        logger.info(f"Found {len(duplicate_groups)} duplicate groups total")

        return {
            "duplicates": duplicate_groups,
            "summary": {
                "total_files_processed": processed_files,
                "hash_errors": hash_errors,
                "duplicate_groups_found": len(duplicate_groups),
                "total_duplicate_files": sum(group["file_count"] for group in duplicate_groups)
            }
        }

    except Exception as e:
        logger.error(f"Error in duplicate detection: {e}")
        raise HTTPException(status_code=500, detail=f"Duplicate detection error: {str(e)}")
    finally:
        conn.close()


@app.get("/duplicates/debug")
def duplicates_debug():
    """Debug endpoint to show file hash information"""
    conn = get_conn()
    cur = conn.cursor()
    
    try:
        cur.execute("SELECT id, file_path, file_name FROM files LIMIT 10")
        rows = cur.fetchall()
        
        debug_info = []
        for r in rows:
            file_id, file_path, file_name = r
            content_hash = compute_file_hash(file_path)
            normalized_hash = compute_normalized_text_hash(file_path)
            
            file_exists = os.path.exists(file_path)
            file_size = os.path.getsize(file_path) if file_exists else 0
            
            debug_info.append({
                "id": file_id,
                "name": file_name,
                "path": file_path,
                "exists": file_exists,
                "size": file_size,
                "content_hash": content_hash,
                "normalized_hash": normalized_hash
            })
        
        return {"debug_info": debug_info}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")
    finally:
        conn.close()

@app.post("/resolve_duplicate")
def resolve(payload: dict):
    """
    payload: { "file_a": <id>, "file_b": <id>, "action": "keep_a"|"keep_b"|"keep_both" }
    This does non-destructive handling: copies the kept file(s) into organized_demo/<label>
    and marks final_label/status in DB.
    """
    file_a = payload.get("file_a")
    file_b = payload.get("file_b")
    action = payload.get("action")
    
    if not file_a or not file_b or not action:
        raise HTTPException(status_code=400, detail="file_a, file_b and action required")
    
    if action not in ["keep_a", "keep_b", "keep_both"]:
        raise HTTPException(status_code=400, detail="action must be 'keep_a', 'keep_b', or 'keep_both'")
    
    conn = get_conn()
    cur = conn.cursor()
    
    try:
        cur.execute("SELECT id,file_path,file_name,proposed_label FROM files WHERE id=?", (file_a,))
        a = cur.fetchone()
        cur.execute("SELECT id,file_path,file_name,proposed_label FROM files WHERE id=?", (file_b,))
        b = cur.fetchone()
        
        if not a or not b:
            raise HTTPException(status_code=404, detail="One or both files not found")
        
        # Check if files actually exist
        if not os.path.exists(a[1]):
            raise HTTPException(status_code=404, detail=f"File A not found at path: {a[1]}")
        if not os.path.exists(b[1]):
            raise HTTPException(status_code=404, detail=f"File B not found at path: {b[1]}")
        
        results = {"copied": [], "updated": []}
        
        if action == "keep_a":
            dest = os.path.join(ORGANIZED_DIR, a[3] or "Duplicates")
            os.makedirs(dest, exist_ok=True)
            dest_path = os.path.join(dest, a[2])
            
            # Keep A
            shutil.copy2(a[1], dest_path)
            cur.execute("UPDATE files SET final_label=?, status=? WHERE id=?", (a[3] or "Duplicates", "approved", a[0]))
            
            # Ignore B → delete the file from organized folder if it exists
            cur.execute("UPDATE files SET status=? WHERE id=?", ("ignored_duplicate", b[0]))
            if os.path.exists(b[1]):
                os.remove(b[1])
            
            results["copied"].append({"file_id": a[0], "dest_path": dest_path})
            results["updated"].extend([a[0], b[0]])
   
        elif action == "keep_b":
            dest = os.path.join(ORGANIZED_DIR, b[3] or "Duplicates")
            os.makedirs(dest, exist_ok=True)
            dest_path = os.path.join(dest, b[2])
            
            # Keep B
            shutil.copy2(b[1], dest_path)
            cur.execute("UPDATE files SET final_label=?, status=? WHERE id=?", (b[3] or "Duplicates", "approved", b[0]))
            
            # Ignore A → delete the file from organized folder if it exists
            cur.execute("UPDATE files SET status=? WHERE id=?", ("ignored_duplicate", a[0]))
            if os.path.exists(a[1]):
                os.remove(a[1])
            
            results["copied"].append({"file_id": b[0], "dest_path": dest_path})
            results["updated"].extend([a[0], b[0]])

            
        elif action == "keep_both":
            desta = os.path.join(ORGANIZED_DIR, a[3] or "Duplicates")
            destb = os.path.join(ORGANIZED_DIR, b[3] or "Duplicates")
            os.makedirs(desta, exist_ok=True)
            os.makedirs(destb, exist_ok=True)
            
            dest_path_a = os.path.join(desta, a[2])
            dest_path_b = os.path.join(destb, b[2])
            
            shutil.copy2(a[1], dest_path_a)
            shutil.copy2(b[1], dest_path_b)
            
            cur.execute("UPDATE files SET final_label=?, status=? WHERE id=?", (a[3] or "Duplicates", "approved", a[0]))
            cur.execute("UPDATE files SET final_label=?, status=? WHERE id=?", (b[3] or "Duplicates", "approved", b[0]))
            
            results["copied"].extend([
                {"file_id": a[0], "dest_path": dest_path_a},
                {"file_id": b[0], "dest_path": dest_path_b}
            ])
            results["updated"].extend([a[0], b[0]])
        
        conn.commit()
        logger.info(f"Resolved duplicate between files {file_a} and {file_b} with action '{action}'")
        
        return {"status": "resolved", "details": results}
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error resolving duplicate: {e}")
        raise HTTPException(status_code=500, detail=f"Error resolving duplicate: {str(e)}")
    finally:
        conn.close()

@app.get("/duplicates/debug_text")
def debug_text():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, file_name, file_path FROM files WHERE file_name IN ('invoice_1.txt','invoice_3.txt')")
    rows = cur.fetchall()
    result = []
    for r in rows:
        fid, fname, fpath = r
        if not os.path.exists(fpath):
            result.append({"file": fname, "exists": False})
            continue
        with open(fpath, "rb") as f:
            content = f.read()
        normalized = normalize_text_content(content)
        h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        result.append({
            "file": fname,
            "exists": True,
            "size": len(content),
            "normalized_text": normalized,
            "hash": h
        })
    return result
