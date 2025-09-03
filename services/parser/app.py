# services/parser/app.py
from fastapi import FastAPI, HTTPException
import os, sqlite3, hashlib, shutil, json, re
from PIL import Image
import fitz  # PyMuPDF
import numpy as np
import requests
from transformers import pipeline
import pytesseract
import chardet  # ADD THIS: pip install chardet

# --- Paths & constants ---
DB = "metadata_db/clarifile.db"
SAMPLE_DIR = "storage/sample_files"
ORGANIZED_DIR = "storage/organized_demo"
ALLOWED_EXTS = {".txt", ".pdf", ".png", ".jpg", ".jpeg"}

app = FastAPI()

# If Tesseract is not on PATH, set this to your Tesseract install path
_possible = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_possible):
    pytesseract.pytesseract.tesseract_cmd = _possible

# Where the embed service runs
EMBED_SERVICE = os.getenv("EMBED_SERVICE", "http://127.0.0.1:8002")

# Lightweight summarizer (fast on CPU)
# pip install transformers torch (CPU)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# --- DB init & migrations ---
def init_db():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT UNIQUE,
        file_name TEXT,
        file_hash TEXT,
        size INTEGER,
        status TEXT,
        proposed_label TEXT,
        final_label TEXT,
        inserted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        summary TEXT,             
        category_id INTEGER       
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER,
        chunk_text TEXT,
        chunk_hash TEXT,
        start_offset INTEGER,
        end_offset INTEGER,
        FOREIGN KEY(file_id) REFERENCES files(id)
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS labels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER,
        label TEXT,
        source TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    # categories table: representative summary + vector for centroid
    cur.execute("""
      CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        rep_summary TEXT,
        rep_vector TEXT
      )
    """)
    conn.commit()
    return conn

conn = init_db()

# --- Enhanced Text Reading with Proper Encoding Detection ---
def read_text_file(path):
    """
    Properly read text files with automatic encoding detection.
    Fixes issues with UTF-16 BOM and other non-UTF8 encodings.
    """
    try:
        # First, read raw bytes to detect encoding
        with open(path, "rb") as f:
            raw_data = f.read()
        
        # Detect encoding
        result = chardet.detect(raw_data)
        encoding = result.get('encoding', 'utf-8')
        confidence = result.get('confidence', 0.0)
        
        # If confidence is very low, try common encodings
        if confidence < 0.7:
            for fallback_encoding in ['utf-8', 'utf-16', 'windows-1252', 'iso-8859-1']:
                try:
                    text = raw_data.decode(fallback_encoding)
                    # Check if decoded text looks reasonable (no excessive control chars)
                    if len([c for c in text[:1000] if ord(c) < 32 and c not in '\n\r\t']) < 50:
                        return text
                except (UnicodeDecodeError, UnicodeError):
                    continue
        
        # Use detected encoding
        try:
            return raw_data.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            # Final fallback with error handling
            return raw_data.decode('utf-8', errors='ignore')
            
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""

# --- Enhanced Text Summarization ---
def summarize_text(long_text: str) -> str:
    """
    Improved summarization that handles all text content, not just first 3000 chars.
    Processes text in chunks and combines summaries for comprehensive coverage.
    """
    if not long_text or not long_text.strip():
        return ""
    
    # Clean text: remove excessive whitespace and control characters
    text = re.sub(r'\s+', ' ', long_text.strip())
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    if len(text) <= 1000:
        # Short text - summarize directly
        try:
            out = summarizer(text, max_length=80, min_length=20, do_sample=False)
            return (out[0].get("summary_text") or "").strip()
        except Exception:
            return ""
    
    # Long text - chunk and summarize each part
    chunks = chunk_text(text, max_chars=1000, overlap=200)
    summaries = []
    
    for _, _, chunk in chunks:
        if not chunk.strip():
            continue
        try:
            out = summarizer(chunk, max_length=120, min_length=40, do_sample=False)
            summary = (out[0].get("summary_text") or "").strip()
            if summary and len(summary) > 10:  # Only keep meaningful summaries
                summaries.append(summary)
        except Exception:
            continue
    
    if not summaries:
        return ""
    
    # If we have multiple chunk summaries, create a meta-summary
    combined = " ".join(summaries)
    if len(summaries) > 1 and len(combined) > 500:
        try:
            # Summarize the combined summaries for final coherent summary
            out = summarizer(combined, max_length=150, min_length=50, do_sample=False)
            return (out[0].get("summary_text") or "").strip()
        except Exception:
            pass
    
    return combined

# --- Improved Category Name Generation ---
def compress_title(summary: str) -> str:
    """
    Generate concise category names from summary.
    Looks for keywords first, then first 2-3 meaningful words.
    """
    if not summary or not summary.strip():
        return "General"
    
    # Lowercase summary for keyword search
    lower = summary.lower()
    if "invoice" in lower or "bill" in lower or "statement" in lower:
        return "Finance"
    if "meeting" in lower or "agenda" in lower or "notes" in lower:
        return "Meeting"
    if "personal" in lower or "note" in lower or "grocery" in lower:
        return "Personal"
    if "registration" in lower or "form" in lower:
        return "Form"

    # Fallback: pick first 2-3 meaningful words
    text = re.sub(r'\s+', ' ', summary.strip())
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]{2,}\b', text)
    selected = words[:3]
    if not selected:
        return "General"
    return " ".join(word.capitalize() for word in selected)

# --- Utilities ---
def file_hash(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for ch in iter(lambda: f.read(8192), b''):
            h.update(ch)
    return h.hexdigest()

def chunk_text(text, max_chars=1000, overlap=200):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + max_chars)
        chunk = text[start:end]
        chunks.append((start, end, chunk))
        if end == L:
            break
        start = end - overlap
    return chunks

def extract_text_from_pdf(path):
    text = ""
    try:
        doc = fitz.open(path)
        for page in doc:
            page_text = page.get_text("text")
            if page_text and page_text.strip():
                text += page_text + "\n"
            else:
                # fallback to image OCR for scanned page
                try:
                    pix = page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        text += ocr_text + "\n"
                except Exception:
                    continue
        doc.close()
    except Exception as e:
        print(f"Error extracting PDF {path}: {e}")
    return text

def extract_text_from_image(path):
    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception as e:
        print(f"Error extracting image {path}: {e}")
        return ""

def propose_label_from_filename(fname):
    lower = fname.lower()
    if "invoice" in lower or "bill" in lower or "statement" in lower:
        return "Finance"
    if "meeting" in lower or "notes" in lower or "agenda" in lower:
        return "Work"
    if "personal" in lower or "grocery" in lower or "note" in lower:
        return "Personal"
    return "Uncategorized"

def embed_texts(texts):
    """Requires /embed_texts endpoint in embed service"""
    try:
        r = requests.post(f"{EMBED_SERVICE}/embed_texts", json={"texts": texts}, timeout=120)
        r.raise_for_status()
        vecs = np.array(r.json()["vectors"], dtype=np.float32)
        return vecs
    except Exception as e:
        print(f"Embedding service error: {e}")
        # Return dummy embeddings if service is down
        return np.random.rand(len(texts), 384).astype(np.float32)

def assign_category_from_summary(summary: str, threshold: float = 0.75):
    """
    Return (category_id, category_name).
    Reuse nearest existing category if similarity >= threshold; else create a new one.
    """
    if not summary or not summary.strip():
        return None, "Uncategorized"

    try:
        vec = embed_texts([summary])[0]  # L2-normalized
        c = conn.cursor()
        c.execute("SELECT id, name, rep_vector FROM categories")
        rows = c.fetchall()

        best_sim, best_row = -1.0, None
        for cid, name, rep_vec_json in rows:
            try:
                rep_vec = np.array(json.loads(rep_vec_json), dtype=np.float32)
                sim = float(np.dot(vec, rep_vec))  # cosine (both L2-normalized)
                if sim > best_sim:
                    best_sim, best_row = sim, (cid, name, rep_vec_json)
            except Exception:
                continue

        if best_row and best_sim >= threshold:
            return best_row[0], best_row[1]

        # Create a new category with this summary as representative
        name = compress_title(summary)
        c.execute(
            "INSERT INTO categories(name, rep_summary, rep_vector) VALUES (?,?,?)",
            (name, summary, json.dumps(vec.tolist()))
        )
        conn.commit()
        return c.lastrowid, name
        
    except Exception as e:
        print(f"Category assignment error: {e}")
        return None, compress_title(summary)

# --- API ---

@app.post("/scan_folder")
def scan_folder():
    """Enhanced scan with proper text extraction and AI summarization"""
    inserted_files_with_chunks = 0
    processed_files = []
    
    for fname in os.listdir(SAMPLE_DIR):
        path = os.path.join(SAMPLE_DIR, fname)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        print("Scanning:", path, "ext:", ext)
        if ext not in ALLOWED_EXTS:
            continue

        try:
            fh = file_hash(path)
            size = os.path.getsize(path)
            cur = conn.cursor()

            # Insert file skeletal row if new
            cur.execute("""INSERT OR IGNORE INTO files
                           (file_path,file_name,file_hash,size,status,proposed_label)
                           VALUES (?,?,?,?,?,?)""",
                        (path, fname, fh, size, "new", propose_label_from_filename(fname)))
            conn.commit()

            cur.execute("SELECT id FROM files WHERE file_hash=?", (fh,))
            row = cur.fetchone()
            if not row:
                continue
            file_id = row[0]

            # Extract text with proper encoding handling
            text = ""
            if ext == ".txt":
                text = read_text_file(path)  # FIXED: proper encoding detection
            elif ext == ".pdf":
                text = extract_text_from_pdf(path)
            elif ext in {".png", ".jpg", ".jpeg"}:
                text = extract_text_from_image(path)

            # Debug: log first 200 chars of extracted text
            print(f"File: {fname}, Text preview: {repr(text[:200])}")

            # Insert chunks if we have text
            if text and text.strip():
                chunks = chunk_text(text)

                if not chunks:
                    print(f"No chunks generated for {fname}, skipping chunk insert.")
                else:
                    for s, e, chunk in chunks:
                        chash = hashlib.sha256((str(file_id)+str(s)+str(e)).encode()).hexdigest()
                        cur.execute("""INSERT OR IGNORE INTO chunks
                                    (file_id,chunk_text,chunk_hash,start_offset,end_offset)
                                    VALUES (?,?,?,?,?)""",
                                    (file_id, chunk, chash, s, e))
                    conn.commit()
                    inserted_files_with_chunks += 1


            # Enhanced summarization and category assignment
            summary = summarize_text(text) if text and text.strip() else ""
            
            if summary and summary.strip():
                cat_id, cat_name = assign_category_from_summary(summary)
                proposed = cat_name
                print(f"Generated summary for {fname}: {summary[:100]}...")
                print(f"Assigned category: {cat_name}")
            else:
                # fallback to filename heuristic if no text/summary
                cat_id = None
                cat_name = propose_label_from_filename(fname)
                proposed = cat_name
                print(f"No summary for {fname}, using filename-based category: {cat_name}")

            cur.execute("""UPDATE files
                           SET summary = ?, proposed_label = ?, category_id = ?
                           WHERE id = ?""",
                        (summary, proposed, cat_id, file_id))
            conn.commit()
            
            processed_files.append({
                "file_name": fname,
                "summary_length": len(summary) if summary else 0,
                "category": cat_name,
                "text_length": len(text) if text else 0
            })
            
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue

    return {
        "status": "scanned", 
        "chunks_inserted": inserted_files_with_chunks,
        "processed_files": processed_files
    }

@app.get("/list_proposals")
def list_proposals():
    cur = conn.cursor()
    cur.execute("""SELECT f.id, f.file_name, f.proposed_label, f.final_label, f.status, 
                          f.summary, c.name as category_name
                   FROM files f
                   LEFT JOIN categories c ON f.category_id = c.id
                   ORDER BY f.inserted_at DESC""")
    rows = cur.fetchall()
    results = []
    for r in rows:
        # Show if summary looks like encoding garbage
        summary = r[5] or ""
        has_encoding_issues = summary.startswith('ÿþ') or 'ÿþ' in summary[:20]
        
        results.append({
            "file_id": r[0],
            "file_name": r[1],
            "proposed_label": r[2],
            "final_label": r[3],
            "status": r[4],
            "summary": summary,
            "summary_preview": summary[:100] + "..." if len(summary) > 100 else summary,
            "category_name": r[6] or r[2],
            "has_encoding_issues": has_encoding_issues,  # Debug flag
            "summary_length": len(summary)
        })
    return results

@app.post("/approve")
def approve(payload: dict):
    file_id = payload.get("file_id")
    label = payload.get("label")
    if not file_id or not label:
        raise HTTPException(status_code=400, detail="file_id and label required")
    cur = conn.cursor()
    cur.execute("SELECT file_path, file_name FROM files WHERE id=?", (file_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="file not found")
    path, fname = row
    dest_dir = os.path.join(ORGANIZED_DIR, label)
    os.makedirs(dest_dir, exist_ok=True)
    try:
        shutil.copy2(path, os.path.join(dest_dir, fname))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    cur.execute("UPDATE files SET final_label=?, status=? WHERE id=?", (label, "approved", file_id))
    cur.execute("INSERT INTO labels (file_id,label,source) VALUES (?,?,?)", (file_id, label, "user"))
    conn.commit()
    return {"status":"approved", "file_id": file_id, "label": label}

@app.get("/list_chunks")
def list_chunks(file_id: int):
    cur = conn.cursor()
    cur.execute("SELECT id, chunk_text FROM chunks WHERE file_id=?", (file_id,))
    rows = cur.fetchall()
    return [{"id": r[0], "chunk_text": r[1]} for r in rows]

@app.get("/categories")
def list_categories():
    c = conn.cursor()
    c.execute("""
       SELECT c.name, c.rep_summary, COUNT(f.id) as file_count
       FROM categories c
       LEFT JOIN files f ON f.category_id = c.id
       GROUP BY c.name, c.rep_summary
       HAVING file_count > 0
       ORDER BY file_count DESC, c.name ASC
    """)
    rows = c.fetchall()
    return [{"name": r[0], "rep_summary": r[1], "file_count": r[2]} for r in rows]

@app.get("/file_summary")
def file_summary(file_id: int):
    c = conn.cursor()
    c.execute("""SELECT f.summary, f.proposed_label, f.final_label, f.category_id, c.name as category_name
                 FROM files f
                 LEFT JOIN categories c ON f.category_id = c.id
                 WHERE f.id=?""", (file_id,))
    row = c.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="file not found")
    return {
        "summary": row[0] or "",
        "proposed_label": row[1] or "",
        "final_label": row[2] or "",
        "category_id": row[3],
        "category_name": row[4] or row[1]  # fallback to proposed_label
    }
@app.on_event("startup")
def merge_duplicate_categories():
    c = conn.cursor()
    c.execute("SELECT name, GROUP_CONCAT(id) FROM categories GROUP BY name HAVING COUNT(*) > 1")
    for name, ids in c.fetchall():
        ids_list = [int(i) for i in ids.split(',')]
        main_id = ids_list[0]
        dup_ids = ids_list[1:]
        if not dup_ids:
            continue
        # Reassign files to main category
        c.execute(f"UPDATE files SET category_id=? WHERE category_id IN ({','.join(map(str, dup_ids))})", (main_id,))
        # Delete duplicates
        c.execute(f"DELETE FROM categories WHERE id IN ({','.join(map(str, dup_ids))})")
    conn.commit()

# New debugging endpoint
@app.get("/debug_text/{file_id}")
def debug_text_extraction(file_id: int):
    """Debug endpoint to see raw extracted text"""
    c = conn.cursor()
    c.execute("SELECT file_path, file_name FROM files WHERE id=?", (file_id,))
    row = c.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="file not found")
    
    path, fname = row
    ext = os.path.splitext(path)[1].lower()
    
    if ext == ".txt":
        text = read_text_file(path)
    elif ext == ".pdf":
        text = extract_text_from_pdf(path)
    elif ext in {".png", ".jpg", ".jpeg"}:
        text = extract_text_from_image(path)
    else:
        text = ""
    
    return {
        "file_name": fname,
        "text_length": len(text),
        "text_preview": text[:500],
        "text_end_preview": text[-200:] if len(text) > 500 else "",
        "contains_bom": text.startswith('\ufeff') if text else False
    }