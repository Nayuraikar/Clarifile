# services/parser/app.py
from fastapi import FastAPI, HTTPException
import os, sqlite3, hashlib, shutil
from typing import List
from PIL import Image
import fitz  # PyMuPDF
import pytesseract

DB = "metadata_db/clarifile.db"
SAMPLE_DIR = "storage/sample_files"
ORGANIZED_DIR = "storage/organized_demo"
ALLOWED_EXTS = {".txt", ".pdf", ".png", ".jpg", ".jpeg"}

app = FastAPI()

# If Tesseract is not on PATH, set this to your Tesseract install path, e.g.
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# The code tries a common path automatically:
_possible = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_possible):
    pytesseract.pytesseract.tesseract_cmd = _possible

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
        inserted_at DATETIME DEFAULT CURRENT_TIMESTAMP
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
    conn.commit()
    return conn

conn = init_db()

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
    doc = fitz.open(path)
    for page in doc:
        page_text = page.get_text("text")
        if page_text and page_text.strip():
            text += page_text + "\n"
        else:
            # fallback to image OCR for scanned page
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img) + "\n"
    return text

def extract_text_from_image(path):
    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception as e:
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

@app.post("/scan_folder")
def scan_folder():
    inserted = 0
    for fname in os.listdir(SAMPLE_DIR):
        path = os.path.join(SAMPLE_DIR, fname)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext not in ALLOWED_EXTS:
            continue
        fh = file_hash(path)
        size = os.path.getsize(path)
        cur = conn.cursor()
        cur.execute("INSERT OR IGNORE INTO files (file_path,file_name,file_hash,size,status,proposed_label) VALUES (?,?,?,?,?,?)",
                    (path, fname, fh, size, "new", propose_label_from_filename(fname)))
        conn.commit()
        cur.execute("SELECT id FROM files WHERE file_hash=?",(fh,))
        row = cur.fetchone()
        if not row:
            continue
        file_id = row[0]

        text = ""
        try:
            if ext == ".txt":
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            elif ext == ".pdf":
                text = extract_text_from_pdf(path)
            elif ext in {".png", ".jpg", ".jpeg"}:
                text = extract_text_from_image(path)
        except Exception as e:
            text = ""

        if text and text.strip():
            chunks = chunk_text(text)
            for s,e,t in chunks:
                chash = hashlib.sha256((str(file_id)+str(s)+str(e)).encode()).hexdigest()
                cur.execute("INSERT INTO chunks (file_id,chunk_text,chunk_hash,start_offset,end_offset) VALUES (?,?,?,?,?)",
                            (file_id, t, chash, s, e))
            conn.commit()
            inserted += 1
    return {"status":"scanned", "chunks_inserted": inserted}

@app.get("/list_proposals")
def list_proposals():
    cur = conn.cursor()
    cur.execute("SELECT id, file_name, proposed_label, final_label, status FROM files ORDER BY inserted_at DESC")
    rows = cur.fetchall()
    results = []
    for r in rows:
        results.append({"file_id": r[0], "file_name": r[1], "proposed_label": r[2], "final_label": r[3], "status": r[4]})
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
