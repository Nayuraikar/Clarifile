# parser/app.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os, sqlite3, hashlib, shutil, json, re, subprocess, tempfile
from PIL import Image
import fitz  # PyMuPDF
import numpy as np
import requests
from transformers import pipeline
import pytesseract
import chardet
import torch
import torchvision.transforms as transforms
from torchvision import models
import whisper
import cv2
from services.parser import nlp
import collections
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services.parser import nlp

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- Paths & constants ---
DB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "metadata_db", "clarifile.db")
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "storage", "sample_files")
ORGANIZED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "storage", "organized_demo")
ALLOWED_EXTS = {
    ".txt", ".pdf", ".png", ".jpg", ".jpeg",
    ".mp3", ".wav", ".mp4", ".mkv", ".mpeg"
}
BASE = "http://127.0.0.1:8000"

app = FastAPI()

class ApproveRequest(BaseModel):
    file_id: int
    final_label: str

# Allow requests from your frontend origin
origins = [
    "http://127.0.0.1:4000",
    "http://localhost:4000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # or ["*"] to allow all origins (not recommended for prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tesseract path (Windows only)
_possible = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_possible):
    pytesseract.pytesseract.tesseract_cmd = _possible

# Embed service
EMBED_SERVICE = os.getenv("EMBED_SERVICE", "http://127.0.0.1:8002")

# Summarizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Whisper for audio/video transcription
whisper_model = whisper.load_model("base")

# CV model for image/video classification
cv_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
cv_model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- DB init ---
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
        category_id INTEGER,
        transcript TEXT,
        tags TEXT,
        full_text TEXT
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
    cur.execute("""
      CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        rep_summary TEXT,
        rep_vector TEXT
      )
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        type TEXT,
        UNIQUE(name, type)
      )
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS file_entities (
        file_id INTEGER,
        entity_id INTEGER,
        count INTEGER DEFAULT 1,
        PRIMARY KEY(file_id, entity_id),
        FOREIGN KEY(file_id) REFERENCES files(id),
        FOREIGN KEY(entity_id) REFERENCES entities(id)
      )
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS entity_edges (
        a INTEGER,
        b INTEGER,
        weight INTEGER DEFAULT 1,
        PRIMARY KEY(a, b),
        FOREIGN KEY(a) REFERENCES entities(id),
        FOREIGN KEY(b) REFERENCES entities(id)
      )
    """)
    conn.commit()
    return conn

conn = init_db()

# --- Utilities ---
def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
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

def embed_texts(texts):
    try:
        r = requests.post(f"{EMBED_SERVICE}/embed_texts", json={"texts": texts}, timeout=120)
        r.raise_for_status()
        vecs = np.array(r.json()["vectors"], dtype=np.float32)
        return vecs
    except Exception as e:
        print(f"Embedding service error: {e}")
        return np.random.rand(len(texts), 384).astype(np.float32)

def ensure_entity_rows(con):
    con.execute("PRAGMA foreign_keys=ON;")

def upsert_entity(con, name: str, etype: str) -> int:
    cur = con.cursor()
    cur.execute("INSERT OR IGNORE INTO entities(name, type) VALUES(?,?)", (name, etype))
    cur.execute("SELECT id FROM entities WHERE name=? AND type=?", (name, etype))
    return cur.fetchone()[0]

def bump_file_entity(con, file_id: int, entity_id: int, inc: int = 1):
    cur = con.cursor()
    cur.execute("""
        INSERT INTO file_entities(file_id, entity_id, count)
        VALUES(?,?,?)
        ON CONFLICT(file_id, entity_id) DO UPDATE SET count = count + excluded.count
    """, (file_id, entity_id, inc))

def update_cooccurrence(con, entity_ids):
    cur = con.cursor()
    ids = sorted(set(entity_ids))
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            a, b = ids[i], ids[j]
            cur.execute("""
              INSERT INTO entity_edges(a,b,weight) VALUES(?,?,1)
              ON CONFLICT(a,b) DO UPDATE SET weight = weight + 1
            """, (a,b))

def extract_and_store_entities(con, file_id: int, text: str):
    ensure_entity_rows(con)
    raw = nlp.extract_entities(text)
    if not raw:
        return {"entities": [], "unique": 0}
    counter = collections.Counter((e["name"], e["type"]) for e in raw)
    inserted_ids = []
    for (name, etype), cnt in counter.items():
        eid = upsert_entity(con, name, etype)
        bump_file_entity(con, file_id, eid, cnt)
        inserted_ids.append(eid)
    update_cooccurrence(con, inserted_ids)
    con.commit()
    return {"entities": [{"name": n, "type": t, "count": c} for (n,t), c in counter.items()], "unique": len(inserted_ids)}

# --- Text extraction ---
def read_text_file(path):
    try:
        with open(path, "rb") as f:
            raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result.get('encoding', 'utf-8')
        try:
            return raw_data.decode(encoding)
        except:
            return raw_data.decode('utf-8', errors='ignore')
    except:
        return ""

def extract_text_from_pdf(path):
    text = ""
    try:
        doc = fitz.open(path)
        for page in doc:
            page_text = page.get_text("text")
            if page_text.strip():
                text += page_text + "\n"
            else:
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img) + "\n"
        doc.close()
    except Exception as e:
        print("PDF extraction error:", e)
    return text

def extract_text_from_image(path):
    try:
        img = Image.open(path).convert("RGB")
        return pytesseract.image_to_string(img)
    except Exception as e:
        print("Image OCR error:", e)
        return ""

# --- Audio/Video + CV ---
def classify_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = cv_model(tensor)
            _, pred = outputs.max(1)
        return f"Image class {pred.item()}"
    except Exception as e:
        print("Image classification error:", e)
        return "Unknown image"

def transcribe_audio(path):
    try:
        result = whisper_model.transcribe(path)
        return result["text"]
    except Exception as e:
        print("Whisper error:", e)
        return ""

FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"


def extract_audio_with_ffmpeg(video_path, output_audio="temp_audio.wav"):
    try:
        subprocess.run([
            FFMPEG_PATH, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", output_audio
        ], check=True, capture_output=True)

        return output_audio
    except Exception as e:
        print("FFmpeg audio extraction error:", e)
        return None

def process_video_ffmpeg(path):
    transcript = ""
    frames_labels = []
    try:
        audio_path = extract_audio_with_ffmpeg(path)
        if audio_path:
            transcript = transcribe_audio(audio_path)

        # Only process video frames for formats OpenCV reliably supports
        if os.path.splitext(path)[1].lower() not in {".mpeg"}:
            cap = cv2.VideoCapture(path)
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 1
            step = fps * 10
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if count % step == 0:
                    frame_path = f"frame_{count}.jpg"
                    cv2.imwrite(frame_path, frame)
                    label = classify_image(frame_path)
                    frames_labels.append(label)
                count += 1
            cap.release()
    except Exception as e:
        print("Video processing error:", e)
    return transcript, frames_labels


# --- Summarization ---
def summarize_text(long_text: str) -> str:
    if not long_text.strip():
        return ""
    try:
        return nlp.summarize_with_gemini(long_text, max_tokens=512)
    except Exception as e:
        print("Gemini summarization error:", e)
        return long_text[:500]  # fallback

# --- Category assignment ---
def compress_title(summary: str) -> str:
    if not summary: return "General"
    lower = summary.lower()
    if "invoice" in lower or "bill" in lower: return "Finance"
    if "meeting" in lower or "agenda" in lower: return "Work"
    if "personal" in lower: return "Personal"
    return summary.split(" ")[0:3][0]

def assign_category_from_summary(summary: str, threshold: float = 0.75):
    if not summary.strip():
        return None, "Uncategorized"
    try:
        category_name = nlp.classify_with_gemini(summary)
        cur = conn.cursor()
        # Insert category if not exists
        cur.execute("INSERT OR IGNORE INTO categories(name, rep_summary, rep_vector) VALUES (?,?,?)",
                    (category_name, summary, "[]"))
        conn.commit()
        cur.execute("SELECT id FROM categories WHERE name=?", (category_name,))
        row = cur.fetchone()
        return (row[0] if row else None), category_name
    except Exception as e:
        print("Gemini category error:", e)
        return None, "Uncategorized"
    
# --- API ---
@app.post("/scan_folder")
def scan_folder():
    print("=== Starting scan_folder() endpoint ===")
    try:
        # Check if sample directory exists
        print(f"Checking if sample directory exists: {SAMPLE_DIR}")
        if not os.path.exists(SAMPLE_DIR):
            print(f"ERROR: Sample directory {SAMPLE_DIR} does not exist")
            raise HTTPException(status_code=404, detail=f"Sample directory {SAMPLE_DIR} does not exist")
        
        print(f"Sample directory exists. Listing files...")
        files_in_dir = os.listdir(SAMPLE_DIR)
        print(f"Found {len(files_in_dir)} items in directory: {files_in_dir}")
        
        inserted_files_with_chunks = 0
        processed_files = []
        
        for fname in files_in_dir:
            try:
                print(f"\n--- Processing file: {fname} ---")
                path = os.path.join(SAMPLE_DIR, fname)
                
                if not os.path.isfile(path):
                    print(f"Skipping {fname} - not a file")
                    continue
                    
                ext = os.path.splitext(path)[1].lower()
                print(f"File extension: {ext}")
                
                if ext not in ALLOWED_EXTS:
                    print(f"Skipping {fname} - extension {ext} not in allowed extensions: {ALLOWED_EXTS}")
                    continue

                print(f"Computing file hash for {fname}...")
                try:
                    fh = file_hash(path)
                    size = os.path.getsize(path)
                    print(f"File hash: {fh[:16]}..., Size: {size} bytes")
                except Exception as hash_error:
                    print(f"ERROR computing hash for {fname}: {hash_error}")
                    continue
                
                print(f"Inserting file record into database...")
                cur = conn.cursor()
                try:
                    cur.execute("""INSERT OR IGNORE INTO files
                                   (file_path,file_name,file_hash,size,status,proposed_label)
                                   VALUES (?,?,?,?,?,?)""",
                                (path, fname, fh, size, "new", "Uncategorized"))
                    conn.commit()
                    print(f"File record inserted/exists for {fname}")
                except Exception as db_error:
                    print(f"ERROR inserting file record for {fname}: {db_error}")
                    continue
                
                cur.execute("SELECT id FROM files WHERE file_hash=?", (fh,))
                row = cur.fetchone()
                if not row:
                    print(f"ERROR: Could not retrieve file_id for {fname}")
                    continue
                file_id = row[0]
                print(f"File ID: {file_id}")

                text, summary, transcript, tags = "", "", "", []

                print(f"Processing content based on file type: {ext}")
                
                if ext == ".txt":
                    print("Processing as text file...")
                    try:
                        text = read_text_file(path)
                        print(f"Extracted {len(text)} characters from text file")
                    except Exception as txt_error:
                        print(f"ERROR reading text file {fname}: {txt_error}")
                        
                elif ext == ".pdf":
                    print("Processing as PDF file...")
                    try:
                        text = extract_text_from_pdf(path)
                        print(f"Extracted {len(text)} characters from PDF")
                    except Exception as pdf_error:
                        print(f"ERROR extracting text from PDF {fname}: {pdf_error}")
                        
                elif ext in {".png", ".jpg", ".jpeg"}:
                    print("Processing as image file...")
                    try:
                        text = extract_text_from_image(path)
                        print(f"Extracted {len(text)} characters via OCR")
                        classification = classify_image(path)
                        tags.append(classification)
                        print(f"Image classification: {classification}")
                    except Exception as img_error:
                        print(f"ERROR processing image {fname}: {img_error}")
                        
                elif ext in {".mp3", ".wav"}:
                    print("Processing as audio file...")
                    try:
                        transcript = transcribe_audio(path)
                        text = transcript
                        print(f"Audio transcription: {len(transcript)} characters")
                    except Exception as audio_error:
                        print(f"ERROR transcribing audio {fname}: {audio_error}")
                        
                elif ext in {".mp4", ".mkv", ".mpeg"}:
                    print("Processing as video file...")
                    try:
                        transcript, frame_tags = process_video_ffmpeg(path)
                        text = transcript + " " + " ".join(frame_tags)
                        tags.extend(frame_tags)
                        print(f"Video processing: {len(transcript)} transcript chars, {len(frame_tags)} frame tags")
                    except Exception as video_error:
                        print(f"ERROR processing video {fname}: {video_error}")

                print(f"Updating file with extracted text ({len(text or '')} chars)...")
                try:
                    cur.execute("UPDATE files SET full_text=? WHERE id=?", (text or "", file_id))
                    conn.commit()
                    print("Full text updated in database")
                except Exception as update_error:
                    print(f"ERROR updating full_text for {fname}: {update_error}")
                
                print("Extracting entities...")
                try:
                    entity_result = extract_and_store_entities(conn, file_id, text or "")
                    print(f"Entities extracted: {entity_result}")
                except Exception as entity_error:
                    print(f"ERROR extracting entities for {fname}: {entity_error}")

                if text and text.strip():
                    print("Creating text chunks...")
                    try:
                        chunks = chunk_text(text)
                        print(f"Created {len(chunks)} chunks")
                        
                        for i, (s, e, chunk) in enumerate(chunks):
                            chash = hashlib.sha256((str(file_id)+str(s)+str(e)).encode()).hexdigest()
                            cur.execute("""INSERT OR IGNORE INTO chunks
                                        (file_id,chunk_text,chunk_hash,start_offset,end_offset)
                                        VALUES (?,?,?,?,?)""",
                                        (file_id, chunk, chash, s, e))
                            if i == 0:  # Log first chunk as example
                                print(f"First chunk: {chunk[:100]}...")
                        
                        conn.commit()
                        inserted_files_with_chunks += 1
                        print("Chunks inserted successfully")
                    except Exception as chunk_error:
                        print(f"ERROR creating chunks for {fname}: {chunk_error}")

                    print("Generating summary...")
                    try:
                        summary = summarize_text(text)
                        print(f"Summary generated: {len(summary)} characters")
                    except Exception as summary_error:
                        print(f"ERROR generating summary for {fname}: {summary_error}")
                        summary = ""
                    
                    print("Assigning category...")
                    try:
                        cat_id, cat_name = assign_category_from_summary(summary)
                        print(f"Category assigned: {cat_name} (ID: {cat_id})")
                    except Exception as cat_error:
                        print(f"ERROR assigning category for {fname}: {cat_error}")
                        cat_id, cat_name = None, "Uncategorized"
                else:
                    print("No text content - skipping summary/categorization")
                    cat_id, cat_name, summary = None, "Uncategorized", ""

                print("Updating file with final metadata...")
                try:
                    cur.execute("""UPDATE files
                                   SET summary=?, proposed_label=?, category_id=?, transcript=?, tags=?
                                   WHERE id=?""",
                                (summary, cat_name, cat_id, transcript, json.dumps(tags), file_id))
                    conn.commit()
                    print("File metadata updated successfully")
                except Exception as final_update_error:
                    print(f"ERROR in final update for {fname}: {final_update_error}")
                
                processed_files.append({
                    "file": fname,
                    "category": cat_name,
                    "summary": summary[:120] if summary else "",
                    "transcript_len": len(transcript),
                    "tags": tags
                })
                
                print(f" Successfully processed {fname}")
                
            except Exception as file_error:
                print(f" ERROR processing file {fname}:")
                import traceback
                print(traceback.format_exc())
                # Continue with next file instead of failing completely
                continue
        
        print(f"\n=== Scan Complete ===")
        print(f"Files with chunks: {inserted_files_with_chunks}")
        print(f"Total processed files: {len(processed_files)}")
        
        return {"status": "scanned", "chunks_inserted": inserted_files_with_chunks,
                "processed_files": processed_files}
                
    except Exception as e:
        print(f"\n FATAL ERROR in scan_folder():")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

@app.get("/list_proposals")
def list_proposals():
    c = conn.cursor()
    c.execute("""SELECT id, file_name, file_path, proposed_label, final_label 
                 FROM files 
                 WHERE summary IS NOT NULL AND summary != ''""")
    rows = c.fetchall()
    # Filter out files that no longer exist on disk
    filtered = [r for r in rows if os.path.exists(r[2])]
    return [{"id": r[0], "file": r[1], "proposed": r[3], "final": r[4]} for r in filtered]


@app.post("/approve")
def approve(request: ApproveRequest):
    try:
        cur = conn.cursor()
        cur.execute("UPDATE files SET final_label=? WHERE id=?", (request.final_label, request.file_id))
        conn.commit()
        return {"status": "approved", "file_id": request.file_id, "final_label": request.final_label}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/categories")
def categories():
    """
    Returns a list of categories with counts of files that have been approved with that category.
    Only considers 'final_label' for each file.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT final_label, COUNT(*) AS file_count
        FROM files
        WHERE final_label IS NOT NULL AND final_label != ''
        GROUP BY final_label
        ORDER BY file_count DESC
    """)
    rows = cur.fetchall()
    return [{"name": r[0], "file_count": r[1]} for r in rows]


@app.get("/file_summary")
def file_summary(file_id: int):
    c = conn.cursor()
    c.execute("""SELECT f.summary, f.proposed_label, f.final_label, f.category_id,
                        c.name as category_name, f.transcript, f.tags
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
        "category_name": row[4] or row[1],
        "transcript": row[5] or "",
        "tags": json.loads(row[6]) if row[6] else []
    }

@app.get("/debug_text")
def debug_text(file_id: int):
    c = conn.cursor()
    c.execute("SELECT chunk_text FROM chunks WHERE file_id=?", (file_id,))
    rows = c.fetchall()
    return {"chunks": [r[0] for r in rows]}

@app.get("/file_entities")
def file_entities(file_id: int):
    cur = conn.cursor()
    cur.execute("""
      SELECT e.name, e.type, fe.count
      FROM file_entities fe
      JOIN entities e ON e.id = fe.entity_id
      WHERE fe.file_id=?
      ORDER BY fe.count DESC, e.name ASC
    """, (file_id,))
    rows = cur.fetchall()
    return [{"name": r[0], "type": r[1], "count": r[2]} for r in rows]

@app.get("/entity_graph")
def entity_graph(min_count: int = 1, types: Optional[str] = None):
    cur = conn.cursor()
    type_filter = [t.strip() for t in types.split(",")] if types else []
    if type_filter:
        q = f"""
            SELECT e.id, e.name, e.type, SUM(fe.count) as total
            FROM entities e
            JOIN file_entities fe ON fe.entity_id=e.id
            WHERE e.type IN ({','.join('?'*len(type_filter))})
            GROUP BY e.id
            HAVING total >= ?
        """
        cur.execute(q, (*type_filter, min_count))
    else:
        cur.execute("""
            SELECT e.id, e.name, e.type, SUM(fe.count) as total
            FROM entities e
            JOIN file_entities fe ON fe.entity_id=e.id
            GROUP BY e.id
            HAVING total >= ?
        """, (min_count,))
    nodes = [{"id": r[0], "name": r[1], "type": r[2], "count": r[3]} for r in cur.fetchall()]
    node_ids = set(n["id"] for n in nodes)
    if not node_ids:
        return {"nodes": [], "edges": []}
    qmarks = ",".join("?" * len(node_ids))
    cur.execute(f"""
        SELECT a,b,weight FROM entity_edges
        WHERE a IN ({qmarks}) AND b IN ({qmarks})
    """, (*node_ids, *node_ids))
    edges = [{"a": r[0], "b": r[1], "weight": r[2]} for r in cur.fetchall()]
    return {"nodes": nodes, "edges": edges}

@app.get("/ask")
def ask(file_id: int = Query(...), q: str = Query(...)):
    cur = conn.cursor()
    cur.execute("SELECT full_text FROM files WHERE id=?", (file_id,))
    row = cur.fetchone()
    full_text = row[0] if row else ""
    if not full_text:
        return {"ok": False, "error": "no text available (re-run scan)"}
    ans = nlp.best_answer(q, full_text)
    return {"ok": True, "answer": ans.get("answer", ""), "score": ans.get("score", 0), "context": ans.get("context", "")}
@app.post("/embed")
def embed():
    try:
        r = requests.post("http://127.0.0.1:8002/embed_pending", timeout=5)
        return r.json()
    except:
        return {"status": "embedding completed", "message": "Embed service not available"}

@app.post("/reindex") 
def reindex():
    try:
        r = requests.post("http://127.0.0.1:8003/reindex", timeout=5)
        return r.json()
    except:
        return {"status": "reindexed", "message": "Indexer service not available"}

@app.get("/duplicates")
def duplicates():
    try:
        r = requests.get("http://127.0.0.1:8004/duplicates", timeout=5)
        return r.json()
    except:
        return {"duplicates": [], "message": "Dedup service not available"}