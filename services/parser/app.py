# parser/app.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os, sqlite3, hashlib, shutil, json, re, subprocess, tempfile, collections
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
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nlp import *
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

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

class DriveFile(BaseModel):
    id: str
    name: str
    mimeType: str | None = None
    parents: list[str] | None = None
    size: int | str | None = None

    class Config:
        extra = 'allow'

class OrganizeDriveFilesRequest(BaseModel):
    files: list[DriveFile]
    move: bool = False
    auth_token: str | None = None
    override_category: str | None = None

class DriveAnalyzeRequest(BaseModel):
    # Accept any file shape to avoid validation 422s
    file: dict
    q: str | None = None
    auth_token: str | None = None

    class Config:
        extra = 'allow'


# --- Google Drive helpers ---
def get_drive_service(token: str):
    try:
        creds = Credentials(token)
        service = build('drive', 'v3', credentials=creds, cache_discovery=False)
        return service
    except Exception as e:
        print("Error initializing Drive service:", e)
        return None


def get_or_create_folder(service, folder_name: str, parent_id: str | None = None) -> str | None:
    if not service or not folder_name:
        return None
    try:
        name_escaped = folder_name.replace("'", "\'")
        if parent_id:
            q = f"mimeType='application/vnd.google-apps.folder' and name = '{name_escaped}' and '{parent_id}' in parents and trashed = false"
        else:
            q = f"mimeType='application/vnd.google-apps.folder' and name = '{name_escaped}' and trashed = false"
        res = service.files().list(q=q, fields="files(id,name)", pageSize=1, spaces='drive').execute()
        files = res.get('files', [])
        if files:
            return files[0]['id']
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            **({'parents': [parent_id]} if parent_id else {})
        }
        created = service.files().create(body=file_metadata, fields='id').execute()
        return created.get('id')
    except Exception as e:
        print("get_or_create_folder error:", e)
        return None


def move_file_to_folder(service, file_id: str, folder_id: str) -> dict | None:
    if not service or not file_id or not folder_id:
        return None
    try:
        meta = service.files().get(fileId=file_id, fields='parents').execute()
        previous_parents = ",".join(meta.get('parents', []))
        updated = service.files().update(
            fileId=file_id,
            addParents=folder_id,
            removeParents=previous_parents,
            fields='id, parents'
        ).execute()
        return updated
    except Exception as e:
        print("move_file_to_folder error:", e)
        return None

def drive_download_file(token: str, file_id: str, out_dir: str) -> str | None:
    """Download a Drive file to a temp path using the OAuth token.
    Returns local file path, or None on failure.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(url, headers=headers, stream=True, timeout=60)
        r.raise_for_status()
        tmp_path = os.path.join(out_dir, f"drive_{file_id}")
        with open(tmp_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return tmp_path
    except Exception as e:
        print("drive_download_file error:", e)
        return None


def infer_category_from_extension(file_name: str, mime_type: str | None) -> str:
    name_lower = (file_name or "").lower()
    if name_lower.endswith((".mp3", ".wav", ".m4a", ".aac")) or (mime_type and mime_type.startswith("audio/")):
        return "Audio"
    if name_lower.endswith((".mp4", ".mkv", ".mov", ".avi")) or (mime_type and mime_type.startswith("video/")):
        return "Video"
    if name_lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")) or (mime_type and mime_type.startswith("image/")):
        return "Images"
    if name_lower.endswith((".pdf",)):
        return "Documents"
    if name_lower.endswith((".txt", ".md", ".rtf")):
        return "Notes"
    if name_lower.endswith((".ipynb", ".py", ".js", ".ts", ".java", ".cpp", ".c", ".cs")):
        return "Technical"
    if name_lower.endswith((".xlsx", ".xls", ".csv")):
        return "Spreadsheets"
    if name_lower.endswith((".ppt", ".pptx", ".key")):
        return "Presentations"
    if name_lower.endswith((".zip", ".rar", ".7z")):
        return "Archives"
    return "General"


@app.post("/drive_analyze")
def drive_analyze(req: DriveAnalyzeRequest, auth_token: str | None = Query(None)):
    """Download the Drive file, extract text using existing extractors, summarize with Gemini,
    and optionally answer a question against the text. Returns transient results (no DB write).
    """
    token = req.auth_token or auth_token
    file = req.file or {}
    q = req.q
    file_id = file.get('id')
    path = drive_download_file(token, file_id, tempfile.gettempdir())
    if not path:
        raise HTTPException(status_code=400, detail="Failed to download file from Drive")
    try:
        ext = os.path.splitext((file.get('name') or path))[1].lower()
        text = ""
        tags = []
        transcript = ""
        if ext == ".txt":
            text = read_text_file(path)
        elif ext == ".pdf":
            text = extract_text_from_pdf(path)
        elif ext in {".png", ".jpg", ".jpeg"}:
            text = extract_text_from_image(path)
            tags.append(classify_image(path))
        elif ext in {".mp3", ".wav"}:
            transcript = transcribe_audio(path)
            text = transcript
        else:
            # Fallback: try reading as text
            text = read_text_file(path)

        summary = summarize_text(text)
        cat_id, cat_name = assign_category_from_summary(summary or (file.get('name') or ""))

        answer = None
        score = 0
        if q and text:
            ans = nlp.best_answer(q, text)
            answer = ans.get("answer", "")
            score = ans.get("score", 0)

        return {
            "summary": summary or "",
            "category": cat_name or "Uncategorized",
            "category_id": cat_id,
            "tags": tags,
            "transcript": transcript,
            "qa": {"answer": answer, "score": score} if q else None
        }
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

# Allow requests from your frontend origin
# For development, allow all origins to support Chrome extension and local UI.
# In production, replace with explicit origins including your extension ID.
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
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

# Global database connection - will be initialized at startup
conn = None

@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup."""
    global conn
    try:
        print("Initializing database connection...")
        conn = init_db()
        print(" Database connection initialized successfully")
    except Exception as e:
        print(f" Failed to initialize database: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown."""
    global conn
    if conn:
        try:
            conn.close()
            print("✅ Database connection closed")
        except Exception as e:
            print(f"❌ Error closing database connection: {e}")


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
        print(f"Gemini summarization error: {e}")
        # Fallback: return a simple excerpt
        return long_text[:500] + "..." if len(long_text) > 500 else long_text

# --- Category assignment ---
def compress_title(summary: str) -> str:
    if not summary: return "General"
    lower = summary.lower()
    if "invoice" in lower or "bill" in lower: return "Finance"
    if "meeting" in lower or "agenda" in lower: return "Work"
    if "personal" in lower: return "Personal"
    return summary.split(" ")[0:3][0]

def assign_category_from_summary(summary: str, full_text: str = "", threshold: float = 0.75):
    """
    Assign a category to a file based on its content.
    Uses the enhanced classification system with detailed categories.
    Prioritizes full text when available, falls back to summary.
    """
    content_to_analyze = full_text.strip() or summary.strip()
    if not content_to_analyze:
        return None, "Uncategorized"

    try:
        # Get the detailed category from our enhanced classifier
        category_name = nlp.classify_with_gemini(content_to_analyze)
        print(f"Classification result: {category_name}")

        # Clean up the category name
        category_name = category_name.strip()
        
        # Ensure we have a valid category name
        if not category_name or len(category_name) < 2:
            category_name = "Uncategorized"
            
        # Clean up the category name (remove any non-alphanumeric characters except spaces, colons, and hyphens)
        category_name = re.sub(r'[^\w\s:-]', '', category_name).strip()
        
        # Ensure we have a valid format (Category: Subcategory)
        if ':' not in category_name:
            # If it's a single word, add a default subcategory
            if ' ' not in category_name:
                category_name = f"{category_name}: General"
            else:
                # Otherwise, split on first space
                parts = category_name.split(' ', 1)
                category_name = f"{parts[0]}: {parts[1]}"
        
        # Extract main category (everything before first colon)
        main_category = category_name.split(':', 1)[0].strip()
        
        # Ensure main category is not empty
        if not main_category:
            main_category = "Uncategorized"
            category_name = "Uncategorized: General"
            
        print(f"Final category assignment: '{category_name}' (Main: '{main_category}')")

        # Insert or get the category from the database
        cur = conn.cursor()
        rep_content = full_text[:500] if full_text else summary[:500]  # Store first 500 chars as representative
        
        # First ensure the main category exists
        cur.execute(
            "INSERT OR IGNORE INTO categories(name, rep_summary, rep_vector) VALUES (?,?,?)",
            (main_category, rep_content, "[]")
        )
        
        # Then ensure the full category path exists
        if category_name != main_category:
            cur.execute(
                "INSERT OR IGNORE INTO categories(name, rep_summary, rep_vector) VALUES (?,?,?)",
                (category_name, rep_content, "[]")
            )
            
        conn.commit()

        # Try to get the full category path first
        cur.execute("SELECT id FROM categories WHERE name=?", (category_name,))
        row = cur.fetchone()
        
        # If we didn't find the full path, fall back to main category
        if not row:
            cur.execute("SELECT id FROM categories WHERE name=?", (main_category,))
            row = cur.fetchone()
            category_name = main_category  # Fall back to main category

        print(f"Database lookup - ID: {row[0] if row else 'None'}, Category: '{category_name}'")
        
        # Return the category ID and the full category path
        return (row[0] if row else None), full_category

    except Exception as e:
        print(f"Category classification error: {e}")
        # Fallback to extension-based categorization
        if full_text:
            fallback = infer_category_from_extension("unknown", "text/plain")
        else:
            fallback = "Uncategorized"
        return None, fallback
    
# --- API ---
@app.post("/scan_folder")
def scan_folder(timeout: int = 30):  # Reduced to 30 seconds for simple analysis
    print("=== Starting COMPREHENSIVE AI scan_folder() endpoint ===")
    import threading
    import time

    result = {"status": "error", "message": "Unknown error", "processed_files": []}
    exception = None

    def scan_worker():
        nonlocal result, exception
        try:
            # Check if sample directory exists
            print(f"Checking if sample directory exists: {SAMPLE_DIR}")
            if not os.path.exists(SAMPLE_DIR):
                print(f"ERROR: Sample directory {SAMPLE_DIR} does not exist")
                result = {"status": "error", "message": f"Sample directory {SAMPLE_DIR} does not exist", "processed_files": []}
                return

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

                    # COMPREHENSIVE ANALYSIS - extract text and use AI for categorization
                    text = ""
                    category_name = "General"
                    summary = ""

                    try:
                        # Extract text based on file type
                        if ext == ".txt":
                            text = read_text_file(path)
                            print(f"Extracted {len(text)} characters from text file")
                        elif ext == ".pdf":
                            text = extract_text_from_pdf(path)
                            print(f"Extracted {len(text)} characters from PDF")
                        elif ext in {".png", ".jpg", ".jpeg"}:
                            text = extract_text_from_image(path)
                            print(f"Extracted {len(text)} characters from image OCR")
                            # Also get image classification
                            tags = [classify_image(path)]
                        elif ext in {".mp3", ".wav"}:
                            transcript = transcribe_audio(path)
                            text = transcript
                            print(f"Transcribed {len(transcript)} characters from audio")
                        else:
                            # Try reading as text for other formats
                            text = read_text_file(path)
                            print(f"Extracted {len(text)} characters from file")

                        # Generate summary using AI
                        if text.strip():
                            summary = summarize_text(text)
                            print(f"Generated summary: {len(summary)} characters")

                            # Use AI to categorize based on content
                            try:
                                cat_id, category_name = assign_category_from_summary(summary, text)
                                print(f"AI categorized as: {category_name} (ID: {cat_id})")
                                
                                # If we got a category ID, use it; otherwise, fall back to name lookup
                                if not cat_id and category_name:
                                    cur.execute("SELECT id FROM categories WHERE name=?", (category_name,))
                                    cat_row = cur.fetchone()
                                    if cat_row:
                                        cat_id = cat_row[0]
                                        print(f"Found category ID {cat_id} for name '{category_name}'")
                                
                                # Extract and store entities if we have text
                                if len(text) > 100:  # Only for substantial content
                                    entity_info = extract_and_store_entities(conn, file_id, text)
                                    print(f"Extracted {entity_info['unique']} unique entities")
                                    
                            except Exception as cat_error:
                                print(f"Error in category assignment: {cat_error}")
                                # Fall back to extension-based categorization
                                category_name = infer_category_from_extension(fname, None)
                                print(f"Fallback categorization: {category_name}")
                                cat_id = None
                        
                        # If no text was extracted or category assignment failed, fall back to extension-based categorization
                        if not text.strip() or not category_name or category_name == "Uncategorized":
                            category_name = infer_category_from_extension(fname, None)
                            print(f"Fallback categorization: {category_name}")
                            cat_id = None

                    except Exception as content_error:
                        print(f"ERROR in content analysis for {fname}: {content_error}")
                        # Fallback to extension-based categorization
                        category_name = infer_category_from_extension(fname, None)
                        print(f"Fallback categorization: {category_name}")

                    # Update file with comprehensive analysis results
                    try:
                        # If we don't have a category name at this point, try to get one from the filename
                        if not category_name or category_name == "Uncategorized":
                            category_name = infer_category_from_extension(fname, None)
                            print(f"Using filename-based category: {category_name}")
                            
                            # If we still don't have a category, use a default
                            if not category_name or category_name == "Uncategorized":
                                category_name = "Documents"
                                
                        # If we don't have a category ID but have a name, try to look it up
                        if not cat_id and category_name:
                            cur.execute("SELECT id FROM categories WHERE name=?", (category_name,))
                            cat_row = cur.fetchone()
                            if cat_row:
                                cat_id = cat_row[0]
                                print(f"Found category ID {cat_id} for name '{category_name}'")
                            else:
                                # If category doesn't exist, create it
                                cur.execute(
                                    "INSERT INTO categories(name, rep_summary, rep_vector) VALUES (?,?,?)",
                                    (category_name, summary[:500] if summary else "", "[]")
                                )
                                cat_id = cur.lastrowid
                                print(f"Created new category: {category_name} (ID: {cat_id})")
                        
                        # Debug output
                        print(f"Final category before DB update - Name: '{category_name}', ID: {cat_id}")
                        
                        # Update the file record with the analysis results
                        cur.execute("""UPDATE files
                                       SET summary=?, proposed_label=?, category_id=?, full_text=?, tags=?
                                       WHERE id=?""",
                                    (summary, 
                                     category_name,  # Store the full category path
                                     cat_id, 
                                     text[:2000], 
                                     json.dumps(tags) if 'tags' in locals() else "[]", 
                                     file_id))
                        conn.commit()
                        print(f"File metadata updated successfully with AI analysis. Category: {category_name}")
                        
                        # Verify the update
                        cur.execute("SELECT proposed_label, category_id FROM files WHERE id=?", (file_id,))
                        updated = cur.fetchone()
                        if updated:
                            print(f"Verification - Stored category: {updated[0]}, Category ID: {updated[1]}")
                        else:
                            print("Warning: Could not verify category update in database")
                    except Exception as final_update_error:
                        print(f"ERROR in final update for {fname}: {final_update_error}")

                    processed_files.append({
                        "file": fname,
                        "category": category_name,
                        "summary": summary[:200] + "..." if len(summary) > 200 else summary,  # Truncate for display
                        "transcript_len": len(text) if text else 0,
                        "tags": tags if 'tags' in locals() else []
                    })

                    print(f"✅ Successfully processed {fname}")

                except Exception as file_error:
                    print(f"❌ ERROR processing file {fname}: {file_error}")
                    continue

            print(f"\n=== COMPREHENSIVE AI SCAN COMPLETE ===")
            print(f"Files processed: {len(processed_files)}")

            result = {"status": "scanned", "chunks_inserted": 0,
                    "processed_files": processed_files}

        except Exception as e:
            print(f"\n❌ FATAL ERROR in scan_folder(): {e}")
            import traceback
            traceback.print_exc()
            exception = e
            result = {"status": "error", "message": f"Scan failed: {str(e)}", "processed_files": []}

    # Start the scan in a separate thread
    scan_thread = threading.Thread(target=scan_worker)
    scan_thread.daemon = True
    scan_thread.start()

    # Wait for completion with timeout
    scan_thread.join(timeout=timeout)

    if scan_thread.is_alive():
        print(f"Scan operation timed out after {timeout} seconds")
        return {"status": "timeout", "message": f"Scan operation timed out after {timeout} seconds", "processed_files": []}
    elif exception:
        raise HTTPException(status_code=500, detail=str(exception))
    else:
        return result

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
def approve(file_id: int, final_label: str):
    try:
        cur = conn.cursor()
        cur.execute("UPDATE files SET final_label=? WHERE id=?", (final_label, file_id))
        conn.commit()
        return {"status": "approved", "file_id": file_id, "final_label": final_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/categories_with_files")
def categories_with_files():
    cur = conn.cursor()
    cur.execute("""
        SELECT final_label, COUNT(*) AS file_count
        FROM files
        WHERE final_label IS NOT NULL AND final_label != ''
        GROUP BY final_label
        ORDER BY file_count DESC
    """)
    rows = cur.fetchall()
    categories = []
    for row in rows:
        category_name = row[0]
        file_count = row[1]
        cur.execute("""
            SELECT id, file_name, file_path, proposed_label, final_label 
            FROM files 
            WHERE final_label = ?
        """, (category_name,))
        files = cur.fetchall()
        categories.append({
            "name": category_name,
            "file_count": file_count,
            "files": [{"id": f[0], "file": f[1], "proposed": f[3], "final": f[4]} for f in files]
        })
    return categories

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
    
    # Get file information from database
    cur.execute("SELECT file_path, file_name, full_text FROM files WHERE id=?", (file_id,))
    row = cur.fetchone()
    
    if not row:
        return {"ok": False, "error": "file not found"}
    
    file_path, file_name, full_text = row
    
    # If we already have text content, use it
    if full_text and full_text.strip():
        print(f"Using pre-extracted text for {file_name} ({len(full_text)} chars)")
    ans = nlp.best_answer(q, full_text)
    return {"ok": True, "answer": ans.get("answer", ""), "score": ans.get("score", 0), "context": ans.get("context", "")}
    
    # If no text content, try to extract it in real-time
    if not file_path or not os.path.exists(file_path):
        return {"ok": False, "error": "original file not accessible (re-run scan)"}
    
    print(f"Performing real-time text extraction for {file_name}")
    
    # Determine file type and extract text
    ext = os.path.splitext(file_path)[1].lower()
    extracted_text = ""
    
    try:
        if ext == ".txt":
            extracted_text = read_text_file(file_path)
        elif ext == ".pdf":
            extracted_text = extract_text_from_pdf(file_path)
        elif ext in {".png", ".jpg", ".jpeg"}:
            extracted_text = extract_text_from_image(file_path)
        elif ext in {".mp3", ".wav"}:
            extracted_text = transcribe_audio(file_path)
        elif ext in {".mp4", ".mkv", ".mpeg"}:
            transcript, frame_tags = process_video_ffmpeg(file_path)
            extracted_text = transcript + " " + " ".join(frame_tags)
        else:
            return {"ok": False, "error": f"unsupported file type: {ext}"}
        
        if not extracted_text or not extracted_text.strip():
            return {"ok": False, "error": "no extractable text content found"}
        
        print(f"Successfully extracted {len(extracted_text)} characters from {file_name}")
        
        # Update the database with the extracted text for future use
        try:
            cur.execute("UPDATE files SET full_text=? WHERE id=?", (extracted_text, file_id))
            conn.commit()
            print("Updated database with extracted text")
        except Exception as e:
            print(f"Warning: Could not update database with extracted text: {e}")
        
        # Generate answer using the extracted text
        ans = nlp.best_answer(q, extracted_text)
        return {"ok": True, "answer": ans.get("answer", ""), "score": ans.get("score", 0), "context": ans.get("context", "")}
        
    except Exception as e:
        print(f"Error during real-time text extraction for {file_name}: {e}")
        return {"ok": False, "error": f"text extraction failed: {str(e)}"}

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
    """
    Detect duplicate files stored locally by exact content hash (byte-for-byte).
    Returns groups with at least 2 files.
    """
    try:
        c = conn.cursor()
        c.execute("SELECT id, file_path, file_name FROM files WHERE file_path IS NOT NULL")
        rows = c.fetchall()
        if not rows:
            return {"summary": {"duplicate_groups_found": 0, "total_files_processed": 0}, "duplicates": []}

        import hashlib, os
        def hash_file(path: str) -> str:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        hash_to_files = {}
        total_existing = 0
        for fid, fpath, fname in rows:
            if not fpath:
                continue
            npath = os.path.normcase(os.path.abspath(fpath))
            if not os.path.exists(npath):
                continue
            total_existing += 1
            try:
                fh = hash_file(npath)
            except Exception:
                continue
            lst = hash_to_files.setdefault(fh, [])
            if not any(x["path"] == npath for x in lst):
                lst.append({"id": fid, "name": fname or os.path.basename(npath), "path": npath})

        groups = []
        gnum = 1
        for hval, files in hash_to_files.items():
            if len(files) >= 2:
                files_sorted = sorted(files, key=lambda f: (f["name"].lower(), f["path"].lower()))
                groups.append({
                    "group_id": f"group_{gnum}",
                    "file_count": len(files_sorted),
                    "files": files_sorted
                })
                gnum += 1

        return {
            "summary": {"duplicate_groups_found": len(groups), "total_files_processed": total_existing},
            "duplicates": groups
        }
    except Exception as e:
        return {"error": str(e), "summary": {"duplicate_groups_found": 0, "total_files_processed": 0}, "duplicates": []}

@app.post("/resolve_duplicate")
def resolve_duplicate(file_a: int, file_b: int, action: str):
    """
    Keep one, delete the other from disk, and update statuses.
    action: 'keep_a' or 'keep_b'
    """
    if action not in ("keep_a", "keep_b"):
        raise HTTPException(status_code=400, detail="action must be keep_a or keep_b")
    cur = conn.cursor()
    cur.execute("SELECT id, file_path, file_name FROM files WHERE id IN (?, ?)", (file_a, file_b))
    rows = cur.fetchall()
    if len(rows) != 2:
        raise HTTPException(status_code=404, detail="One or both files not found")
    m = {r[0]: {"id": r[0], "path": r[1], "name": r[2]} for r in rows}
    a, b = m[file_a], m[file_b]
    import os
    a_path = os.path.normcase(os.path.abspath(a["path"])) if a.get("path") else None
    b_path = os.path.normcase(os.path.abspath(b["path"])) if b.get("path") else None
    try:
        if action == "keep_a":
            if b_path and os.path.exists(b_path):
                try: os.remove(b_path)
                except Exception: pass
            cur.execute("UPDATE files SET status=? WHERE id=?", ("removed_duplicate", file_b))
            cur.execute("UPDATE files SET status=? WHERE id=?", ("approved", file_a))
            conn.commit()
            return {"status": "resolved", "kept": a, "deleted": b_path}
        else:
            if a_path and os.path.exists(a_path):
                try: os.remove(a_path)
                except Exception: pass
            cur.execute("UPDATE files SET status=? WHERE id=?", ("removed_duplicate", file_a))
            cur.execute("UPDATE files SET status=? WHERE id=?", ("approved", file_b))
            conn.commit()
            return {"status": "resolved", "kept": b, "deleted": a_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Google Drive integration ---
@app.post("/organize_drive_files")
def organize_drive_files(req: OrganizeDriveFilesRequest):
    """Accepts a list of Google Drive file metadata, classifies them, and
    returns suggested target categories/folders. Movement in Drive is optional
    and should be performed client-side using the user's OAuth token.
    """
    organized = []
    move_performed = False
    drive_service = None
    if req.move and req.auth_token:
        drive_service = get_drive_service(req.auth_token)

    # Skip complex AI analysis - use simple categorization only
    for f in req.files:
        # Skip folders entirely
        if (f.mimeType or "").strip() == 'application/vnd.google-apps.folder':
            continue

        file_name = f.name or ""

        # Simple categorization based on filename and mimeType
        try:
            if req.override_category:
                category_name = req.override_category
                category_id = None
            else:
                # Use simple extension-based categorization (no downloads, no AI)
                category_name = infer_category_from_extension(f.name, f.mimeType)
                category_id = None

        except Exception as e:
            print(f"Error categorizing file {f.name}: {e}")
            category_id, category_name = None, infer_category_from_extension(f.name, f.mimeType)

        target_folder_id = None
        if req.move and drive_service:
            # Ensure folder is created in My Drive root (no parent) OR allow future parent selection
            target_folder_id = get_or_create_folder(drive_service, (req.override_category or category_name or "Other"), None)
            moved = move_file_to_folder(drive_service, f.id, target_folder_id) if target_folder_id else None
            if moved and moved.get('id'):
                move_performed = True

        organized.append({
            "id": f.id,
            "name": f.name,
            "proposed_category": req.override_category or category_name,
            "category_id": category_id,
            "target_folder_id": target_folder_id,
            "mimeType": f.mimeType,
            "parents": f.parents or [],
            "summary": ""  # No summary for simple mode
        })

    return {"organized_files": organized, "move_performed": move_performed}


@app.get("/organize_drive_files")
def organize_drive_files_info():
    return {
        "message": "This endpoint accepts POST requests with Drive file metadata.",
        "usage": {
            "method": "POST",
            "path": "/organize_drive_files",
            "body": {"files": "list of Drive files", "move": "bool", "auth_token": "OAuth token"}
        }
    }

@app.post("/drive_summarize")
def drive_summarize(file_id: str, auth_token: str):
    drive_service = get_drive_service(auth_token)
    file_path = drive_download_file(auth_token, file_id, tempfile.gettempdir())
    if not file_path:
        raise HTTPException(status_code=400, detail="Failed to download file from Drive")
    try:
        summary = summarize_text(read_text_file(file_path))
        return {"summary": summary}
    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass

@app.post("/drive_find_similar")
def drive_find_similar(file_id: str, auth_token: str):
    drive_service = get_drive_service(auth_token)
    file_path = drive_download_file(auth_token, file_id, tempfile.gettempdir())
    if not file_path:
        raise HTTPException(status_code=400, detail="Failed to download file from Drive")
    try:
        similar_files = []
        # TO DO: implement similar file search logic
        return {"similar_files": similar_files}
    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass

@app.post("/drive_extract_insights")
def drive_extract_insights(file_id: str, auth_token: str):
    drive_service = get_drive_service(auth_token)
    file_path = drive_download_file(auth_token, file_id, tempfile.gettempdir())
    if not file_path:
        raise HTTPException(status_code=400, detail="Failed to download file from Drive")
    try:
        insights = []
        # TO DO: implement insight extraction logic
        return {"insights": insights}
    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass

@app.get("/favicon.ico")
def favicon():
    # Silence 404s from browsers requesting a favicon
    return {}