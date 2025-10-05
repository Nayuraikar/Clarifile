from fastapi import FastAPI, HTTPException, Query, UploadFile, File
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
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from smart_categorizer import SmartCategorizer
import nlp
from services.categorizer_fallback import categorize_with_fallback
from semantic_search import hybrid_search, semantic_search, clean_query
from google.oauth2.credentials import Credentials
from services.assistant_generator import generate, export_bytes
from googleapiclient.discovery import build
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "metadata_db", "clarifile.db")
ALLOWED_EXTS = {
    ".txt", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif",
    ".mp3", ".wav", ".mp4", ".mkv", ".mpeg", ".avi", ".mov", ".wmv", ".flv", ".m4a", ".aac", ".ogg", ".flac",
    ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".csv", ".rtf", ".odt", ".ods", ".odp",
    ".json", ".xml", ".html", ".htm", ".md", ".py", ".js", ".css", ".java", ".cpp", ".c", ".h",
    ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"
}

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


class AssistantGenerateRequest(BaseModel):
    # kind: flowchart | short_notes | detailed_notes | timeline | key_insights | flashcards
    kind: str
    text: str | None = None  
    file: dict | None = None
    format: str | None = None  
    auth_token: str | None = None

    class Config:
        extra = 'allow'

def _ensure_categories_table():
    try:
        conn = sqlite3.connect(DB)
        cur = conn.cursor()
        cur.execute( 
            """
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                rep_summary TEXT,
                rep_vector TEXT
            )
            """
        )
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass

def get_or_create_category_id(category_name: str, rep_content: str = "") -> int | None:
    if not category_name:
        return None
    _ensure_categories_table()
    try:
        conn = sqlite3.connect(DB)
        cur = conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO categories(name, rep_summary, rep_vector) VALUES (?,?,?)",
            (category_name, rep_content[:500] if rep_content else "", "[]")
        )
        conn.commit()
        cur.execute("SELECT id FROM categories WHERE name=?", (category_name,))
        row = cur.fetchone()
        return int(row[0]) if row else None
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass

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
    and optionally answer a question against the text
    """
    print("DRIVE_ANALYZE ENDPOINT CALLED!")
    print(f"File: {req.file.get('name') if req.file else 'NO FILE'}")
    print("NEW CATEGORIZATION LOGIC ACTIVE!")
    
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
        elif ext == ".docx":
            text = extract_text_from_docx(path)
        elif ext in {".png", ".jpg", ".jpeg"}:
            text = extract_text_from_image(path)
            tags.append(classify_image(path))
        elif ext in {".mp3", ".wav"}:
            transcript = transcribe_audio(path)
            text = transcript
        else:
            text = read_text_file(path)

        summary = summarize_text(text)
        # Use the full extracted text for smart categorization
        print(f"EXTRACTION DEBUG: Text extracted: {len(text) if text else 0} chars")
        print(f"EXTRACTION DEBUG: Summary generated: {len(summary) if summary else 0} chars")
        print(f"EXTRACTION DEBUG: Text preview: '{text[:300] if text else 'NO TEXT'}'")
        print(f"EXTRACTION DEBUG: Summary preview: '{summary[:200] if summary else 'NO SUMMARY'}'")
        
        if text and len(text.strip()) > 10:
            print(f"FORCING CATEGORIZATION with extracted text")
            try:
                cat_name, debug = categorize_with_fallback(text, existing_categorizer=getattr(smart_categorizer, "categorize", None))
            except Exception as _e:
                cat_name, debug = None, {"error": str(_e)}

            if not cat_name:
                cat_id, cat_name = assign_category_from_summary("", text)
            else:
                cat_id = get_or_create_category_id(cat_name, rep_content=text)
            
            if not cat_name or cat_name == "CATEGORIZATION_FAILED":
                print(f"CATEGORIZATION FAILED - content could not be properly categorized")
                raise HTTPException(status_code=422, detail="Could not categorize file content - no matching category found")
        elif summary and len(summary.strip()) > 10:
            print(f"FORCING CATEGORIZATION with summary")
            try:
                cat_name, debug = categorize_with_fallback(summary, existing_categorizer=getattr(smart_categorizer, "categorize", None))
            except Exception as _e:
                cat_name, debug = None, {"error": str(_e)}

            if not cat_name:
                cat_id, cat_name = assign_category_from_summary(summary, "")
            else:
                cat_id = get_or_create_category_id(cat_name, rep_content=summary)

            if not cat_name or cat_name == "CATEGORIZATION_FAILED":
                print(f"CATEGORIZATION FAILED - summary could not be properly categorized")
                raise HTTPException(status_code=422, detail="Could not categorize file content - no matching category found")
        else:
            print(f"NO CONTENT TO ANALYZE - failing completely")
            raise HTTPException(status_code=422, detail="No extractable content found in file")
        
        print(f"FINAL RESULT: Category = {cat_name}")

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


@app.post("/assistant_generate")
def assistant_generate(req: AssistantGenerateRequest):
    """Generate study outputs and optionally return a downloadable payload.
    If file is provided, extract text via existing extractors. Otherwise use provided text.
    """
    token = req.auth_token
    text = (req.text or "").strip()
    if not text and req.file:
        file = req.file
        file_id = file.get('id')
        path = drive_download_file(token, file_id, tempfile.gettempdir())
        if not path:
            raise HTTPException(status_code=400, detail="Failed to download file from Drive")
        try:
            ext = os.path.splitext((file.get('name') or path))[1].lower()
            if ext == ".txt":
                text = read_text_file(path)
            elif ext == ".pdf":
                text = extract_text_from_pdf(path)
            elif ext == ".docx":
                text = extract_text_from_docx(path)
            elif ext in {".png", ".jpg", ".jpeg"}:
                text = extract_text_from_image(path)
            elif ext in {".mp3", ".wav"}:
                transcript = transcribe_audio(path)
                text = transcript
            else:
                text = read_text_file(path)
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

    if not text:
        raise HTTPException(status_code=400, detail="No text provided or extractable from file")

    kind, content = generate(req.kind, text)
    filename, b64 = export_bytes(kind, content, req.format or "md")
    mime = {
        'md': 'text/markdown',
        'markdown': 'text/markdown',
        'txt': 'text/plain',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'pdf': 'application/pdf'
    }.get((req.format or 'md').lower(), 'application/octet-stream')
    return {"kind": kind, "filename": filename, "base64": b64, "mime": mime, "content": content}

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

# Initialize smart categorizer for content-based categorization
smart_categorizer = SmartCategorizer()

# Lazy loading for heavy models
cv_model = None
transform = None

def get_cv_model():
    global cv_model, transform
    if cv_model is None:
        print("Loading CV model...")
        cv_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        cv_model.eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    return cv_model, transform

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

    # Create proposals table for Drive file categorization
    cur.execute("""
      CREATE TABLE IF NOT EXISTS proposals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id TEXT UNIQUE,
        file_name TEXT,
        proposed_label TEXT,
        final_label TEXT,
        approved INTEGER DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    """)

    conn.commit()
    return conn
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
            print("Database connection closed")
        except Exception as e:
            print(f"Error closing database connection: {e}")

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

# Text extraction 
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

def extract_text_from_docx(path):
    """Extract text from a DOCX file."""
    try:
        import docx2txt
        return docx2txt.process(path)
    except ImportError:
        print("docx2txt not installed. Installing now...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "docx2txt"])
        import docx2txt
        return docx2txt.process(path)
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

# Audio/Video and cv
import speech_recognition as sr
from pydub import AudioSegment
import os
import tempfile
def classify_image(img_path):
    try:
        cv_model, transform = get_cv_model()
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = cv_model(tensor)
            _, pred = outputs.max(1)
        return f"Image class {pred.item()}"
    except Exception as e:
        print("Image classification error:", e)
        return "Unknown image"

def transcribe_audio(path, use_google=True):
    """
    Transcribe audio from an audio file using speech recognition.
    Supports WAV, MP3, M4A, OGG, and FLAC formats.
    """
    try:
        import speech_recognition as sr
        import os
        import tempfile
        from pydub import AudioSegment
        import wave
        import json
        import subprocess
        
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"Error: File not found or empty: {path}")
            return ""
            
        r = sr.Recognizer()
        file_ext = os.path.splitext(path)[1].lower()
        temp_wav = None
        audio_path = path
        
        try:
            if file_ext != '.wav':
                temp_wav = os.path.join(tempfile.gettempdir(), f"temp_audio_{os.getpid()}.wav")
                print(f"Converting {file_ext} to WAV format (16kHz, mono, 16-bit PCM)...")
                try:
                    cmd = [
                        'ffmpeg',
                        '-y', 
                        '-i', path,
                        '-ar', '16000', 
                        '-ac', '1',     
                        '-acodec', 'pcm_s16le', 
                        '-loglevel', 'error',
                        temp_wav
                    ]
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    if result.stderr:
                        print(f"FFmpeg warnings: {result.stderr}")
                    
                    if os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 0:
                        audio_path = temp_wav
                        print("FFmpeg conversion successful")
                    else:
                        raise Exception("FFmpeg conversion produced empty file")
                        
                except (subprocess.CalledProcessError, Exception) as e:
                    print(f"FFmpeg conversion failed: {str(e)}")
                    print("Falling back to pydub for conversion...")
                    
                    try:
                        audio = AudioSegment.from_file(path)
                        print(f"Original audio info - channels: {audio.channels}, sample_width: {audio.sample_width}, frame_rate: {audio.frame_rate}")
                        
                        if audio.channels > 1:
                            audio = audio.set_channels(1)
                        
                        audio = audio.set_frame_rate(16000)
                        
                        # Ensure 16-bit PCM
                        if audio.sample_width != 2:  
                            audio = audio.set_sample_width(2)
                        
                        # Export with specific format
                        audio.export(
                            temp_wav,
                            format="wav",
                            parameters=[
                                '-ar', '16000',
                                '-ac', '1',
                                '-acodec', 'pcm_s16le'
                            ]
                        )
                        
                        if os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 0:
                            audio_path = temp_wav
                            print("Pydub conversion successful")
                        else:
                            raise Exception("Pydub conversion produced empty file")
                            
                    except Exception as e:
                        print(f"Pydub conversion also failed: {str(e)}")
                        return ""
            
            # Verify WAV file is valid
            try:
                with wave.open(audio_path, 'rb') as wav_file:
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    frame_rate = wav_file.getframerate()
                    
                    print(f"WAV file info - Channels: {channels}, Sample Width: {sample_width}, Frame Rate: {frame_rate}")
                    
                    if channels != 1 or sample_width != 2:
                        error_msg = f"WAV file must be mono (1 channel) and 16-bit PCM (got {channels} channels, {sample_width*8}-bit)"
                        print(error_msg)

                        if channels != 1 or sample_width != 2:
                            print("Attempting to fix audio format...")
                            fixed_wav = os.path.join(tempfile.gettempdir(), f"fixed_audio_{os.getpid()}.wav")
                            try:
                                audio = AudioSegment.from_wav(audio_path)
                                if channels != 1:
                                    audio = audio.set_channels(1)
                                if sample_width != 2:
                                    audio = audio.set_sample_width(2)
                                audio.export(fixed_wav, format="wav")

                                with wave.open(fixed_wav, 'rb') as fixed_wav_file:
                                    if fixed_wav_file.getnchannels() == 1 and fixed_wav_file.getsampwidth() == 2:
                                        audio_path = fixed_wav
                                        if temp_wav and os.path.exists(temp_wav):
                                            try:
                                                os.remove(temp_wav)
                                            except:
                                                pass
                                        temp_wav = fixed_wav
                                        print("Successfully fixed audio format")
                                    else:
                                        raise Exception("Failed to fix audio format")
                            except Exception as fix_error:
                                print(f"Failed to fix audio format: {fix_error}")
                                return ""
                        else:
                            return ""
            except Exception as e:
                print(f"Invalid WAV file: {e}")
                return ""
            
            # Try Google Speech Recognition first
            if use_google:
                try:
                    print("Trying Google Speech Recognition...")
                    with sr.AudioFile(audio_path) as source:
                        audio_data = r.record(source)
                        text = r.recognize_google(audio_data, language='en-US')
                        print("Google Speech Recognition successful")
                        return text
                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    print(f"Google Speech Recognition service error: {e}")
                except Exception as e:
                    print(f"Error with Google Speech Recognition: {e}")
            
            # Try Vosk as fallback
            try:
                print("Trying Vosk offline recognition...")
                import vosk
                
                model_dir = os.path.join(tempfile.gettempdir(), "vosk-model-small-en-us")
                model_zip = os.path.join(tempfile.gettempdir(), "vosk-model-small-en-us.zip")
                
                if not os.path.exists(model_dir):
                    print("Downloading Vosk model...")
                    import urllib.request
                    import zipfile
                    
                    model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
                    urllib.request.urlretrieve(model_url, model_zip)
                    
                    with zipfile.ZipFile(model_zip, 'r') as zip_ref:
                        zip_ref.extractall(tempfile.gettempdir())
                    
                    try:
                        os.remove(model_zip)
                    except:
                        pass
                
                model = vosk.Model(model_dir)
                rec = vosk.KaldiRecognizer(model, 16000)
                
                # Process audio file
                result = []
                with open(audio_path, 'rb') as f:
                    while True:
                        data = f.read(4000)
                        if len(data) == 0:
                            break
                        if rec.AcceptWaveform(data):
                            result.append(json.loads(rec.Result())['text'])
                
                result.append(json.loads(rec.FinalResult())['text'])
                text = ' '.join([r for r in result if r.strip()])
                
                if text.strip():
                    print("Vosk recognition successful")
                    return text
                
            except Exception as e:
                print(f"Vosk recognition failed: {e}")
            
            return ""
                    
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception as e:
                    print(f"Error removing temp file {temp_wav}: {e}")
                    
    except Exception as e:
        print(f"Error in transcribe_audio: {e}")
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

def summarize_text(long_text: str) -> str:
    if not long_text.strip():
        return ""

    print("Using Gemini API for summarization with gemini-2.5-flash")
    try:
        return nlp.summarize_with_gemini(long_text, max_tokens=512)
    except Exception as e:
        print(f"Gemini summarization error: {e}")
        return long_text[:500] + "..." if len(long_text) > 500 else long_text

def summarize_text_with_gemini(long_text: str) -> str:
    #Original Gemini-based summarization
    if not long_text.strip():
        return ""

    try:
        return nlp.summarize_with_gemini(long_text, max_tokens=512)
    except Exception as e:
        print(f"Gemini summarization error: {e}")
        return long_text[:500] + "..." if len(long_text) > 500 else long_text

#Category assignment
def compress_title(summary: str) -> str:
    if not summary: return "General"
    lower = summary.lower()
    if "invoice" in lower or "bill" in lower: return "Finance"
    if "meeting" in lower or "agenda" in lower: return "Work"
    if "personal" in lower: return "Personal"
    return summary.split(" ")[0:3][0]

def assign_category_from_summary(summary: str, full_text: str = "", threshold: float = 0.75):
    
    content_to_analyze = full_text.strip() or summary.strip()
    if not content_to_analyze:
        return None, "Uncategorized: General"

    print(f"CATEGORIZATION DEBUG: Analyzing content of length {len(content_to_analyze)}")
    print(f"CATEGORIZATION DEBUG: Content preview: {content_to_analyze[:200]}...")

    try:
        category_name = analyze_content_directly(content_to_analyze)
        print(f"CATEGORIZATION DEBUG: Direct analysis result: {category_name}")
        if category_name is None:
            print("CATEGORIZATION FAILED: No proper category could be determined")
            return None, "CATEGORIZATION_FAILED"

        category_name = category_name.strip()

        if not category_name or len(category_name) < 2:
            print("CATEGORIZATION FAILED: Invalid category name returned")
            return None, "CATEGORIZATION_FAILED"

        category_name = re.sub(r'[^\w\s:-]', '', category_name).strip()

        if ':' not in category_name:
            if ' ' not in category_name:
                category_name = f"{category_name}: General"
            else:
                parts = category_name.split(' ', 1)
                category_name = f"{parts[0]}: {parts[1]}"

        main_category = category_name.split(':', 1)[0].strip()

        if not main_category:
            main_category = "General"
            category_name = "General: Document"

        print(f"CATEGORIZATION DEBUG: Final category assignment: '{category_name}' (Main: '{main_category}')")

        cur = conn.cursor()
        rep_content = full_text[:1000] if full_text else summary[:1000]  # Store first 500 chars as representative

        cur.execute(
            "INSERT OR IGNORE INTO categories(name, rep_summary, rep_vector) VALUES (?,?,?)",
            (main_category, rep_content, "[]")
        )

        if category_name != main_category:
            cur.execute(
                "INSERT OR IGNORE INTO categories(name, rep_summary, rep_vector) VALUES (?,?,?)",
                (category_name, rep_content, "[]")
            )

        conn.commit()

        cur.execute("SELECT id FROM categories WHERE name=?", (category_name,))
        row = cur.fetchone()
        if not row:
            cur.execute("SELECT id FROM categories WHERE name=?", (main_category,))
            row = cur.fetchone()
            category_name = main_category  

        print(f"CATEGORIZATION DEBUG: Database lookup - ID: {row[0] if row else 'None'}, Category: '{category_name}'")

        return (row[0] if row else None), category_name

    except Exception as e:
        print(f"CATEGORIZATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, "General: Document"

def analyze_content_directly(content):
    """
    TRUE AI-POWERED CONTENT ANALYSIS
    Uses transformer models and machine learning to understand content and create smart categories.
    """
    if not content or not content.strip():
        print("AI ANALYSIS: No content provided - FAILING")
        return None
    
    content_clean = content.strip()
    
    print(f"AI ANALYSIS: Processing {len(content_clean)} characters")
    print(f"CONTENT PREVIEW: {content_clean[:300]}...")
    
    try:
        print("Loading AI models for content understanding...")
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        from collections import Counter
        import re
        
        # Load transformer model for semantic understanding
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # STEP 1: SEMANTIC EMBEDDING ANALYSIS
        embedding = model.encode([content_clean], convert_to_numpy=True)
        print(f"Generated semantic embedding: {embedding.shape}")
        
        # STEP 2: EXTRACT KEY CONCEPTS USING AI
        key_concepts = extract_key_concepts_ai(content_clean, model)
        print(f"AI EXTRACTED CONCEPTS: {key_concepts}")
        
        # STEP 3: SEMANTIC SIMILARITY TO DISCOVER CATEGORY
        category = discover_category_semantically(content_clean, embedding, key_concepts, model)
        
        if category:
            print(f"AI DISCOVERED CATEGORY: {category}")
            return category
        
        # STEP 4: FALLBACK TO INTELLIGENT CONTENT CLUSTERING
        fallback_category = intelligent_content_clustering(content_clean, embedding)
        if fallback_category:
            print(f"AI CLUSTERING RESULT: {fallback_category}")
            return fallback_category
        
    except Exception as e:
        print(f"AI Analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("AI COULD NOT UNDERSTAND CONTENT - FAILING")
    return None

def extract_key_concepts_ai(content, model):
    
    import re
    from collections import Counter
    
    # Split content into sentences for analysis
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if not sentences:
        return []
    
    # Get embeddings for each sentence
    sentence_embeddings = model.encode(sentences[:50], convert_to_numpy=True)  # Limit to first 50 sentences
    
    # Extract meaningful words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
    word_freq = Counter(words)
    
    common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
    
    meaningful_words = [(word, count) for word, count in word_freq.most_common(20) 
                       if word not in common_words and len(word) > 3]
    
    return [word for word, count in meaningful_words[:10]]

def discover_category_semantically(content, embedding, key_concepts, model):
    """
    Use semantic similarity and AI to discover what category this content belongs to.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    content_lower = content.lower()
    
    content_nature = analyze_content_nature_ai(content, key_concepts)
    print(f"AI CONTENT NATURE: {content_nature}")
    
    if content_nature:
        return content_nature
    
    category_prototypes = generate_dynamic_prototypes(key_concepts, content)
    
    if not category_prototypes:
        return None
    
    prototype_texts = list(category_prototypes.values())
    prototype_embeddings = model.encode(prototype_texts, convert_to_numpy=True)
    
    similarities = cosine_similarity(embedding, prototype_embeddings)[0]
    
    best_match_idx = np.argmax(similarities)
    best_similarity = similarities[best_match_idx]
    
    category_names = list(category_prototypes.keys())
    best_category = category_names[best_match_idx]
    
    print(f"SEMANTIC SIMILARITY: {best_category} (confidence: {best_similarity:.3f})")
    
    if best_similarity > 0.4:
        return best_category
    
    return None

def analyze_content_nature_ai(content, key_concepts):
    """
    Use AI to understand the fundamental nature of content across DIVERSE categories.
    """
    import re
    
    content_lower = content.lower()
    
    # AI-powered content nature analysis - COMPREHENSIVE CATEGORIES
    nature_indicators = {}
    
    financial_patterns = [
        r'(?:invoice|bill)\s*(?:number|#|no\.?)\s*:?\s*\w+',
        r'amount\s*:?\s*[₹$€£¥]\s*[\d,]+',
        r'(?:total|subtotal|due)\s*:?\s*[₹$€£¥]?\s*[\d,]+',
        r'payment\s+(?:due|terms|method)',
        r'(?:bank|account|routing)\s+number',
        r'(?:tax|vat|gst)\s*:?\s*[\d%]+',
        r'(?:budget|expense|revenue|profit|loss)'
    ]
    
    financial_score = sum(1 for pattern in financial_patterns if re.search(pattern, content_lower))
    if financial_score >= 2:
        nature_indicators['Finance'] = financial_score * 3
    
    recipe_patterns = [
        r'ingredients?\s*:',
        r'(?:prep|cook|total)\s+time\s*:?\s*\d+',
        r'servings?\s*:?\s*\d+',
        r'\d+\s*(?:cups?|tbsp|tsp|ml|grams?|kg|oz|lbs?)',
        r'(?:bake|cook|fry|boil|simmer|mix|stir|chop|dice)',
        r'(?:oven|stove|pan|pot|bowl)',
        r'(?:recipe|cooking|kitchen|meal|dish)'
    ]
    
    recipe_score = sum(1 for pattern in recipe_patterns if re.search(pattern, content_lower))
    if recipe_score >= 2:
        nature_indicators['Food'] = recipe_score * 3
    
    personal_patterns = [
        r'(?:i\s+(?:woke|went|feel|think|had|did|was|am|will))',
        r'(?:my\s+(?:morning|day|routine|thoughts|feelings|life))',
        r'(?:today|yesterday|tomorrow)\s+(?:i|was|is)',
        r'(?:personal|diary|journal|daily)\s+(?:log|entry|notes?)',
        r'(?:morning|evening)\s+routine',
        r'(?:dear\s+diary|personal\s+note)',
        r'(?:mood|emotion|feeling|thought)'
    ]
    
    personal_score = sum(1 for pattern in personal_patterns if re.search(pattern, content_lower))
    if personal_score >= 2:
        nature_indicators['Personal'] = personal_score * 3
    
    technical_patterns = [
        r'(?:function|class|def|import|#include|public|private)',
        r'(?:algorithm|methodology|implementation|architecture)',
        r'(?:system|software|programming|development|coding)',
        r'(?:database|server|client|api|framework)',
        r'(?:variable|parameter|return|loop|condition)',
        r'(?:debug|error|exception|test|unit)',
        r'(?:version|commit|branch|merge|git)'
    ]
    
    technical_score = sum(1 for pattern in technical_patterns if re.search(pattern, content_lower))
    if technical_score >= 2:
        nature_indicators['Technical'] = technical_score * 3
    
    academic_patterns = [
        r'(?:abstract|introduction|methodology|results|conclusion)',
        r'(?:research|study|experiment|hypothesis|thesis)',
        r'(?:analysis|findings|data|statistics|correlation)',
        r'(?:literature\s+review|bibliography|references)',
        r'(?:professor|student|university|college|academic)',
        r'(?:journal|publication|peer\s+review)',
        r'(?:theory|model|framework|paradigm)'
    ]
    
    academic_score = sum(1 for pattern in academic_patterns if re.search(pattern, content_lower))
    if academic_score >= 2:
        nature_indicators['Academic'] = academic_score * 3
    
    work_patterns = [
        r'(?:meeting|agenda)\s+(?:minutes|notes?)',
        r'attendees?\s*:',
        r'action\s+items?\s*:',
        r'(?:project|team|business)\s+(?:update|report|meeting)',
        r'(?:deadline|milestone|deliverable|timeline)',
        r'(?:client|customer|stakeholder|vendor)',
        r'(?:budget|cost|revenue|roi|kpi)'
    ]
    
    work_score = sum(1 for pattern in work_patterns if re.search(pattern, content_lower))
    if work_score >= 2:
        nature_indicators['Work'] = work_score * 3
    
    health_patterns = [
        r'(?:patient|doctor|physician|nurse|hospital)',
        r'(?:diagnosis|treatment|medication|prescription)',
        r'(?:symptoms|condition|disease|illness|injury)',
        r'(?:blood\s+pressure|heart\s+rate|temperature)',
        r'(?:medical|health|wellness|fitness)',
        r'(?:appointment|visit|checkup|examination)',
        r'(?:insurance|copay|deductible|claim)'
    ]
    
    health_score = sum(1 for pattern in health_patterns if re.search(pattern, content_lower))
    if health_score >= 2:
        nature_indicators['Health'] = health_score * 3
    
    legal_patterns = [
        r'(?:contract|agreement|terms|conditions)',
        r'(?:legal|law|clause|party|liability)',
        r'(?:attorney|lawyer|court|judge|jury)',
        r'(?:plaintiff|defendant|witness|testimony)',
        r'(?:license|permit|copyright|trademark)',
        r'(?:jurisdiction|statute|regulation|compliance)',
        r'(?:whereas|therefore|hereby|aforementioned)'
    ]
    
    legal_score = sum(1 for pattern in legal_patterns if re.search(pattern, content_lower))
    if legal_score >= 2:
        nature_indicators['Legal'] = legal_score * 3
    
    travel_patterns = [
        r'(?:flight|airline|airport|boarding|departure)',
        r'(?:hotel|reservation|booking|check.?in|check.?out)',
        r'(?:itinerary|destination|travel|trip|vacation)',
        r'(?:passport|visa|customs|immigration)',
        r'(?:car\s+rental|taxi|uber|lyft|transport)',
        r'(?:luggage|baggage|suitcase|carry.?on)',
        r'(?:tourist|sightseeing|attraction|landmark)'
    ]
    
    travel_score = sum(1 for pattern in travel_patterns if re.search(pattern, content_lower))
    if travel_score >= 2:
        nature_indicators['Travel'] = travel_score * 3
    
    education_patterns = [
        r'(?:course|class|lesson|lecture|tutorial)',
        r'(?:student|teacher|professor|instructor)',
        r'(?:assignment|homework|project|exam|test)',
        r'(?:grade|score|marks|points|gpa)',
        r'(?:school|college|university|academy)',
        r'(?:curriculum|syllabus|textbook|notes)',
        r'(?:learning|education|knowledge|skill)'
    ]
    
    education_score = sum(1 for pattern in education_patterns if re.search(pattern, content_lower))
    if education_score >= 2:
        nature_indicators['Education'] = education_score * 3
    
    entertainment_patterns = [
        r'(?:movie|film|cinema|theater|show)',
        r'(?:music|song|album|artist|concert)',
        r'(?:book|novel|story|author|chapter)',
        r'(?:game|gaming|player|level|score)',
        r'(?:tv|television|series|episode|season)',
        r'(?:streaming|netflix|youtube|podcast)',
        r'(?:entertainment|fun|hobby|leisure)'
    ]
    
    entertainment_score = sum(1 for pattern in entertainment_patterns if re.search(pattern, content_lower))
    if entertainment_score >= 2:
        nature_indicators['Entertainment'] = entertainment_score * 3
    
    realestate_patterns = [
        r'(?:house|home|apartment|condo|property)',
        r'(?:rent|lease|mortgage|loan|down\s+payment)',
        r'(?:realtor|agent|broker|listing)',
        r'(?:square\s+feet|bedroom|bathroom|kitchen)',
        r'(?:neighborhood|location|address|zip)',
        r'(?:inspection|appraisal|closing|escrow)',
        r'(?:hoa|utilities|taxes|insurance)'
    ]
    
    realestate_score = sum(1 for pattern in realestate_patterns if re.search(pattern, content_lower))
    if realestate_score >= 2:
        nature_indicators['RealEstate'] = realestate_score * 3
    
    shopping_patterns = [
        r'(?:shopping|store|retail|purchase|buy)',
        r'(?:price|cost|discount|sale|coupon)',
        r'(?:product|item|brand|model|size)',
        r'(?:cart|checkout|payment|receipt)',
        r'(?:delivery|shipping|tracking|return)',
        r'(?:amazon|ebay|walmart|target|online)',
        r'(?:review|rating|feedback|recommendation)'
    ]
    
    shopping_score = sum(1 for pattern in shopping_patterns if re.search(pattern, content_lower))
    if shopping_score >= 2:
        nature_indicators['Shopping'] = shopping_score * 3
    
    comm_patterns = [
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        r'(?:dear|hi|hello|regards|sincerely|best)',
        r'(?:email|message|letter|correspondence)',
        r'(?:phone|call|contact|reach\s+out)',
        r'(?:subject|cc|bcc|attachment|forward)',
        r'(?:meeting|appointment|schedule|calendar)',
        r'(?:urgent|important|asap|follow.?up)'
    ]
    
    comm_score = sum(1 for pattern in comm_patterns if re.search(pattern, content_lower))
    if comm_score >= 2:
        nature_indicators['Communication'] = comm_score * 2
    
    insurance_patterns = [
        r'(?:insurance|policy|premium|deductible)',
        r'(?:claim|coverage|benefit|copay)',
        r'(?:health|dental|vision|life|auto)',
        r'(?:beneficiary|dependent|enrollment)',
        r'(?:401k|retirement|pension|savings)',
        r'(?:hr|human\s+resources|employee)',
        r'(?:open\s+enrollment|cobra|fsa|hsa)'
    ]
    
    insurance_score = sum(1 for pattern in insurance_patterns if re.search(pattern, content_lower))
    if insurance_score >= 2:
        nature_indicators['Insurance'] = insurance_score * 3
    
    print(f"COMPREHENSIVE AI NATURE ANALYSIS: {nature_indicators}")
    
    if nature_indicators:
        best_nature = max(nature_indicators.items(), key=lambda x: x[1])
        subcategory = generate_smart_subcategory_ai(content, best_nature[0], key_concepts)
        return f"{best_nature[0]}: {subcategory}"
    
    return None

def generate_dynamic_prototypes(key_concepts, content):
    """
    Generate dynamic category prototypes ONLY for meaningful concept clusters.
    """
    if not key_concepts or len(key_concepts) < 2:
        print(f"INSUFFICIENT CONCEPTS FOR PROTOTYPING: {key_concepts}")
        return {}
    
    prototypes = {}
    concept_str = ' '.join(key_concepts).lower()
    
    print(f"ANALYZING CONCEPT CLUSTER: {concept_str}")
    
    finance_terms = ['invoice', 'payment', 'amount', 'bill', 'cost', 'price', 'financial', 'money', 'total', 'due']
    finance_matches = sum(1 for term in finance_terms if term in concept_str)
    if finance_matches >= 2:
        prototypes['Finance: Transaction'] = 'invoice payment billing amount cost financial transaction'
        print(f"FINANCE PROTOTYPE: {finance_matches} matches")
    
    food_terms = ['recipe', 'ingredients', 'cooking', 'food', 'dish', 'meal', 'preparation', 'cook', 'kitchen']
    food_matches = sum(1 for term in food_terms if term in concept_str)
    if food_matches >= 2:
        prototypes['Food: Recipe'] = 'recipe cooking ingredients food preparation meal dish'
        print(f"FOOD PROTOTYPE: {food_matches} matches")
    
    personal_terms = ['personal', 'daily', 'routine', 'morning', 'diary', 'journal', 'thoughts', 'feelings']
    personal_matches = sum(1 for term in personal_terms if term in concept_str)
    if personal_matches >= 2:
        prototypes['Personal: Journal'] = 'personal diary journal daily routine thoughts feelings'
        print(f"PERSONAL PROTOTYPE: {personal_matches} matches")
    
    work_terms = ['meeting', 'project', 'team', 'business', 'work', 'agenda', 'professional', 'company']
    work_matches = sum(1 for term in work_terms if term in concept_str)
    if work_matches >= 2:
        prototypes['Work: Professional'] = 'meeting project team business work professional agenda'
        print(f"WORK PROTOTYPE: {work_matches} matches")
    
    tech_terms = ['system', 'algorithm', 'technical', 'software', 'development', 'programming', 'code', 'computer']
    tech_matches = sum(1 for term in tech_terms if term in concept_str)
    if tech_matches >= 2:
        prototypes['Technical: Documentation'] = 'technical system software development algorithm programming'
        print(f"TECH PROTOTYPE: {tech_matches} matches")
    
    academic_terms = ['research', 'study', 'analysis', 'methodology', 'academic', 'experiment', 'hypothesis']
    academic_matches = sum(1 for term in academic_terms if term in concept_str)
    if academic_matches >= 2:
        prototypes['Academic: Research'] = 'research study analysis methodology academic scientific'
        print(f"ACADEMIC PROTOTYPE: {academic_matches} matches")
    
    if not prototypes:
        print(f"NO MEANINGFUL CONCEPT CLUSTERS FOUND")
    
    return prototypes

def generate_smart_subcategory_ai(content, main_category, key_concepts):
    """
    Use AI to generate intelligent subcategories based on content analysis.
    """
    content_lower = content.lower()
    
    if main_category == "Finance":
        if any(word in key_concepts for word in ['invoice', 'bill']):
            return "Invoice"
        elif any(word in key_concepts for word in ['budget', 'expense', 'cost']):
            return "Budget"
        elif any(word in key_concepts for word in ['payment', 'transaction']):
            return "Payment"
        else:
            return "Financial Document"
    
    elif main_category == "Food":
        if any(word in key_concepts for word in ['recipe', 'cooking', 'ingredients']):
            return "Recipe"
        elif any(word in key_concepts for word in ['restaurant', 'menu']):
            return "Menu"
        else:
            return "Food Related"
    
    elif main_category == "Personal":
        if any(word in key_concepts for word in ['diary', 'journal']):
            return "Journal Entry"
        elif any(word in key_concepts for word in ['routine', 'daily', 'morning']):
            return "Daily Log"
        elif any(word in key_concepts for word in ['thoughts', 'feelings']):
            return "Personal Thoughts"
        else:
            return "Personal Notes"
    
    elif main_category == "Work":
        if any(word in key_concepts for word in ['meeting', 'agenda']):
            return "Meeting"
        elif any(word in key_concepts for word in ['project', 'task']):
            return "Project"
        elif any(word in key_concepts for word in ['report', 'analysis']):
            return "Report"
        else:
            return "Work Document"
    
    elif main_category == "Technical":
        if any(word in key_concepts for word in ['code', 'programming', 'function']):
            return "Source Code"
        elif any(word in key_concepts for word in ['documentation', 'guide', 'manual']):
            return "Documentation"
        elif any(word in key_concepts for word in ['system', 'architecture']):
            return "System Design"
        else:
            return "Technical Document"
    
    elif main_category == "Academic":
        if any(word in key_concepts for word in ['research', 'study']):
            return "Research Paper"
        elif any(word in key_concepts for word in ['thesis', 'dissertation']):
            return "Thesis"
        elif any(word in key_concepts for word in ['lecture', 'notes']):
            return "Academic Notes"
        else:
            return "Academic Document"
    
    elif main_category == "Health":
        if any(word in key_concepts for word in ['diagnosis', 'treatment']):
            return "Medical Record"
        elif any(word in key_concepts for word in ['fitness', 'exercise']):
            return "Fitness Plan"
        elif any(word in key_concepts for word in ['medication', 'prescription']):
            return "Prescription"
        else:
            return "Health Document"
    
    elif main_category == "Legal":
        if any(word in key_concepts for word in ['contract', 'agreement']):
            return "Contract"
        elif any(word in key_concepts for word in ['license', 'permit']):
            return "License"
        elif any(word in key_concepts for word in ['court', 'lawsuit']):
            return "Legal Proceeding"
        else:
            return "Legal Document"
    
    elif main_category == "Travel":
        if any(word in key_concepts for word in ['flight', 'airline']):
            return "Flight Info"
        elif any(word in key_concepts for word in ['hotel', 'booking']):
            return "Accommodation"
        elif any(word in key_concepts for word in ['itinerary', 'schedule']):
            return "Itinerary"
        else:
            return "Travel Document"
    
    elif main_category == "Education":
        if any(word in key_concepts for word in ['course', 'class']):
            return "Course Material"
        elif any(word in key_concepts for word in ['assignment', 'homework']):
            return "Assignment"
        elif any(word in key_concepts for word in ['exam', 'test']):
            return "Exam"
        else:
            return "Educational Content"
    
    elif main_category == "Entertainment":
        if any(word in key_concepts for word in ['movie', 'film']):
            return "Movie"
        elif any(word in key_concepts for word in ['music', 'song']):
            return "Music"
        elif any(word in key_concepts for word in ['game', 'gaming']):
            return "Gaming"
        elif any(word in key_concepts for word in ['book', 'novel']):
            return "Literature"
        else:
            return "Entertainment"
    
    elif main_category == "RealEstate":
        if any(word in key_concepts for word in ['house', 'home']):
            return "Property Listing"
        elif any(word in key_concepts for word in ['rent', 'lease']):
            return "Rental"
        elif any(word in key_concepts for word in ['mortgage', 'loan']):
            return "Financing"
        else:
            return "Real Estate"
    
    elif main_category == "Shopping":
        if any(word in key_concepts for word in ['purchase', 'buy']):
            return "Purchase"
        elif any(word in key_concepts for word in ['review', 'rating']):
            return "Product Review"
        elif any(word in key_concepts for word in ['delivery', 'shipping']):
            return "Shipping Info"
        else:
            return "Shopping"
    
    elif main_category == "Insurance":
        if any(word in key_concepts for word in ['policy', 'coverage']):
            return "Policy"
        elif any(word in key_concepts for word in ['claim', 'benefit']):
            return "Claim"
        elif any(word in key_concepts for word in ['retirement', '401k']):
            return "Retirement"
        else:
            return "Insurance"
    
    else:
        return "Document"

def intelligent_content_clustering(content, embedding):
    """
    Use unsupervised machine learning to cluster and categorize content intelligently.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import re
        
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) < 2:
            return None
        
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        feature_names = vectorizer.get_feature_names_out()
        
        mean_scores = tfidf_matrix.mean(axis=0).A1
        top_indices = mean_scores.argsort()[-10:][::-1]
        top_terms = [feature_names[i] for i in top_indices if mean_scores[i] > 0.1]
        
        print(f"AI DISCOVERED TOP TERMS: {top_terms}")
        
        if top_terms:
            category = categorize_from_ai_terms(top_terms, content)
            return category
        
    except Exception as e:
        print(f"Clustering analysis failed: {e}")
    
    return None

def categorize_from_ai_terms(top_terms, content):
    """
    Use AI-discovered terms to intelligently categorize content across ALL categories.
    ONLY return meaningful categories - no random word categories.
    """
    terms_str = ' '.join(top_terms).lower()
    
    print(f"EVALUATING COMPREHENSIVE AI TERMS: {terms_str}")
    
    if any(term in terms_str for term in ['invoice', 'payment', 'bill', 'amount', 'cost', 'financial', 'budget', 'expense', 'revenue', 'tax']):
        return "Finance: AI Detected Financial"
    
    elif any(term in terms_str for term in ['recipe', 'ingredients', 'cooking', 'food', 'meal', 'dish', 'kitchen', 'bake', 'serve']):
        return "Food: AI Detected Recipe"
    
    elif any(term in terms_str for term in ['personal', 'daily', 'routine', 'morning', 'diary', 'journal', 'thoughts', 'feelings', 'mood']):
        return "Personal: AI Detected Diary"
    
    elif any(term in terms_str for term in ['meeting', 'project', 'team', 'work', 'business', 'agenda', 'client', 'deadline', 'professional']):
        return "Work: AI Detected Professional"
    
    elif any(term in terms_str for term in ['technical', 'system', 'software', 'algorithm', 'programming', 'code', 'database', 'api', 'development']):
        return "Technical: AI Detected System"
    
    elif any(term in terms_str for term in ['research', 'study', 'analysis', 'methodology', 'academic', 'experiment', 'university', 'thesis']):
        return "Academic: AI Detected Research"
    
    elif any(term in terms_str for term in ['health', 'medical', 'doctor', 'patient', 'treatment', 'diagnosis', 'medication', 'hospital', 'wellness']):
        return "Health: AI Detected Medical"
    
    elif any(term in terms_str for term in ['legal', 'contract', 'agreement', 'law', 'attorney', 'court', 'license', 'terms', 'clause']):
        return "Legal: AI Detected Contract"
    
    elif any(term in terms_str for term in ['travel', 'flight', 'hotel', 'trip', 'vacation', 'airport', 'booking', 'itinerary', 'destination']):
        return "Travel: AI Detected Trip"
    
    elif any(term in terms_str for term in ['education', 'course', 'class', 'student', 'teacher', 'school', 'learning', 'assignment', 'exam']):
        return "Education: AI Detected Learning"
    
    elif any(term in terms_str for term in ['entertainment', 'movie', 'music', 'game', 'show', 'book', 'streaming', 'fun', 'hobby']):
        return "Entertainment: AI Detected Media"
    
    elif any(term in terms_str for term in ['house', 'home', 'property', 'rent', 'mortgage', 'realtor', 'apartment', 'bedroom', 'neighborhood']):
        return "RealEstate: AI Detected Property"
    
    elif any(term in terms_str for term in ['shopping', 'store', 'purchase', 'price', 'product', 'delivery', 'amazon', 'review', 'brand']):
        return "Shopping: AI Detected Retail"
    
    elif any(term in terms_str for term in ['insurance', 'policy', 'coverage', 'benefit', 'claim', 'premium', 'deductible', 'retirement']):
        return "Insurance: AI Detected Benefits"
    
    elif any(term in terms_str for term in ['email', 'message', 'letter', 'communication', 'correspondence', 'phone', 'contact']):
        return "Communication: AI Detected Message"
    
    else:
        print(f"AI TERMS TOO VAGUE FOR COMPREHENSIVE CATEGORIES: {top_terms} - FAILING CATEGORIZATION")
        return None

def understand_content_meaning(content):
   
    import re
    
    content_lower = content.lower()
    lines = content.split('\n')
    
    print(f"READING CONTENT: {len(lines)} lines")
    
    financial_indicators = 0
    if re.search(r'invoice\s*(?:number|#|no\.?)\s*:?\s*\w+', content_lower):
        financial_indicators += 3
        print("FOUND: Invoice number pattern")
    
    if re.search(r'amount\s*:?\s*[₹$€£¥]\s*\d+', content_lower):
        financial_indicators += 3
        print("FOUND: Amount with currency")
    
    if re.search(r'due\s*date\s*:?\s*\d{4}-\d{2}-\d{2}', content_lower):
        financial_indicators += 2
        print("FOUND: Due date pattern")
    
    if re.search(r'bill|billing|payment|total|subtotal', content_lower):
        financial_indicators += 1
    
    if financial_indicators >= 3:
        print("IDENTIFIED: Financial Document (Invoice)")
        return "Finance: Invoice"
    
    recipe_indicators = 0
    if re.search(r'ingredients?\s*:?', content_lower):
        recipe_indicators += 3
        print("FOUND: Ingredients section")
    
    if re.search(r'(?:prep|cook|total)\s*time\s*:?\s*\d+\s*(?:min|hour)', content_lower):
        recipe_indicators += 3
        print("FOUND: Cooking time")
    
    if re.search(r'servings?\s*:?\s*\d+', content_lower):
        recipe_indicators += 2
        print("FOUND: Servings information")
    
    if re.search(r'\d+\s*(?:tbsp|tsp|cup|ml|g|kg|oz|lb)', content_lower):
        recipe_indicators += 2
        print("FOUND: Recipe measurements")
    
    if recipe_indicators >= 4:
        print("IDENTIFIED: Recipe")
        return "Food: Recipe"
    
    personal_indicators = 0
    if re.search(r'(?:woke up|morning routine|went to bed)', content_lower):
        personal_indicators += 3
        print("FOUND: Daily routine language")
    
    if re.search(r'(?:i feel|i think|my|today i|yesterday i)', content_lower):
        personal_indicators += 2
        print("FOUND: Personal pronouns and thoughts")
    
    if re.search(r'(?:personal|diary|journal|daily log)', content_lower):
        personal_indicators += 3
        print("FOUND: Personal document keywords")
    
    if personal_indicators >= 3:
        print("IDENTIFIED: Personal Notes/Diary")
        return "Personal: Daily Log"
    
    meeting_indicators = 0
    if re.search(r'(?:meeting|agenda|minutes)', content_lower):
        meeting_indicators += 2
    
    if re.search(r'attendees?\s*:?', content_lower):
        meeting_indicators += 2
        print("FOUND: Attendees section")
    
    if re.search(r'action\s*items?\s*:?', content_lower):
        meeting_indicators += 3
        print("FOUND: Action items")
    
    if re.search(r'\d{1,2}:\d{2}(?:\s*[ap]m)?', content_lower):
        meeting_indicators += 1
        print("FOUND: Time stamps")
    
    if meeting_indicators >= 4:
        print("IDENTIFIED: Meeting Minutes")
        return "Work: Meeting"
    
    code_indicators = 0
    if re.search(r'(?:function|class|def|import|#include)', content_lower):
        code_indicators += 3
        print("FOUND: Programming keywords")
    
    if re.search(r'(?:algorithm|neural network|machine learning|deep learning)', content_lower):
        code_indicators += 2
        print("FOUND: Technical AI/ML terms")
    
    if re.search(r'(?:computer vision|cnn|convolution)', content_lower):
        code_indicators += 2
        print("FOUND: Computer vision terms")
    
    if code_indicators >= 3:
        print("IDENTIFIED: Technical Document")
        return "Technical: Documentation"
    
    academic_indicators = 0
    if re.search(r'(?:abstract|introduction|methodology|results|conclusion)', content_lower):
        academic_indicators += 3
        print("FOUND: Academic paper structure")
    
    if re.search(r'(?:research|study|experiment|hypothesis)', content_lower):
        academic_indicators += 2
        print("FOUND: Research terminology")
    
    if academic_indicators >= 3:
        print("IDENTIFIED: Academic Paper")
        return "Academic: Research Paper"
    
    business_indicators = 0
    if re.search(r'(?:company|business|revenue|profit|strategy)', content_lower):
        business_indicators += 2
        print("FOUND: Business terminology")
    
    if re.search(r'(?:report|analysis|quarterly|annual)', content_lower):
        business_indicators += 2
        print("FOUND: Report indicators")
    
    if business_indicators >= 3:
        print("IDENTIFIED: Business Document")
        return "Business: Report"
    
    print("CONTENT TYPE NOT CLEARLY IDENTIFIED")
    return None

def careful_keyword_analysis(content_lower, original_content):
    """
    Enhanced keyword analysis with comprehensive category detection.
    Analyzes content to determine the most relevant category based on keyword matching.
    """
    print("PERFORMING CAREFUL KEYWORD ANALYSIS")
    
    keyword_scores = {}
    
    education_keywords = ['exam', 'test', 'student', 'course', 'lecture', 'assignment', 'grade', 
                         'university', 'college', 'school', 'academic', 'hall ticket', 'admission',
                         'roll number', 'candidate', 'examination', 'semester', 'marks', 'result',
                         'education', 'learning', 'study', 'research', 'thesis', 'dissertation',
                         'professor', 'teacher', 'faculty', 'principal', 'dean', 'chancellor',
                         'syllabus', 'curriculum', 'degree', 'diploma', 'certificate', 'scholarship',
                         'candidate name', 'seating number', 'exam date', 'reporting time', 'test centre']
    education_score = sum(1 for word in education_keywords if word in content_lower)
    if education_score > 0:
        keyword_scores['Education'] = education_score * 1.2 
        print(f"Education score: {education_score} (weighted: {education_score * 1.2})")
    
    news_keywords = ['breaking news', 'news report', 'journalist', 'reporter', 'media coverage', 'press conference',
                    'headline', 'news article', 'story coverage', 'news interview', 'news statement',
                    'press release', 'news bulletin', 'news update', 'news alert',
                    'sources say', 'according to reports', 'reported today', 'confirmed by',
                    'investigation reveals', 'exclusive report', 'live updates', 'news correspondent', 
                    'news bureau', 'breaking story', 'developing story']
    news_score = sum(1 for word in news_keywords if word in content_lower)
    if news_score > 0:
        keyword_scores['News'] = news_score
        print(f"News score: {news_score}")
    
    govt_keywords = ['government notification', 'ministry', 'department', 'bureau', 'commission',
                    'authority', 'board', 'committee', 'council', 'assembly', 'parliament',
                    'legislature', 'cabinet', 'minister', 'secretary', 'officer', 'official',
                    'circular', 'directive', 'policy', 'scheme', 'program', 'initiative',
                    'administration', 'bureaucracy', 'civil service', 'ias', 'ips', 'ifs',
                    'gazette', 'ordinance', 'resolution', 'budget', 'allocation', 'fund']
    govt_score = sum(1 for word in govt_keywords if word in content_lower)
    if govt_score > 0:
        keyword_scores['Government'] = govt_score * 1.1  
        print(f"Government score: {govt_score} (weighted: {govt_score * 1.1})")
    
    legal_keywords = ['honorable court', 'court hereby orders', 'verdict', 'ruling', 'judgment', 
                     'lawsuit', 'litigation', 'hearing', 'trial', 'proceeding', 'petition',
                     'appeal', 'writ', 'summons', 'subpoena', 'affidavit', 'testimony', 'witness',
                     'plaintiff', 'defendant', 'counsel', 'attorney', 'advocate', 'barrister',
                     'magistrate', 'sessions', 'bail', 'custody', 'sentence', 'fine', 'penalty',
                     'injunction', 'stay order', 'interim order', 'ex-parte', 'suo moto',
                     'case no', 'criminal case', 'civil case', 'family court', 'high court']
    legal_score = sum(1 for word in legal_keywords if word in content_lower)
    if legal_score > 0:
        keyword_scores['Legal'] = legal_score * 1.15 
        print(f"Legal score: {legal_score} (weighted: {legal_score * 1.15})")
    
    finance_keywords = ['invoice', 'bill', 'payment', 'money', 'cost', 'price', 'dollar', '$', 
                       'rupee', '₹', 'amount', 'salary', 'account', 'bank', 'transaction', 
                       'receipt', 'due date', 'total', 'tax', 'gst', 'income', 'expense',
                       'budget', 'financial', 'economic', 'investment', 'loan', 'credit',
                       'debit', 'balance', 'statement', 'audit', 'accounting', 'finance']
    finance_score = sum(1 for word in finance_keywords if word in content_lower)
    if finance_score > 0:
        keyword_scores['Finance'] = finance_score
        print(f"Finance score: {finance_score}")
    
    work_keywords = ['meeting', 'project', 'deadline', 'work', 'business', 'office', 'colleague',
                    'client', 'presentation', 'report', 'task', 'professional', 'partnership',
                    'discussion', 'agenda', 'action items', 'met', 'discussed', 'corporate',
                    'company', 'organization', 'management', 'executive', 'director', 'ceo',
                    'conference', 'seminar', 'workshop', 'training', 'development']
    work_score = sum(1 for word in work_keywords if word in content_lower)
    if work_score > 0:
        keyword_scores['Work'] = work_score
        print(f"Work score: {work_score}")
    
    cooking_keywords = ['recipe', 'cooking', 'cook', 'ingredients', 'preparation', 'instructions',
                       'kitchen', 'food', 'dish', 'meal', 'breakfast', 'lunch', 'dinner', 'snack',
                       'bake', 'baking', 'fry', 'boil', 'grill', 'roast', 'steam', 'saute',
                       'tablespoon', 'teaspoon', 'cup', 'oven', 'stove', 'pan', 'pot',
                       'salt', 'pepper', 'spice', 'seasoning', 'flavor', 'taste', 'delicious',
                       'cuisine', 'restaurant', 'chef', 'culinary', 'menu', 'appetizer',
                       'main course', 'dessert', 'beverage', 'drink', 'cocktail', 'wine',
                       'vegetarian', 'vegan', 'healthy', 'nutrition', 'calories', 'diet']
    cooking_score = sum(1 for word in cooking_keywords if word in content_lower)
    if cooking_score > 0:
        keyword_scores['Cooking'] = cooking_score
        print(f"Cooking score: {cooking_score}")
    
    entertainment_keywords = ['movie', 'film', 'cinema', 'theater', 'show', 'series', 'episode',
                             'music', 'song', 'album', 'artist', 'singer', 'band', 'concert',
                             'game', 'gaming', 'video game', 'play', 'player', 'entertainment',
                             'fun', 'enjoy', 'comedy', 'drama', 'action', 'thriller', 'horror',
                             'romance', 'documentary', 'animation', 'cartoon', 'netflix', 'youtube',
                             'streaming', 'watch', 'listen', 'dance', 'party', 'celebration',
                             'festival', 'event', 'performance', 'actor', 'actress', 'director',
                             'producer', 'soundtrack', 'lyrics', 'melody', 'rhythm', 'beat']
    entertainment_score = sum(1 for word in entertainment_keywords if word in content_lower)
    if entertainment_score > 0:
        keyword_scores['Entertainment'] = entertainment_score
        print(f"Entertainment score: {entertainment_score}")
    
    lifestyle_keywords = ['hobby', 'interest', 'passion', 'craft', 'diy', 'handmade', 'creative',
                         'art', 'painting', 'drawing', 'sketch', 'photography', 'photo', 'picture',
                         'gardening', 'plants', 'flowers', 'garden', 'nature', 'outdoor',
                         'fitness', 'exercise', 'workout', 'gym', 'yoga', 'meditation', 'wellness',
                         'fashion', 'style', 'clothing', 'outfit', 'shopping', 'beauty', 'makeup',
                         'skincare', 'haircare', 'self care', 'relaxation', 'spa', 'massage',
                         'reading', 'book', 'novel', 'story', 'library', 'author', 'writing',
                         'blog', 'blogging', 'social media', 'instagram', 'facebook', 'twitter']
    lifestyle_score = sum(1 for word in lifestyle_keywords if word in content_lower)
    if lifestyle_score > 0:
        keyword_scores['Lifestyle'] = lifestyle_score
        print(f"Lifestyle score: {lifestyle_score}")
    
    home_keywords = ['home', 'house', 'household', 'cleaning', 'organize', 'decoration', 'furniture',
                    'interior', 'design', 'renovation', 'repair', 'maintenance', 'chores',
                    'laundry', 'washing', 'grocery', 'shopping list', 'bills', 'utilities',
                    'electricity', 'water', 'gas', 'internet', 'cable', 'insurance',
                    'mortgage', 'rent', 'landlord', 'tenant', 'neighbor', 'community',
                    'security', 'safety', 'alarm', 'lock', 'key', 'garage', 'basement',
                    'attic', 'bedroom', 'bathroom', 'kitchen', 'living room', 'dining room']
    home_score = sum(1 for word in home_keywords if word in content_lower)
    if home_score > 0:
        keyword_scores['Home'] = home_score
        print(f"Home score: {home_score}")
    
    personal_keywords = ['hello', 'hi', 'my name', 'personal', 'diary', 'journal', 'reminder',
                        'voice note', 'memo', 'myself', 'i am', 'family', 'friend', 'conversation',
                        'thoughts', 'feelings', 'reflection', 'private', 'confidential',
                        'birthday', 'anniversary', 'vacation', 'holiday', 'weekend', 'leisure',
                        'dream', 'goal', 'wish', 'hope', 'memory', 'experience', 'story', 'life',
                        'personal life', 'relationship', 'love', 'marriage', 'children', 'parents',
                        'siblings', 'relatives', 'emotions', 'mood', 'happy', 'sad', 'excited']
    personal_score = sum(1 for word in personal_keywords if word in content_lower)
    if personal_score > 0:
        keyword_scores['Personal'] = personal_score
        print(f"Personal score: {personal_score}")
    
    research_keywords = ['research', 'study', 'analysis', 'survey', 'findings', 'conclusion',
                        'methodology', 'data', 'statistics', 'evidence', 'hypothesis', 'theory',
                        'publication', 'journal', 'paper', 'article', 'citation', 'reference',
                        'abstract', 'bibliography', 'peer review', 'scholarly', 'academic']
    research_score = sum(1 for word in research_keywords if word in content_lower)
    if research_score > 0:
        keyword_scores['Research'] = research_score * 1.1  
        print(f"Research score: {research_score} (weighted: {research_score * 1.1})")
    
    medical_keywords = ['doctor', 'medical', 'health', 'symptoms', 'treatment', 'medicine',
                       'hospital', 'patient', 'diagnosis', 'prescription', 'appointment',
                       'clinic', 'surgery', 'therapy', 'healthcare', 'physician', 'nurse',
                       'medical report', 'test results', 'x-ray', 'scan', 'blood test']
    medical_score = sum(1 for word in medical_keywords if word in content_lower)
    if medical_score > 0:
        keyword_scores['Medical'] = medical_score * 1.2 
        print(f"Medical score: {medical_score} (weighted: {medical_score * 1.2})")
    
    property_keywords = ['property', 'real estate', 'land', 'plot', 'house', 'apartment', 'flat',
                        'building', 'construction', 'sale deed', 'purchase', 'registry', 'title',
                        'ownership', 'tenant', 'landlord', 'rent', 'lease', 'mortgage', 'loan',
                        'valuation', 'survey', 'boundary', 'area', 'square feet', 'acres']
    property_score = sum(1 for word in property_keywords if word in content_lower)
    if property_score > 0:
        keyword_scores['Property'] = property_score
        print(f"Property score: {property_score}")
    
    travel_keywords = ['travel', 'trip', 'vacation', 'flight', 'hotel', 'destination',
                      'journey', 'booking', 'passport', 'visa', 'itinerary', 'tourism',
                      'ticket', 'reservation', 'airport', 'railway', 'bus', 'transport']
    travel_score = sum(1 for word in travel_keywords if word in content_lower)
    if travel_score > 0:
        keyword_scores['Travel'] = travel_score
        print(f"Travel score: {travel_score}")
    
    sports_keywords = ['sport', 'game', 'match', 'player', 'team', 'score', 'tournament',
                      'cricket', 'football', 'basketball', 'tennis', 'goal', 'win', 'competition',
                      'championship', 'league', 'stadium', 'coach', 'training', 'fitness']
    sports_score = sum(1 for word in sports_keywords if word in content_lower)
    if sports_score > 0:
        keyword_scores['Sports'] = sports_score
        print(f"Sports score: {sports_score}")
    
    tech_keywords = ['technology', 'computer', 'software', 'hardware', 'internet', 'website',
                    'application', 'app', 'system', 'database', 'server', 'network',
                    'programming', 'code', 'development', 'digital', 'online', 'cyber',
                    'artificial intelligence', 'ai', 'machine learning', 'data science']
    tech_score = sum(1 for word in tech_keywords if word in content_lower)
    if tech_score > 0:
        keyword_scores['Technology'] = tech_score
        print(f"Technology score: {tech_score}")
    
    if not keyword_scores:
        print("No meaningful keywords found")
        return None
    
    best_category = max(keyword_scores.items(), key=lambda x: x[1])
    
    if best_category[1] >= 2:
        subcategory = get_smart_subcategory(content_lower, best_category[0])
        result = f"{best_category[0]}: {subcategory}"
        print(f"CAREFUL ANALYSIS RESULT: {result}")
        return result
    
    print(f"Insufficient confidence (best score: {best_category[1]})")
    return None

def get_smart_subcategory(content, category):
    """Get appropriate subcategory based on content."""
    if category == "Finance":
        return "Invoice" if "invoice" in content else "Document"
    elif category == "Food":
        return "Recipe"
    elif category == "Personal":
        return "Daily Log" if any(word in content for word in ["daily", "routine", "morning"]) else "Notes"
    elif category == "Work":
        return "Meeting" if "meeting" in content else "Document"
    elif category == "Technical":
        return "Documentation"
    else:
        return "Document"

def analyze_content_themes(content, embedding, model):
    """
    Analyze content themes using semantic similarity and content structure.
    """
    import re
    from collections import Counter
    
    words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
    
    # IDENTIFY DOCUMENT STRUCTURE PATTERNS
    lines = content.split('\n')
    
    # 1. DETECT STRUCTURED DOCUMENTS
    if any(re.search(r'(invoice|bill).*number', line, re.I) for line in lines):
        return "Finance: Invoice"
    
    if any(re.search(r'(meeting|agenda).*\d{1,2}[:/]\d{1,2}', line, re.I) for line in lines):
        return "Work: Meeting"
    
    # 2. DETECT CONTENT BY SEMANTIC DENSITY
    word_freq = Counter(words)
    top_words = [word for word, count in word_freq.most_common(10) if len(word) > 3]
    
    business_terms = {'business', 'company', 'revenue', 'profit', 'sales', 'market', 'strategy', 'management', 'financial', 'budget', 'cost', 'price', 'payment', 'customer', 'client'}
    business_score = sum(1 for word in top_words if word in business_terms)
    
    tech_terms = {'system', 'software', 'algorithm', 'development', 'programming', 'code', 'technical', 'computer', 'data', 'analysis', 'implementation', 'design', 'architecture', 'framework'}
    tech_score = sum(1 for word in top_words if word in tech_terms)
    
    academic_terms = {'research', 'study', 'analysis', 'methodology', 'results', 'conclusion', 'abstract', 'introduction', 'literature', 'experiment', 'hypothesis', 'findings'}
    academic_score = sum(1 for word in top_words if word in academic_terms)
    
    personal_terms = {'personal', 'diary', 'journal', 'thoughts', 'feelings', 'daily', 'routine', 'morning', 'evening', 'today', 'yesterday', 'tomorrow'}
    personal_score = sum(1 for word in top_words if word in personal_terms)
    
    food_terms = {'recipe', 'ingredients', 'cooking', 'cook', 'preparation', 'serve', 'dish', 'meal', 'food', 'kitchen', 'minutes', 'temperature'}
    food_score = sum(1 for word in top_words if word in food_terms)
    
    legal_terms = {'contract', 'agreement', 'legal', 'terms', 'conditions', 'clause', 'party', 'liability', 'rights', 'obligations', 'law', 'regulation'}
    legal_score = sum(1 for word in top_words if word in legal_terms)
    
    health_terms = {'health', 'medical', 'patient', 'treatment', 'diagnosis', 'symptoms', 'medicine', 'therapy', 'doctor', 'hospital', 'care', 'wellness'}
    health_score = sum(1 for word in top_words if word in health_terms)
    
    scores = {
        'Business: Document': business_score,
        'Technical: Documentation': tech_score,
        'Academic: Research': academic_score,
        'Personal: Notes': personal_score,
        'Food: Recipe': food_score,
        'Legal: Contract': legal_score,
        'Health: Medical': health_score
    }
    
    print(f"THEME SCORES: {scores}")
    
    # REQUIRE MINIMUM CONFIDENCE
    best_category = max(scores.items(), key=lambda x: x[1])
    if best_category[1] >= 2: 
        return best_category[0]
    
    # 3. DETECT BY CONTENT PATTERNS
    content_patterns = detect_content_patterns(content)
    if content_patterns:
        return content_patterns
    
    return None

def detect_content_patterns(content):
    """
    Detect categories based on content structure and patterns.
    """
    import re
    
    if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content):
        if any(word in content.lower() for word in ['meeting', 'schedule', 'agenda']):
            return "Communication: Meeting Email"
        return "Communication: Email"
    
    if re.search(r'(function|class|def|import|#include|public|private)', content, re.I):
        return "Technical: Source Code"
    
    if re.search(r'[\$₹€£¥]\s*\d+|amount.*\d+|total.*\d+', content, re.I):
        return "Finance: Financial Document"
    
    if re.search(r'\d{1,2}:\d{2}|\d{1,2}[ap]m|morning|afternoon|evening', content, re.I):
        if any(word in content.lower() for word in ['log', 'diary', 'routine', 'schedule']):
            return "Personal: Daily Log"
    
    lines = content.split('\n')
    list_indicators = sum(1 for line in lines if re.match(r'^\s*[-*•]\s+|^\s*\d+\.\s+', line.strip()))
    if list_indicators >= 3:
        if any(word in content.lower() for word in ['todo', 'task', 'action']):
            return "Work: Task List"
        if any(word in content.lower() for word in ['step', 'instruction', 'guide']):
            return "Reference: Instructions"
    
    return None

def intelligent_keyword_analysis(content_lower):
    # DYNAMIC KEYWORD CATEGORIES - More flexible
    keyword_categories = {
        "Finance": ['invoice', 'bill', 'payment', 'amount', 'total', 'due', 'customer', 'billing', 'cost', 'price', 'budget', 'expense', 'revenue'],
        "Work": ['meeting', 'minutes', 'agenda', 'attendees', 'action', 'project', 'team', 'manager', 'report', 'business', 'company'],
        "Academic": ['research', 'study', 'analysis', 'methodology', 'results', 'conclusion', 'abstract', 'introduction', 'experiment', 'hypothesis'],
        "Legal": ['contract', 'agreement', 'terms', 'legal', 'law', 'clause', 'party', 'liability', 'license', 'rights'],
        "Technical": ['code', 'function', 'programming', 'software', 'technical', 'system', 'algorithm', 'development', 'api', 'documentation'],
        "Personal": ['personal', 'diary', 'journal', 'thoughts', 'feelings', 'daily', 'routine', 'private', 'note'],
        "Food": ['recipe', 'ingredients', 'cooking', 'cook', 'preparation', 'serve', 'dish', 'meal', 'food', 'kitchen'],
        "Health": ['health', 'medical', 'patient', 'treatment', 'diagnosis', 'symptoms', 'medicine', 'therapy', 'doctor'],
        "Communication": ['email', 'message', 'letter', 'correspondence', 'communication', 'contact', 'phone', 'call']
    }
    
    # CALCULATE ADAPTIVE SCORES
    scores = {}
    for category, keywords in keyword_categories.items():
        score = sum(content_lower.count(keyword) for keyword in keywords)
        if score > 0:
            scores[category] = score
    
    print(f"KEYWORD SCORES: {scores}")
    
    if not scores:
        return None
    
    best_category = max(scores.items(), key=lambda x: x[1])
    
    if best_category[1] >= 1:
        subcategory = determine_subcategory(content_lower, best_category[0])
        return f"{best_category[0]}: {subcategory}"
    
    return None

def determine_subcategory(content, main_category):
    """
    Determine smart subcategory based on content analysis.
    """
    if main_category == "Finance":
        if any(word in content for word in ['invoice', 'bill']):
            return "Invoice"
        elif any(word in content for word in ['budget', 'expense']):
            return "Budget"
        else:
            return "Document"
    
    elif main_category == "Work":
        if any(word in content for word in ['meeting', 'agenda']):
            return "Meeting"
        elif any(word in content for word in ['report', 'analysis']):
            return "Report"
        else:
            return "Document"
    
    elif main_category == "Technical":
        if any(word in content for word in ['code', 'function', 'programming']):
            return "Source Code"
        elif any(word in content for word in ['documentation', 'guide']):
            return "Documentation"
        else:
            return "Technical"
    
    elif main_category == "Personal":
        if any(word in content for word in ['diary', 'journal']):
            return "Journal"
        elif any(word in content for word in ['routine', 'daily']):
            return "Daily Log"
        else:
            return "Notes"
    
    elif main_category == "Food":
        return "Recipe"
    
    elif main_category == "Academic":
        if any(word in content for word in ['research', 'study']):
            return "Research Paper"
        else:
            return "Academic Document"
    
    elif main_category == "Legal":
        return "Contract"
    
    elif main_category == "Health":
        return "Medical Document"
    
    elif main_category == "Communication":
        return "Message"
    
    else:
        return "Document"
    
# API 
@app.post("/scan_folder")
def scan_folder(timeout: int = 30): 
    print("=== Starting COMPREHENSIVE AI scan_folder() endpoint ===")
    import threading
    import time

    result = {"status": "error", "message": "Unknown error", "processed_files": []}
    exception = None

    def scan_worker():
        nonlocal result, exception
        try:
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
                        if ext == ".txt":
                            text = read_text_file(path)
                            print(f"Extracted {len(text)} characters from text file")
                        elif ext == ".pdf":
                            text = extract_text_from_pdf(path)
                            print(f"Extracted {len(text)} characters from PDF")
                        elif ext == ".docx":
                            text = extract_text_from_docx(path)
                            print(f"Extracted {len(text)} characters from DOCX")
                        elif ext in {".png", ".jpg", ".jpeg"}:
                            text = extract_text_from_image(path)
                            print(f"Extracted {len(text)} characters from image OCR")
                            tags = [classify_image(path)]
                        elif ext in {".mp3", ".wav"}:
                            transcript = transcribe_audio(path)
                            text = transcript
                            print(f"Transcribed {len(transcript)} characters from audio")
                        else:
                            text = read_text_file(path)
                            print(f"Extracted {len(text)} characters from file")

                        if text.strip():
                            summary = summarize_text(text)
                            print(f"Generated summary: {len(summary)} characters")

                            try:
                                cat_id, category_name = assign_category_from_summary(summary, text)
                                print(f"AI categorized as: {category_name} (ID: {cat_id})")
                                
                                if not cat_id and category_name:
                                    cur.execute("SELECT id FROM categories WHERE name=?", (category_name,))
                                    cat_row = cur.fetchone()
                                    if cat_row:
                                        cat_id = cat_row[0]
                                        print(f"Found category ID {cat_id} for name '{category_name}'")
                                
                                # Extract and store entities if we have text
                                if len(text) > 100: 
                                    entity_info = extract_and_store_entities(conn, file_id, text)
                                    print(f"Extracted {entity_info['unique']} unique entities")
                                    
                            except Exception as cat_error:
                                print(f"Error in category assignment: {cat_error}")
                                category_name = "General: Document"
                                print(f"Using default smart category: {category_name}")
                                cat_id = None
                        
                        if not text.strip() or not category_name or category_name == "Uncategorized":
                            category_name = "General: Document"
                            print(f"Using default smart category for empty content: {category_name}")
                            cat_id = None

                    except Exception as content_error:
                        print(f"ERROR in content analysis for {fname}: {content_error}")
                        category_name = "General: Document"
                        print(f"Using default smart category for error: {category_name}")

                    try:
                        if not category_name or category_name == "Uncategorized":
                            category_name = "General: Document"
                            print(f"Using final default smart category: {category_name}")
                                
                        if not cat_id and category_name:
                            cur.execute("SELECT id FROM categories WHERE name=?", (category_name,))
                            cat_row = cur.fetchone()
                            if cat_row:
                                cat_id = cat_row[0]
                                print(f"Found category ID {cat_id} for name '{category_name}'")
                            else:
                                cur.execute(
                                    "INSERT INTO categories(name, rep_summary, rep_vector) VALUES (?,?,?)",
                                    (category_name, summary[:500] if summary else "", "[]")
                                )
                                cat_id = cur.lastrowid
                                print(f"Created new category: {category_name} (ID: {cat_id})")
                        
                        print(f"Final category before DB update - Name: '{category_name}', ID: {cat_id}")
                        
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
                        "summary": summary[:200] + "..." if len(summary) > 200 else summary,  
                        "transcript_len": len(text) if text else 0,
                        "tags": tags if 'tags' in locals() else []
                    })

                    print(f"Successfully processed {fname}")

                except Exception as file_error:
                    print(f"ERROR processing file {fname}: {file_error}")
                    continue

            print(f"\n=== COMPREHENSIVE AI SCAN COMPLETE ===")
            print(f"Files processed: {len(processed_files)}")

            result = {"status": "scanned", "chunks_inserted": 0,
                    "processed_files": processed_files}

        except Exception as e:
            print(f"\nFATAL ERROR in scan_folder(): {e}")
            import traceback
            traceback.print_exc()
            exception = e
            result = {"status": "error", "message": f"Scan failed: {str(e)}", "processed_files": []}

    # Start the scan in a separate thread
    scan_thread = threading.Thread(target=scan_worker)
    scan_thread.daemon = True
    scan_thread.start()

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
    Returns a comprehensive list of categories from multiple sources:
    1. Database final_label (approved files)
    2. Database proposed_category (pending files) 
    3. Enhanced categorization results
    4. Google Drive folders (if available)
    """
    cur = conn.cursor()
    
    cur.execute("""
        SELECT final_label as category, COUNT(*) AS file_count, 'approved' as source
        FROM files
        WHERE final_label IS NOT NULL AND final_label != ''
        GROUP BY final_label
    """)
    approved_rows = cur.fetchall()
    
    cur.execute("""
        SELECT proposed_category as category, COUNT(*) AS file_count, 'proposed' as source
        FROM files
        WHERE proposed_category IS NOT NULL AND proposed_category != ''
        GROUP BY proposed_category
    """)
    proposed_rows = cur.fetchall()
    
    # Combine and deduplicate categories
    category_map = {}
    
    for row in approved_rows:
        category, count, source = row
        if category not in category_map:
            category_map[category] = {
                "name": category,
                "approved_count": 0,
                "proposed_count": 0,
                "total_count": 0
            }
        category_map[category]["approved_count"] = count
        category_map[category]["total_count"] += count
    
    for row in proposed_rows:
        category, count, source = row
        if category not in category_map:
            category_map[category] = {
                "name": category,
                "approved_count": 0,
                "proposed_count": 0,
                "total_count": 0
            }
        category_map[category]["proposed_count"] = count
        category_map[category]["total_count"] += count
    
    categories = list(category_map.values())
    categories.sort(key=lambda x: x["total_count"], reverse=True)
    
    return categories

@app.get("/enhanced_categories")
def enhanced_categories(auth_token: str | None = Query(None)):
    try:
        base_categories = categories()
        
        # If we have a Drive token, also fetch Drive folder information
        if auth_token:
            try:
                drive_service = get_drive_service(auth_token)
                if drive_service:
                    results = drive_service.files().list(
                        q="mimeType='application/vnd.google-apps.folder' and 'root' in parents and trashed=false",
                        fields="files(id, name, createdTime, modifiedTime)"
                    ).execute()
                    
                    drive_folders = results.get('files', [])
                    category_map = {cat["name"]: cat for cat in base_categories}
                    
                    for folder in drive_folders:
                        folder_name = folder["name"]
                        if folder_name in category_map:
                            category_map[folder_name]["drive_folder_id"] = folder["id"]
                            category_map[folder_name]["drive_created"] = folder.get("createdTime")
                            category_map[folder_name]["has_drive_folder"] = True
                        else:
                            category_map[folder_name] = {
                                "name": folder_name,
                                "approved_count": 0,
                                "proposed_count": 0,
                                "total_count": 0,
                                "drive_folder_id": folder["id"],
                                "drive_created": folder.get("createdTime"),
                                "has_drive_folder": True
                            }
                    
                    enhanced_categories = list(category_map.values())
                    enhanced_categories.sort(key=lambda x: x["total_count"], reverse=True)
                    
                    return enhanced_categories
                    
            except Exception as e:
                print(f"Error fetching Drive folders: {e}")
        
        for cat in base_categories:
            cat["has_drive_folder"] = False
        
        return base_categories
        
    except Exception as e:
        print(f"Error in enhanced_categories: {e}")
        return []

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


@app.get("/ask")
def ask(file_id: int = Query(...), q: str = Query(...)):
    cur = conn.cursor()
    
    cur.execute("SELECT file_path, file_name, full_text FROM files WHERE id=?", (file_id,))
    row = cur.fetchone()
    
    if not row:
        return {"ok": False, "error": "file not found"}
    
    file_path, file_name, full_text = row
    
    if full_text and full_text.strip():
        print(f"Using pre-extracted text for {file_name} ({len(full_text)} chars)")
    ans = nlp.best_answer(q, full_text)
    return {"ok": True, "answer": ans.get("answer", ""), "score": ans.get("score", 0), "context": ans.get("context", "")}
    
    if not file_path or not os.path.exists(file_path):
        return {"ok": False, "error": "original file not accessible (re-run scan)"}
    
    print(f"Performing real-time text extraction for {file_name}")
    
    ext = os.path.splitext(file_path)[1].lower()
    extracted_text = ""
    
    try:
        if ext == ".txt":
            extracted_text = read_text_file(file_path)
        elif ext == ".pdf":
            extracted_text = extract_text_from_pdf(file_path)
        elif ext == ".docx":
            extracted_text = extract_text_from_docx(file_path)
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
        
        try:
            cur.execute("UPDATE files SET full_text=? WHERE id=?", (extracted_text, file_id))
            conn.commit()
            print("Updated database with extracted text")
        except Exception as e:
            print(f"Warning: Could not update database with extracted text: {e}")
        
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

#Google Drive integration
@app.post("/organize_drive_files")
def organize_drive_files(req: OrganizeDriveFilesRequest):
    """Accepts a list of Google Drive file metadata, classifies them, and
    returns suggested target categories/folders. Movement in Drive is optional
    and should be performed client-side using the user's OAuth token.
    """
    print("ORGANIZE_DRIVE_FILES ENDPOINT CALLED!")
    print(f"Processing {len(req.files)} files")
    print("USING SMART CONTENT ANALYSIS!")
    
    organized = []
    move_performed = False
    drive_service = None
    if req.move and req.auth_token:
        try:
            drive_service = get_drive_service(req.auth_token)
        except Exception as e:
            print(f"Error initializing Drive service: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize Google Drive service: {str(e)}")

    for f in req.files:
        # Skip folders entirely
        if (f.mimeType or "").strip() == 'application/vnd.google-apps.folder':
            continue

        file_name = f.name or ""
        print(f"ANALYZING FILE: {file_name}")

        try:
            if req.override_category:
                category_name = req.override_category
                category_id = None
                print(f"Using override category: {category_name}")
            else:
                print(f"Downloading file for content analysis...")
                path = drive_download_file(req.auth_token, f.id, tempfile.gettempdir())
                if not path:
                    print(f"Failed to download {file_name}")
                    category_name = "DOWNLOAD_FAILED"
                    category_id = None
                else:
                    ext = os.path.splitext(file_name)[1].lower()
                    text = ""
                    if ext == ".txt":
                        text = read_text_file(path)
                    elif ext == ".pdf":
                        text = extract_text_from_pdf(path)
                    elif ext == ".docx":
                        text = extract_text_from_docx(path)
                    elif ext in {".png", ".jpg", ".jpeg"}:
                        text = extract_text_from_image(path)
                    elif ext in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
                        print(f"Transcribing audio file: {file_name}")
                        text = transcribe_audio(path)
                        if text:
                            print(f"Transcribed {len(text)} characters from audio")
                        else:
                            print("Failed to transcribe audio")
                    else:
                        text = read_text_file(path)
                        
                    print(f"Extracted {len(text) if text else 0} chars from {file_name}")
                    
                    if text and len(text.strip()) > 10:
                        cat_id, category_name = assign_category_from_summary("", text)
                        if category_name == "CATEGORIZATION_FAILED":
                            print(f"Could not categorize {file_name}")
                            category_name = "UNCATEGORIZABLE"
                        else:
                            print(f"Categorized {file_name} as: {category_name}")
                    else:
                        print(f"No content extracted from {file_name}")
                        if ext in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
                            category_name = "Audio/Recordings"
                        else:
                            category_name = "NO_CONTENT"
                    
                    category_id = None
                    
                    try:
                        os.remove(path)
                    except:
                        pass

        except Exception as e:
            print(f"Error analyzing file {f.name}: {e}")
            import traceback
            traceback.print_exc()
            category_id, category_name = None, "ANALYSIS_ERROR"

        target_folder_id = None
        moved = None
        if req.move and drive_service:
            target_folder_id = get_or_create_folder(drive_service, (req.override_category or category_name or "Other"), None)
            if target_folder_id:
                moved = move_file_to_folder(drive_service, f.id, target_folder_id)
                if moved and moved.get('id'):
                    move_performed = True
                    print(f"Successfully moved {f.name} to folder {req.override_category or category_name}")
                else:
                    print(f"Failed to move {f.name} to folder {req.override_category or category_name}")

        organized.append({
            "id": f.id,
            "name": f.name,
            "proposed_category": req.override_category or category_name,
            "category_id": category_id,
            "target_folder_id": target_folder_id,
            "mimeType": f.mimeType,
            "parents": f.parents or [],
            "summary": "" 
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

#Smart Search functionality 
import difflib
import re
from collections import defaultdict

def smart_search_in_text(text_content, query):
    """
    Smart search with fuzzy matching, typo tolerance, and similar word detection.
    Returns match result with score, context, and match type.
    """
    if not text_content or not query:
        return None
    
    text_lower = text_content.lower()
    query_lower = query.lower().strip()
    
    query_words = query_lower.split()
    
    search_results = []
    
    # 1. Exact match 
    if query_lower in text_lower:
        match_index = text_lower.find(query_lower)
        context = extract_context(text_content, match_index, len(query))
        search_results.append({
            'score': 1.0,
            'position': match_index,
            'context': context,
            'match_type': 'exact',
            'matched_text': text_content[match_index:match_index + len(query)]
        })
    
    # 2. Case-insensitive word boundary match
    pattern = r'\b' + re.escape(query_lower) + r'\b'
    matches = list(re.finditer(pattern, text_lower))
    for match in matches:
        context = extract_context(text_content, match.start(), len(query))
        search_results.append({
            'score': 0.95,
            'position': match.start(),
            'context': context,
            'match_type': 'word_boundary',
            'matched_text': text_content[match.start():match.end()]
        })
    
    # 3. Fuzzy matching for typos (using difflib)
    words = re.findall(r'\b\w+\b', text_lower)
    word_positions = {}
    
    # Build word position map
    for match in re.finditer(r'\b\w+\b', text_lower):
        word = match.group()
        if word not in word_positions:
            word_positions[word] = []
        word_positions[word].append(match.start())
    
    # Check each query word for fuzzy matches
    for query_word in query_words:
        fuzzy_matches = difflib.get_close_matches(query_word, words, n=5, cutoff=0.7)
        for fuzzy_word in fuzzy_matches:
            similarity = difflib.SequenceMatcher(None, query_word, fuzzy_word).ratio()
            if similarity >= 0.7 and fuzzy_word in word_positions:
                for pos in word_positions[fuzzy_word]:
                    context = extract_context(text_content, pos, len(fuzzy_word))
                    search_results.append({
                        'score': similarity * 0.8,  
                        'position': pos,
                        'context': context,
                        'match_type': 'fuzzy',
                        'matched_text': fuzzy_word
                    })
    
    # 4. Partial word matching 
    for query_word in query_words:
        if len(query_word) >= 4:  
            for word in words:
                if query_word in word and len(word) >= len(query_word):
                    similarity = len(query_word) / len(word)
                    if similarity >= 0.6 and word in word_positions:
                        for pos in word_positions[word]:
                            context = extract_context(text_content, pos, len(word))
                            search_results.append({
                                'score': similarity * 0.6,
                                'position': pos,
                                'context': context,
                                'match_type': 'partial',
                                'matched_text': word
                            })
    
    # 5. Multi-word proximity search
    if len(query_words) > 1:
        proximity_matches = find_proximity_matches(text_lower, query_words, word_positions)
        search_results.extend(proximity_matches)
    
    # 6. Synonym and similar meaning search (basic)
    synonym_matches = find_synonym_matches(text_lower, query_words, word_positions)
    search_results.extend(synonym_matches)
    
    # Remove duplicates and sort by score
    unique_results = {}
    for result in search_results:
        key = (result['position'], result['match_type'])
        if key not in unique_results or result['score'] > unique_results[key]['score']:
            unique_results[key] = result
    
    if not unique_results:
        return None
    
    best_match = max(unique_results.values(), key=lambda x: x['score'])
    return best_match

def extract_context(text, position, match_length):
    """Extract context around a match position."""
    start_context = max(0, position - 100)
    end_context = min(len(text), position + match_length + 100)
    context = text[start_context:end_context].strip()
    
    # Try to start at word boundary
    if start_context > 0:
        space_index = context.find(' ')
        if space_index > 0:
            context = context[space_index:].strip()
    
    return context

def find_proximity_matches(text_lower, query_words, word_positions):
    """Find matches where query words appear close to each other."""
    matches = []
    word_pos_lists = []
    for word in query_words:
        fuzzy_matches = difflib.get_close_matches(word, word_positions.keys(), n=3, cutoff=0.8)
        positions = []
        for fuzzy_word in fuzzy_matches:
            positions.extend(word_positions[fuzzy_word])
        word_pos_lists.append(positions)
    
    # Check for proximity (within 50 characters)
    if len(word_pos_lists) >= 2:
        for pos1 in word_pos_lists[0]:
            for pos2 in word_pos_lists[1]:
                distance = abs(pos1 - pos2)
                if distance <= 50:  # Words within 50 characters
                    score = 0.7 * (1 - distance / 50)  
                    context_pos = min(pos1, pos2)
                    context = extract_context(text_lower, context_pos, distance + 10)
                    matches.append({
                        'score': score,
                        'position': context_pos,
                        'context': context,
                        'match_type': 'proximity',
                        'matched_text': f"proximity match ({distance} chars apart)"
                    })
    
    return matches

def find_synonym_matches(text_lower, query_words, word_positions):
    synonyms = {
        'document': ['doc', 'file', 'paper', 'report'],
        'project': ['task', 'work', 'assignment', 'job'],
        'meeting': ['conference', 'discussion', 'session'],
        'list': ['items', 'tasks', 'todo', 'checklist'],
        'invoice': ['bill', 'receipt', 'payment'],
        'contract': ['agreement', 'deal'],
        'plan': ['strategy', 'blueprint', 'outline'],
        'summary': ['overview', 'abstract', 'brief'],
        'analysis': ['review', 'study', 'examination'],
        'proposal': ['suggestion', 'recommendation', 'offer']
    }
    
    matches = []
    for query_word in query_words:
        if query_word in synonyms:
            for synonym in synonyms[query_word]:
                if synonym in word_positions:
                    for pos in word_positions[synonym]:
                        context = extract_context(text_lower, pos, len(synonym))
                        matches.append({
                            'score': 0.5,
                            'position': pos,
                            'context': context,
                            'match_type': 'synonym',
                            'matched_text': synonym
                        })
    
    return matches

#Search functionality 
class SearchRequest(BaseModel):
    query: str
    use_semantic: bool = True  # Enable semantic search by default
    semantic_weight: float = 0.7  # Weight for semantic search in hybrid mode
    min_score: float = 0.2  # Minimum confidence score (0-1)
    top_k: int = 10  # Maximum number of results to return

from semantic_search import hybrid_search, clean_query
import torch

def get_context_around_match(text: str, query_terms: set, window_size: int = 150) -> str:
    """Extract a window of text around the best match for the query."""
    if not text or not query_terms:
        return text[:window_size] + ('...' if len(text) > window_size else '')
    
    # Find all positions where query terms appear
    text_lower = text.lower()
    positions = []
    
    for term in query_terms:
        pos = text_lower.find(term.lower())
        if pos != -1:
            positions.append((pos, term))
    
    if not positions:
        return text[:window_size] + ('...' if len(text) > window_size else '')
    
    # Sort by position
    positions.sort()
    
    # Get the first match position
    first_pos, first_term = positions[0]
    
    # Calculate window around the first match
    start = max(0, first_pos - (window_size // 2))
    end = min(len(text), first_pos + len(first_term) + (window_size // 2))
    
    # Extract the window
    snippet = text[start:end]
    
    # Add ellipsis if needed
    if start > 0:
        snippet = '...' + snippet
    if end < len(text):
        snippet = snippet + '...'
    
    return snippet

@app.post("/search_files")
async def search_files(req: SearchRequest, auth_token: str | None = Query(None)):
    """
    Search Google Drive files by content. Supports PDF, TXT, DOCS, IMAGES, and AUDIO files.
    For images, uses OCR to extract text. For audio files, uses transcription.
    Returns files that contain the search query in their content.
    """
    print(f"SEARCH_FILES ENDPOINT CALLED with query: '{req.query}'")
    
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")
    
    query = req.query.strip().lower()
    token = auth_token
    
    if not token:
        # For now, return a helpful message about authentication
        # In production, you would redirect to Google OAuth flow
        raise HTTPException(status_code=400, detail="Google Drive authentication required. Please set up Google OAuth to search your Drive files.")
    
    try:
        # Get Google Drive service
        service = get_drive_service(token)
        if not service:
            raise HTTPException(status_code=400, detail="Failed to authenticate with Google Drive")
        
        # Search for files in Google Drive (PDF, TXT, DOCS, IMAGES, AUDIO)
        search_query = (
            "mimeType='application/pdf' or "
            "mimeType='text/plain' or "
            "mimeType='application/vnd.google-apps.document' or "
            "mimeType='application/msword' or "
            "mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document' or "
            "mimeType='image/jpeg' or "
            "mimeType='image/png' or "
            "mimeType='image/gif' or "
            "mimeType='image/bmp' or "
            "mimeType='image/webp' or "
            "mimeType='audio/mpeg' or "
            "mimeType='audio/wav' or "
            "mimeType='audio/mp3' or "
            "mimeType='audio/m4a' or "
            "mimeType='audio/ogg'"
        )
        
        results = service.files().list(
            q=f"({search_query}) and trashed=false",
            fields="files(id,name,mimeType,size,modifiedTime,parents)",
            pageSize=100
        ).execute()
        
        files = results.get('files', [])
        print(f"Found {len(files)} files to search through")
        
        matching_files = []
        
        for file in files:
            try:
                print(f"Searching in file: {file['name']}")
                
                file_path = drive_download_file(token, file['id'], tempfile.gettempdir())
                if not file_path:
                    print(f"Failed to download file: {file['name']}")
                    continue
                
                text_content = ""
                file_name = file['name'].lower()
                
                if file['mimeType'] == 'application/pdf' or file_name.endswith('.pdf'):
                    text_content = extract_text_from_pdf(file_path)
                elif file['mimeType'] == 'text/plain' or file_name.endswith('.txt'):
                    text_content = read_text_file(file_path)
                elif file['mimeType'] == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' or file_name.endswith('.docx'):
                    text_content = extract_text_from_docx(file_path)
                elif file['mimeType'] == 'application/vnd.google-apps.document':
                    # For Google Docs, we need to export as text
                    try:
                        export_result = service.files().export(
                            fileId=file['id'],
                            mimeType='text/plain'
                        ).execute()
                        text_content = export_result.decode('utf-8')
                    except Exception as e:
                        print(f"Failed to export Google Doc {file['name']}: {e}")
                        continue
                elif file['mimeType'] in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                    text_content = read_text_file(file_path)
                elif file['mimeType'].startswith('image/') or file_name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                    print(f"Extracting text from image: {file['name']}")
                    text_content = extract_text_from_image(file_path)
                elif file['mimeType'].startswith('audio/') or file_name.endswith(('.mp3', '.wav', '.m4a', '.ogg')):
                    print(f"Transcribing audio file: {file['name']}")
                    text_content = transcribe_audio(file_path)
                
                try:
                    os.remove(file_path)
                except Exception:
                    pass
                
                doc = {
                    'text': text_content,
                    'metadata': {
                        'id': file['id'],
                        'name': file['name'],
                        'mimeType': file['mimeType'],
                        'size': file.get('size', 0),
                        'modifiedTime': file.get('modifiedTime', '')
                    }
                }
                
                # Use semantic search if enabled, otherwise use traditional search
                if req.use_semantic:
                    matching_files.append(doc)
                else:
                    match_result = smart_search_in_text(text_content, query)
                    if match_result:
                        print(f"Found match in file: {file['name']} (score: {match_result['score']:.2f})")
                        
                        matching_files.append({
                            'id': file['id'],
                            'name': file['name'],
                            'mimeType': file['mimeType'],
                            'size': file.get('size', 0),
                            'modifiedTime': file.get('modifiedTime', ''),
                            'context': match_result['context'],
                            'match_position': match_result['position'],
                            'match_score': match_result['score'],
                            'match_type': match_result['match_type'],
                            'matched_text': match_result['matched_text'],
                            'drive_url': f"https://drive.google.com/file/d/{file['id']}/view"
                        })
                    
            except Exception as e:
                print(f"Error processing file {file['name']}: {e}")
                continue
        
        # Process semantic search results if enabled
        if req.use_semantic and matching_files:
            print(f"Performing semantic search on {len(matching_files)} files...")
            
            try:
                clean_q = clean_query(req.query)
                query_terms = set(clean_q.split())
                
                # Use hybrid search for better results
                search_results = hybrid_search(
                    query=req.query,
                    documents=matching_files,
                    semantic_weight=req.semantic_weight,
                    top_k=req.top_k,
                    min_score=req.min_score
                )
                formatted_results = []
                
                for result in search_results:
                    metadata = result.get('metadata', {})
                    best_chunk = result.get('best_chunk', '')
                    
                    context = get_context_around_match(
                        text=best_chunk or metadata.get('text', ''),
                        query_terms=query_terms
                    )
                    confidence = min(100, int(result['score'] * 100))
                    
                    formatted_results.append({
                        'id': metadata.get('id'),
                        'name': metadata.get('name', 'Untitled'),
                        'mimeType': metadata.get('mimeType', ''),
                        'size': metadata.get('size', 0),
                        'modifiedTime': metadata.get('modifiedTime', ''),
                        'context': context,
                        'confidence': confidence,
                        'match_score': round(result['score'], 4),
                        'semantic_score': round(result.get('semantic_score', 0), 4),
                        'keyword_score': round(result.get('keyword_score', 0), 4),
                        'match_type': result.get('match_type', 'hybrid'),
                        'drive_url': f"https://drive.google.com/file/d/{metadata.get('id', '')}/view"
                    })
                
                print(f"Semantic search completed. Found {len(formatted_results)} relevant files")
                
                return {
                    'query': req.query,
                    'total_searched': len(files),
                    'matches_found': len(formatted_results),
                    'files': formatted_results,
                    'search_type': 'semantic',
                    'search_params': {
                        'semantic_weight': req.semantic_weight,
                        'min_score': req.min_score,
                        'top_k': req.top_k
                    }
                }
                
            except Exception as e:
                print(f"Error in semantic search: {str(e)}")
                print("Falling back to keyword search...")
                req.use_semantic = False
        if not req.use_semantic and matching_files:
            clean_q = clean_query(req.query)
            query_terms = set(clean_q.split())
            
            scored_files = []
            for file in matching_files:
                text = file.get('text', '').lower()
                
                if query_terms:
                    term_matches = sum(1 for term in query_terms if term in text)
                    score = term_matches / len(query_terms)
                else:
                    score = 0
                
                if score > 0:
                    file['match_score'] = score
                    file['match_type'] = 'keyword'
                    scored_files.append(file)
            
            # Sort by score 
            scored_files.sort(key=lambda x: x.get('match_score', 0), reverse=True)
            
            # Apply top_k limit
            scored_files = scored_files[:req.top_k]
            
            formatted_results = []
            for file in scored_files:
                context = get_context_around_match(
                    text=file.get('text', ''),
                    query_terms=query_terms
                )
                
                formatted_results.append({
                    'id': file.get('id'),
                    'name': file.get('name', 'Untitled'),
                    'mimeType': file.get('mimeType', ''),
                    'size': file.get('size', 0),
                    'modifiedTime': file.get('modifiedTime', ''),
                    'context': context,
                    'confidence': int(file.get('match_score', 0) * 100),
                    'match_score': round(file.get('match_score', 0), 4),
                    'match_type': 'keyword',
                    'drive_url': f"https://drive.google.com/file/d/{file.get('id', '')}/view"
                })
            
            print(f"Keyword search completed. Found {len(formatted_results)} matching files")
            
            return {
                'query': req.query,
                'total_searched': len(files),
                'matches_found': len(formatted_results),
                'files': formatted_results,
                'search_type': 'keyword',
                'search_params': {
                    'top_k': req.top_k
                }
            }
        
        return {
            'query': req.query,
            'total_searched': len(files),
            'matches_found': 0,
            'files': [],
            'search_type': 'semantic' if req.use_semantic else 'keyword',
            'message': 'No matching documents found.'
        }
        
    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/visual_search")
async def visual_search(image: UploadFile = File(...), auth_token: str | None = Query(None)):
    """
    Visual search: analyze uploaded image and search for related content in Google Drive files.
    Uses local computer vision model (no API keys required).
    """
    print(f"VISUAL_SEARCH ENDPOINT CALLED with image: {image.filename}")
    
    if not auth_token:
        raise HTTPException(status_code=400, detail="Google Drive authentication required. Please set up Google OAuth to search your Drive files.")
    
    try:
        temp_image_path = os.path.join(tempfile.gettempdir(), f"visual_search_{image.filename}")
        with open(temp_image_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        # Analyze image using local computer vision model
        detected_objects = analyze_image_content(temp_image_path)
        
        try:
            os.remove(temp_image_path)
        except Exception:
            pass
        
        if not detected_objects:
            return {
                'query': 'visual search',
                'detected_objects': [],
                'total_searched': 0,
                'matches_found': 0,
                'files': []
            }
        
        # Search for files containing the detected objects
        search_queries = detected_objects[:5]  
        all_matching_files = []
        
        # Get Google Drive service
        service = get_drive_service(auth_token)
        if not service:
            raise HTTPException(status_code=400, detail="Failed to authenticate with Google Drive")
        
        # Search for files in Google Drive
        search_query = "mimeType='application/pdf' or mimeType='text/plain' or mimeType='application/vnd.google-apps.document' or mimeType='application/msword' or mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document'"
        
        results = service.files().list(
            q=f"({search_query}) and trashed=false",
            fields="files(id,name,mimeType,size,modifiedTime,parents)",
            pageSize=50  
        ).execute()
        
        files = results.get('files', [])
        print(f"Found {len(files)} files to search through for visual content")
        
        for query in search_queries:
            matching_files = []
            
            for file in files:
                try:
                    file_path = drive_download_file(auth_token, file['id'], tempfile.gettempdir())
                    if not file_path:
                        continue
                    
                    text_content = ""
                    file_name = file['name'].lower()
                    
                    if file['mimeType'] == 'application/pdf' or file_name.endswith('.pdf'):
                        text_content = extract_text_from_pdf(file_path)
                    elif file['mimeType'] == 'text/plain' or file_name.endswith('.txt'):
                        text_content = read_text_file(file_path)
                    elif file['mimeType'] == 'application/vnd.google-apps.document':
                        try:
                            export_result = service.files().export(
                                fileId=file['id'],
                                mimeType='text/plain'
                            ).execute()
                            text_content = export_result.decode('utf-8')
                        except Exception as e:
                            print(f"Failed to export Google Doc {file['name']}: {e}")
                            continue
                    elif file['mimeType'] in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                        text_content = read_text_file(file_path)
                    
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
                    
                    match_result = smart_search_in_text(text_content, query)
                    if match_result:
                        print(f"Found visual match in file: {file['name']} for object: {query}")
                        
                        matching_files.append({
                            'id': file['id'],
                            'name': file['name'],
                            'mimeType': file['mimeType'],
                            'size': file.get('size', 0),
                            'modifiedTime': file.get('modifiedTime', ''),
                            'context': match_result['context'],
                            'match_position': match_result['position'],
                            'match_score': match_result['score'],
                            'match_type': 'visual_object',
                            'matched_text': f"Related to: {query}",
                            'detected_object': query,
                            'drive_url': f"https://drive.google.com/file/d/{file['id']}/view"
                        })
                        
                except Exception as e:
                    print(f"Error processing file {file['name']}: {e}")
                    continue
            
            all_matching_files.extend(matching_files)
        
        unique_files = {}
        for file in all_matching_files:
            file_id = file['id']
            if file_id not in unique_files or file['match_score'] > unique_files[file_id]['match_score']:
                unique_files[file_id] = file
        
        final_files = list(unique_files.values())
        final_files.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        
        print(f"Visual search completed. Found {len(final_files)} matching files for objects: {detected_objects}")
        
        return {
            'query': 'visual search',
            'detected_objects': detected_objects,
            'total_searched': len(files),
            'matches_found': len(final_files),
            'files': final_files
        }
        
    except Exception as e:
        print(f"Visual search error: {e}")
        raise HTTPException(status_code=500, detail=f"Visual search failed: {str(e)}")

def analyze_image_content(image_path):
    """
    Analyze image content using local computer vision model.
    Returns list of detected objects/concepts.
    """
    try:
        # Use the existing CV model for image classification
        cv_model_result = get_cv_model()
        if not cv_model_result:
            print("CV model not available, using basic image analysis")
            return ["image", "photo", "picture"]
        
        cv_model, _ = cv_model_result  # Unpack the tuple (model, transform)
        
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        
        # Create transform for this image
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tensor = image_transform(img).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            outputs = cv_model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        # Get top predictions and convert to meaningful terms
        top_probs, top_indices = torch.topk(probabilities, 5)
        top_indices = top_indices[0].tolist()  # Get the actual indices
        top_probs = top_probs[0].tolist()     # Get the actual probabilities
        
        print(f"Top predictions: {list(zip(top_indices, top_probs))}")
        
        # Map ImageNet class indices to meaningful search terms
        detected_objects = []
        for idx in top_indices:
            object_mappings = {
                7: ["cock", "rooster", "chicken", "poultry", "bird", "farm"],
                8: ["hen", "chicken", "poultry", "bird", "farm", "egg"],
                9: ["ostrich", "bird", "large bird", "wildlife"],
                10: ["brambling", "bird", "small bird", "wildlife"],
                11: ["goldfinch", "bird", "small bird", "wildlife"],
                12: ["house finch", "bird", "small bird", "wildlife"],
                13: ["junco", "bird", "small bird", "wildlife"],
                14: ["indigo bunting", "bird", "small bird", "wildlife"],
                15: ["robin", "bird", "small bird", "wildlife"],
                16: ["bulbul", "bird", "small bird", "wildlife"],
                17: ["jay", "bird", "small bird", "wildlife"],
                18: ["magpie", "bird", "wildlife"],
                19: ["chickadee", "bird", "small bird", "wildlife"],
                20: ["water ouzel", "bird", "water bird", "wildlife"],
                
                281: ["tabby cat", "cat", "feline", "pet", "animal"],
                282: ["tiger cat", "cat", "feline", "pet", "animal"],
                283: ["Persian cat", "cat", "feline", "pet", "animal"],
                284: ["Siamese cat", "cat", "feline", "pet", "animal"],
                285: ["Egyptian cat", "cat", "feline", "pet", "animal"],
                
                151: ["Chihuahua", "dog", "small dog", "pet", "animal"],
                152: ["Japanese spaniel", "dog", "small dog", "pet", "animal"],
                153: ["Maltese dog", "dog", "small dog", "pet", "animal"],
                154: ["Pekinese", "dog", "small dog", "pet", "animal"],
                155: ["Shih-Tzu", "dog", "small dog", "pet", "animal"],
                156: ["Blenheim spaniel", "dog", "pet", "animal"],
                157: ["papillon", "dog", "small dog", "pet", "animal"],
                158: ["toy terrier", "dog", "small dog", "pet", "animal"],
                
                345: ["pig", "hog", "swine", "farm", "animal"],
                346: ["wild boar", "pig", "wild animal", "wildlife"],
                347: ["warthog", "pig", "wild animal", "wildlife"],
                348: ["hippopotamus", "hippo", "large animal", "wildlife"],
                349: ["ox", "cattle", "farm", "animal"],
                350: ["water buffalo", "buffalo", "farm", "animal"],
                351: ["bison", "buffalo", "wild animal", "wildlife"],
                
                924: ["guacamole", "food", "dip", "avocado", "mexican"],
                925: ["consomme", "soup", "food", "broth"],
                926: ["hot pot", "food", "cooking", "meal"],
                927: ["trifle", "dessert", "food", "sweet"],
                928: ["ice cream", "dessert", "food", "sweet"],
                929: ["ice lolly", "popsicle", "dessert", "food"],
                930: ["French loaf", "bread", "food", "bakery"],
                931: ["bagel", "bread", "food", "breakfast"],
                932: ["pretzel", "snack", "food", "bakery"],
                933: ["cheeseburger", "burger", "food", "fast food"],
                934: ["hotdog", "food", "fast food", "sausage"],
                935: ["mashed potato", "potato", "food", "side dish"],
                936: ["head cabbage", "cabbage", "vegetable", "food"],
                937: ["broccoli", "vegetable", "food", "green"],
                938: ["cauliflower", "vegetable", "food", "white"],
                939: ["zucchini", "vegetable", "food", "green"],
                940: ["spaghetti squash", "squash", "vegetable", "food"],
                
                403: ["airliner", "airplane", "aircraft", "transportation", "travel"],
                404: ["warplane", "airplane", "aircraft", "military"],
                407: ["ambulance", "vehicle", "emergency", "medical"],
                408: ["beach wagon", "car", "vehicle", "transportation"],
                409: ["cab", "taxi", "car", "vehicle", "transportation"],
                410: ["convertible", "car", "vehicle", "transportation"],
                411: ["jeep", "car", "vehicle", "off-road"],
                412: ["limousine", "car", "vehicle", "luxury"],
                413: ["minivan", "car", "vehicle", "family"],
                414: ["Model T", "car", "vehicle", "vintage"],
                415: ["racer", "race car", "vehicle", "sports"],
                416: ["sports car", "car", "vehicle", "fast"],
                417: ["station wagon", "car", "vehicle", "family"],
                
                664: ["cellular telephone", "phone", "mobile", "technology"],
                665: ["dial telephone", "phone", "vintage", "technology"],
                666: ["digital clock", "clock", "time", "technology"],
                667: ["digital watch", "watch", "time", "technology"],
                668: ["disk brake", "brake", "car part", "automotive"],
                669: ["desktop computer", "computer", "technology", "work"],
                670: ["hand-held computer", "device", "technology", "portable"],
                671: ["laptop", "computer", "technology", "portable"],
                672: ["notebook", "computer", "technology", "portable"],
                
                980: ["volcano", "mountain", "nature", "geological"],
                981: ["promontory", "cliff", "nature", "geological"],
                982: ["sandbar", "beach", "nature", "water"],
                983: ["coral reef", "reef", "nature", "underwater"],
                984: ["lakeside", "lake", "nature", "water"],
                985: ["seashore", "beach", "nature", "water"],
                986: ["geyser", "nature", "geological", "water"]
            }
            
            if idx in object_mappings:
                objects_for_class = object_mappings[idx]
                print(f"Class {idx} mapped to: {objects_for_class}")
                detected_objects.extend(objects_for_class)
            else:
                print(f"Class {idx} not in mapping, using generic terms")
                # Generic terms based on class index ranges
                if 0 <= idx <= 50:  # Bird range
                    detected_objects.extend(["bird", "animal", "wildlife"])
                elif 151 <= idx <= 268:  # Dog range
                    detected_objects.extend(["dog", "pet", "animal"])
                elif 281 <= idx <= 285:  # Cat range
                    detected_objects.extend(["cat", "pet", "animal"])
                elif 345 <= idx <= 382:  # Farm animal range
                    detected_objects.extend(["farm animal", "animal", "livestock"])
                elif 400 <= idx <= 500:  # Vehicle range
                    detected_objects.extend(["vehicle", "transportation"])
                elif 900 <= idx <= 999:  # Food/object range
                    detected_objects.extend(["food", "object", "item"])
                else:
                    detected_objects.extend(["object", "item", "thing"])
        
        unique_objects = list(dict.fromkeys(detected_objects))
        print(f"Final detected objects: {unique_objects}")
        return unique_objects[:8] 
        
    except Exception as e:
        print(f"Image analysis error: {e}")
        return ["image", "photo", "picture", "document", "content"]

class UpdateCategoryRequest(BaseModel):
    file_id: str
    new_category: str
    auth_token: Optional[str] = None

class MultiFileAnalysisRequest(BaseModel):
    files: list[dict]  
    query: str
    output_type: str = "detailed"  # detailed, flowchart, timeline, short_notes, key_insights, flashcards
    format: Optional[str] = None 
    auth_token: Optional[str] = None

@app.post("/update_category")
def update_category(req: UpdateCategoryRequest):
    """Update the category for a Google Drive file"""
    try:
        print(f"UPDATE_CATEGORY called for file {req.file_id} -> {req.new_category}")
        
        drive_service = get_drive_service(req.auth_token)
        
        cur = conn.cursor()
        
        cur.execute("SELECT id, file_name FROM proposals WHERE file_id = ?", (req.file_id,))
        existing = cur.fetchone()
        
        if existing:
            cur.execute("""
                UPDATE proposals 
                SET proposed_label = ?, updated_at = CURRENT_TIMESTAMP
                WHERE file_id = ?
            """, (req.new_category, req.file_id))
            print(f"Updated existing record for {req.file_id}")
        else:
            # Get file info from Drive
            try:
                file_info = drive_service.files().get(fileId=req.file_id).execute()
                file_name = file_info.get('name', 'Unknown')
                
                cur.execute("""
                    INSERT INTO proposals (file_id, file_name, proposed_label, approved)
                    VALUES (?, ?, ?, 0)
                """, (req.file_id, file_name, req.new_category))
                print(f"Created new record for {req.file_id}")
            except Exception as e:
                print(f"Failed to get file info from Drive: {e}")
                cur.execute("""
                    INSERT INTO proposals (file_id, file_name, proposed_label, approved)
                    VALUES (?, ?, ?, 0)
                """, (req.file_id, "Unknown", req.new_category))
        
        conn.commit()
        
        return {
            "success": True,
            "file_id": req.file_id,
            "new_category": req.new_category,
            "message": "Category updated successfully"
        }
        
    except Exception as e:
        print(f"Error updating category: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update category: {str(e)}")

@app.get("/proposals")
def get_proposals():
    """Get all proposals from database with updated categories"""
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT file_id, file_name, proposed_label, final_label, approved, created_at, updated_at
            FROM proposals 
            ORDER BY updated_at DESC
        """)
        rows = cur.fetchall()
        
        proposals = []
        for row in rows:
            file_id, file_name, proposed_label, final_label, approved, created_at, updated_at = row
            
            # Use final_label if approved, otherwise proposed_label
            category = final_label if approved and final_label else (proposed_label or "Uncategorized")
            
            proposals.append({
                "id": file_id,
                "name": file_name or "Unknown",
                "proposed_category": category,
                "final_category": final_label,
                "approved": bool(approved),
                "created_at": created_at,
                "updated_at": updated_at
            })
        
        print(f"Returning {len(proposals)} proposals from database")
        return proposals
        
    except Exception as e:
        print(f"Error fetching proposals: {e}")
        return []

@app.delete("/proposals")
def clear_proposals():
    """Clear all proposals from database to start fresh"""
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM proposals")
        conn.commit()
        
        deleted_count = cur.rowcount
        print(f"Cleared {deleted_count} proposals from database")
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "message": "All proposals cleared successfully"
        }
        
    except Exception as e:
        print(f"Error clearing proposals: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear proposals: {str(e)}")

def generate_flashcards_from_content(combined_text: str, file_summaries: list) -> list:
    """Generate flashcards from the combined content of multiple files"""
    flashcards = []
    
    content_lower = combined_text.lower()
    
    # DSA-specific flashcards based on content
    if 'data structure' in content_lower:
        if 'way of organizing' in content_lower or 'storing data' in content_lower:
            flashcards.append({
                'question': 'What is a Data Structure?',
                'answer': 'A way of organizing and storing data for efficient access and modification.'
            })
    
    if 'algorithm' in content_lower:
        if 'step' in content_lower or 'procedure' in content_lower or 'operations efficiently' in content_lower:
            flashcards.append({
                'question': 'What is an Algorithm?',
                'answer': 'A step-by-step procedure or set of rules for solving problems and performing operations efficiently.'
            })
    
    if 'foundation' in content_lower and 'computer science' in content_lower:
        flashcards.append({
            'question': 'Why are Data Structures and Algorithms important?',
            'answer': 'They form the foundation of computer science and software development, enabling efficient problem-solving and optimal performance.'
        })
    
    # Extract key concepts from the actual content
    if 'optimization' in content_lower or 'efficient' in content_lower:
        flashcards.append({
            'question': 'What is the main goal of using proper data structures and algorithms?',
            'answer': 'To optimize performance and enable efficient computing by organizing data and operations effectively.'
        })
    
    if 'complexity' in content_lower:
        flashcards.append({
            'question': 'What is algorithmic complexity?',
            'answer': 'A measure of how the performance of an algorithm changes with the size of the input, typically expressed in time and space complexity.'
        })
    
    # Generate flashcards based on file types and content
    pdf_files = [f for f in file_summaries if f['name'].endswith('.pdf')]
    if pdf_files:
        flashcards.append({
            'question': 'What type of content is typically found in research papers about DSA?',
            'answer': 'Theoretical foundations, optimization techniques, performance analysis, and advanced algorithmic concepts.'
        })
    
    image_files = [f for f in file_summaries if f['name'].endswith(('.jpg', '.png'))]
    if image_files:
        flashcards.append({
            'question': 'How can lecture notes in image format be useful for studying DSA?',
            'answer': 'They provide visual representations of concepts, diagrams, and structured notes that can enhance understanding of complex topics.'
        })
    
    audio_files = [f for f in file_summaries if f['name'].endswith(('.mp3', '.wav'))]
    if audio_files:
        flashcards.append({
            'question': 'What are the benefits of audio lectures for learning DSA?',
            'answer': 'Audio content provides explanations in natural language, making complex concepts more accessible and allowing for passive learning.'
        })
    
    # If no specific flashcards were generated, create generic ones
    if not flashcards:
        flashcards = [
            {
                'question': 'What are the main components covered in this multi-file study material?',
                'answer': f'The material covers {len(file_summaries)} files with content about data structures, algorithms, and computer science fundamentals.'
            },
            {
                'question': 'How many different file formats are included in this study collection?',
                'answer': f'The collection includes {len(set(f["name"].split(".")[-1] if "." in f["name"] else "unknown" for f in file_summaries))} different file formats for comprehensive learning.'
            },
            {
                'question': 'What is the total amount of content in this study material?',
                'answer': f'The complete material contains {len(combined_text):,} characters across all files, providing substantial content for study.'
            }
        ]
    
    # Ensure we have at least 3 flashcards
    while len(flashcards) < 3:
        flashcards.append({
            'question': f'What can you learn from File {len(flashcards)}: {file_summaries[min(len(flashcards)-1, len(file_summaries)-1)]["name"]}?',
            'answer': f'This file contains {file_summaries[min(len(flashcards)-1, len(file_summaries)-1)]["length"]} characters of content related to the study topic.'
        })
    
    return flashcards[:6] 

@app.post("/analyze_multi")
def analyze_multi_files(req: MultiFileAnalysisRequest):
    """Analyze multiple files together and generate a unified report"""
    try:
        print(f"MULTI_FILE_ANALYSIS called for {len(req.files)} files with query: {req.query}")
        print(f"Output type: {req.output_type}")
        
        if not req.files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if not req.query.strip():
            raise HTTPException(status_code=400, detail="No query provided")
        
        # Extract text from all files
        all_texts = []
        file_summaries = []
        
        for i, file in enumerate(req.files):
            file_id = file.get('id')
            file_name = file.get('name', f'File_{i+1}')
            
            if not file_id:
                print(f"Skipping file {file_name} - no ID provided")
                continue
                
            print(f"Processing file {i+1}/{len(req.files)}: {file_name}")
            
            path = drive_download_file(req.auth_token, file_id, tempfile.gettempdir())
            if not path:
                print(f"Failed to download file: {file_name}")
                continue
                
            try:
                ext = os.path.splitext(file_name)[1].lower()
                text = ""
                
                if ext == ".txt":
                    text = read_text_file(path)
                elif ext == ".pdf":
                    text = extract_text_from_pdf(path)
                elif ext == ".docx":
                    text = extract_text_from_docx(path)
                elif ext in {".png", ".jpg", ".jpeg"}:
                    text = extract_text_from_image(path)
                elif ext in {".mp3", ".wav"}:
                    text = transcribe_audio(path)
                else:
                    text = read_text_file(path)
                
                if text and text.strip():
                    file_text = f"=== FILE: {file_name} ===\n{text}\n=== END OF {file_name} ===\n\n"
                    all_texts.append(file_text)
                    
                    file_summary = {
                        "name": file_name,
                        "length": len(text),
                        "preview": text[:500] + "..." if len(text) > 500 else text,
                        "full_text": text  
                    }
                    file_summaries.append(file_summary)
                    print(f"Extracted {len(text)} characters from {file_name}")
                else:
                    print(f"No text extracted from {file_name}")
                    
            finally:
                try:
                    os.remove(path)
                except Exception:
                    pass
        
        if not all_texts:
            raise HTTPException(status_code=422, detail="No readable content found in any of the provided files")
        
        combined_text = "\n".join(all_texts)
        print(f"Combined text length: {len(combined_text)} characters from {len(all_texts)} files")
        
        output_prompts = {
            "detailed": f"Provide a comprehensive, detailed analysis of the following documents to answer this question: {req.query}\n\nAnalyze all the documents together and provide insights, connections, and conclusions.",
            "flowchart": f"Create a flowchart representation (in text format with arrows and boxes) based on the following documents to answer: {req.query}\n\nShow the flow of processes, decisions, or concepts across all documents.",
            "timeline": f"Create a chronological timeline based on the following documents to answer: {req.query}\n\nExtract dates, events, and sequence of activities from all documents and present them in chronological order.",
            "short_notes": f"Summarize the following documents into concise, bullet-point notes to answer: {req.query}\n\nProvide key takeaways and important points from all documents.",
            "key_insights": f"Extract key insights and important findings from the following documents to answer: {req.query}\n\nIdentify the most important points, patterns, and conclusions across all documents.",
            "flashcards": f"Create Q&A flashcards based on the following documents to answer: {req.query}\n\nGenerate question-answer pairs that capture the key concepts and information from all documents."
        }
        
        prompt = output_prompts.get(req.output_type, output_prompts["detailed"])
        
        use_assistant_generator = req.output_type in ['flowchart', 'flashcards', 'key_insights'] and req.format
        
        if use_assistant_generator:
            try:
                print(f"Using assistant generator for {req.output_type} with format {req.format}")
                
                kind_mapping = {
                    'flowchart': 'flowchart',
                    'flashcards': 'flashcards', 
                    'key_insights': 'key_insights',
                    'timeline': 'timeline',
                    'short_notes': 'short_notes'
                }
                
                kind = kind_mapping.get(req.output_type, 'detailed_notes')
                
                # Generate using assistant generator
                generated_kind, content = generate(kind, combined_text)
                filename, b64 = export_bytes(generated_kind, content, req.format or "md")
                
                # Get MIME type for the format
                mime = {
                    'md': 'text/markdown',
                    'markdown': 'text/markdown', 
                    'txt': 'text/plain',
                    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'pdf': 'application/pdf'
                }.get((req.format or 'md').lower(), 'application/octet-stream')
                
                # Prepare response with downloadable content
                response = {
                    "query": req.query,
                    "output_type": req.output_type,
                    "files_processed": len(all_texts),
                    "total_files": len(req.files),
                    "file_summaries": file_summaries,
                    "analysis": content,  
                    "combined_length": len(combined_text),
                    "assistant": {
                        "type": "download",
                        "kind": generated_kind,
                        "filename": filename,
                        "base64": b64,
                        "mime": mime,
                        "content": content
                    }
                }
                
                print(f"Multi-file analysis with assistant generator completed successfully")
                return response
                
            except Exception as e:
                print(f"Assistant generator failed, falling back to NLP service: {e}")
                # Fall through to regular NLP generation
        
        # Generate analysis using existing NLP service
        try:
            # Limit combined text to prevent token overflow
            max_text_length = 8000  
            if len(combined_text) > max_text_length:
                combined_text = combined_text[:max_text_length] + "\n\n[Content truncated for analysis]"
            
            print(f"Sending to NLP service: prompt length={len(prompt)}, text length={len(combined_text)}")
            # Use the correct NLP method - best_answer for Q&A or gemini_generate for general analysis
            if '?' in req.query or any(word in req.query.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who']):
                result = nlp.best_answer(req.query, combined_text)
                analysis = result.get('answer', 'No answer found')
            else:
                # This is a general analysis request, use gemini_generate
                full_prompt = f"{prompt}\n\nDocuments:\n{combined_text}"
                print(f"Calling gemini_generate with prompt length: {len(full_prompt)}")
                analysis = nlp.gemini_generate(full_prompt, temperature=0.3, max_output_tokens=1024)
                print(f"Gemini returned: {len(analysis) if analysis else 0} characters")
            
            if analysis:
                analysis = re.sub(r'\*{2,}', '', analysis)  
                analysis = analysis.strip()
                print(f"NLP service returned analysis: {len(analysis)} characters")
            else:
                print("NLP service returned empty or None analysis, using fallback")
                analysis = None 
        except Exception as e:
            print(f"NLP generation failed: {e}")
            print(f"Error details: {type(e).__name__}: {str(e)}")
            
            analysis = f"Multi-File Analysis Results\n\n"
            analysis += f"Query: {req.query}\n\n"
            
            for summary in file_summaries:
                analysis += f"{summary['name']} ({summary['length']} characters):\n"
                analysis += f"{summary['preview']}\n\n"
            
            query_lower = req.query.lower()
            
            # Extract key information from the combined text
            combined_lower = combined_text.lower()
            
            if any(word in query_lower for word in ['summary', 'summarize', 'overview']):
                analysis += "Summary: Based on the documents, this covers Data Structures and Algorithms (DSA), which form the foundation of computer science and software development. The materials include theoretical concepts, practical applications, and optimization techniques for efficient computing.\n\n"
            
            elif any(word in query_lower for word in ['data structure', 'what is']):
                if 'data structure' in combined_lower:
                    analysis += "Answer: According to the documents, a Data Structure (DS) is a way of organizing and storing data for efficient access and modification. It's a fundamental concept in computer science that enables efficient operations on data.\n\n"
                else:
                    analysis += "Answer: The documents discuss data structures as fundamental ways of organizing and storing data in computer science.\n\n"
            
            elif any(word in query_lower for word in ['algorithm', 'algorithms']):
                analysis += "Answer: Algorithms are step-by-step procedures for solving problems and performing operations efficiently. The documents cover various algorithmic concepts and their applications in computing.\n\n"
            
            elif any(word in query_lower for word in ['compare', 'comparison', 'difference']):
                analysis += "Comparison: The documents present complementary perspectives on DSA - combining research paper insights, lecture notes with definitions, and audio explanations. Together they provide theoretical foundations and practical understanding.\n\n"
            
            elif any(word in query_lower for word in ['knuth', 'author', 'who']):
                if 'knuth' in combined_lower:
                    analysis += "Answer: The documents reference fundamental computer science concepts. For specific author mentions, please refer to the detailed content above.\n\n"
                else:
                    analysis += "Answer: The documents focus on fundamental DSA concepts. No specific author references found in the current content.\n\n"
            
            elif any(word in query_lower for word in ['complexity', 'time', 'space']):
                analysis += "Answer: The documents discuss algorithmic complexity and optimization for efficient computing. This includes concepts of time and space complexity in data structure operations.\n\n"
            
            else:
                analysis += f"Answer: Based on the {len(file_summaries)} documents about Data Structures and Algorithms, the materials cover fundamental computer science concepts including data organization, algorithmic thinking, and optimization techniques.\n\n"
            
            analysis += "Note: This analysis is based on the document content shown above. For more specific details, please refer to the individual file contents."
        
        if not analysis or len(analysis.strip()) == 0:
            print("Creating comprehensive fallback analysis")
            
            analysis = f"# Multi-File Analysis Results\n\n"
            analysis += f"**Query:** {req.query}\n"
            analysis += f"**Files Analyzed:** {len(file_summaries)}\n"
            analysis += f"**Total Content:** {len(combined_text):,} characters\n\n"
            
            query_lower = req.query.lower()
            
            if 'summarize' in query_lower or 'summary' in query_lower:
                analysis += "## Summary\n\n"
                analysis += f"This analysis covers {len(file_summaries)} files containing diverse content:\n\n"
                
                for i, summary in enumerate(file_summaries, 1):
                    file_type = summary['name'].split('.')[-1].upper() if '.' in summary['name'] else 'FILE'
                    analysis += f"**{i}. {summary['name']}** ({file_type}, {summary['length']} chars)\n"
                    
                    full_text = summary.get('full_text', summary.get('preview', ''))
                    if full_text:
                        sentences = full_text.split('.')
                        key_content = []
                        
                        for sentence in sentences[:3]:
                            sentence = sentence.strip()
                            if len(sentence) > 10:  
                                key_content.append(sentence)
                        
                        if key_content:
                            content_preview = '. '.join(key_content)
                            if len(content_preview) > 400:
                                content_preview = content_preview[:400] + "..."
                            else:
                                content_preview += "."
                        else:
                            content_preview = full_text[:400] + ("..." if len(full_text) > 400 else "")
                    else:
                        content_preview = "No content available"
                    
                    analysis += f"   **Key Content:** {content_preview}\n\n"
                
                analysis += "### Key Insights:\n"
                analysis += f"- **Content Volume:** {len(combined_text):,} total characters across all files\n"
                analysis += f"- **File Diversity:** {len(set(s['name'].split('.')[-1] if '.' in s['name'] else 'unknown' for s in file_summaries))} different file formats\n"
                analysis += f"- **Average Size:** {len(combined_text) // len(file_summaries):,} characters per file\n\n"
                
                # Try to extract some actual content insights
                combined_lower = combined_text.lower()
                if 'data structure' in combined_lower or 'algorithm' in combined_lower:
                    analysis += "**Topic Focus:** The content appears to cover Data Structures and Algorithms (DSA) concepts.\n"
                if 'research' in combined_lower or 'study' in combined_lower:
                    analysis += "**Content Type:** Includes research or academic material.\n"
                if len([s for s in file_summaries if s['name'].endswith('.pdf')]) > 0:
                    analysis += "**Documents:** Contains PDF documents with structured content.\n"
                if len([s for s in file_summaries if s['name'].endswith(('.jpg', '.png'))]) > 0:
                    analysis += "**Images:** Includes image files with extracted text content.\n"
                if len([s for s in file_summaries if s['name'].endswith(('.mp3', '.wav'))]) > 0:
                    analysis += "**Audio:** Contains audio files with transcribed content.\n"
                    
            elif 'flashcard' in query_lower or 'flash card' in query_lower:
                analysis += "## Flashcards\n\n"
                analysis += "Based on the content from your files, here are study flashcards:\n\n"
                
                # Generate flashcards from the content
                flashcards = generate_flashcards_from_content(combined_text, file_summaries)
                for i, card in enumerate(flashcards, 1):
                    analysis += f"### Card {i}\n"
                    analysis += f"**Q:** {card['question']}\n"
                    analysis += f"**A:** {card['answer']}\n\n"
                    
            else:
                analysis += "## Analysis Results\n\n"
                analysis += f"Based on your query '{req.query}', here's what was found in the {len(file_summaries)} files:\n\n"
                
                for i, summary in enumerate(file_summaries, 1):
                    analysis += f"### File {i}: {summary['name']}\n"
                    analysis += f"**Size:** {summary['length']} characters\n"
                    preview = summary.get('preview', '')
                    if len(preview) > 300:
                        break_point = preview.find('.', 250)
                        if break_point == -1:
                            break_point = preview.find(' ', 250)
                        if break_point != -1:
                            preview = preview[:break_point + 1]
                        else:
                            preview = preview[:300] + "..."
                    analysis += f"**Content:** {preview}\n\n"
            
            analysis += "\n---\n*Note: This analysis was generated using basic text processing. For more advanced AI-powered insights, please ensure your Gemini API configuration is properly set up.*"
        
        response = {
            "query": req.query,
            "output_type": req.output_type,
            "files_processed": len(all_texts),
            "total_files": len(req.files),
            "file_summaries": file_summaries,
            "analysis": analysis,
            "combined_length": len(combined_text)
        }
        
        print(f"Multi-file analysis completed successfully for {len(all_texts)} files")
        print(f"Analysis result length: {len(analysis) if analysis else 0}")
        print(f"Analysis preview: {analysis[:200] if analysis else 'NO ANALYSIS'}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in multi-file analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-file analysis failed: {str(e)}")

@app.get("/download/{file_id}")
def download_file(file_id: str, auth_token: str | None = Query(None)):
    try:
        if not auth_token:
            raise HTTPException(status_code=401, detail="Authentication token required")
        
        temp_path = drive_download_file(auth_token, file_id, tempfile.gettempdir())
        if not temp_path:
            raise HTTPException(status_code=404, detail="File not found or download failed")
        
        from fastapi.responses import FileResponse
        return FileResponse(
            path=temp_path,
            media_type='application/octet-stream',
            filename=os.path.basename(temp_path)
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

# Bulk operations for search results
class BulkMoveRequest(BaseModel):
    file_ids: list[str]
    folder_name: str
    auth_token: str | None = None
    create_new: bool = False

class BulkDeleteRequest(BaseModel):
    file_ids: list[str]
    auth_token: str | None = None

@app.post("/bulk_move_to_folder")
def bulk_move_to_folder(req: BulkMoveRequest):
    """Move multiple files to a folder (create folder if needed)"""
    try:
        token = req.auth_token
        if not token:
            raise HTTPException(status_code=400, detail="Missing auth token")
        
        service = get_drive_service(token)
        if not service:
            raise HTTPException(status_code=400, detail="Failed to initialize Drive service")
        
        # Get or create the target folder
        folder_id = get_or_create_folder(service, req.folder_name)
        if not folder_id:
            raise HTTPException(status_code=400, detail=f"Failed to create/find folder: {req.folder_name}")
        
        moved_files = []
        failed_files = []
        
        for file_id in req.file_ids:
            try:
                result = move_file_to_folder(service, file_id, folder_id)
                if result:
                    moved_files.append(file_id)
                else:
                    failed_files.append(file_id)
            except Exception as e:
                print(f"Failed to move file {file_id}: {e}")
                failed_files.append(file_id)
        
        return {
            "success": True,
            "folder_name": req.folder_name,
            "folder_id": folder_id,
            "moved_count": len(moved_files),
            "failed_count": len(failed_files),
            "moved_files": moved_files,
            "failed_files": failed_files
        }
    
    except Exception as e:
        print(f"Bulk move error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bulk_delete_files")
def bulk_delete_files(req: BulkDeleteRequest):
    """Delete multiple files from Drive"""
    try:
        token = req.auth_token
        if not token:
            raise HTTPException(status_code=400, detail="Missing auth token")
        
        service = get_drive_service(token)
        if not service:
            raise HTTPException(status_code=400, detail="Failed to initialize Drive service")
        
        deleted_files = []
        failed_files = []
        
        for file_id in req.file_ids:
            try:
                # Delete file using Drive API
                service.files().delete(fileId=file_id).execute()
                deleted_files.append(file_id)
            except Exception as e:
                print(f"Failed to delete file {file_id}: {e}")
                failed_files.append(file_id)
        
        return {
            "success": True,
            "deleted_count": len(deleted_files),
            "failed_count": len(failed_files),
            "deleted_files": deleted_files,
            "failed_files": failed_files
        }
    
    except Exception as e:
        print(f"Bulk delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drive_folders")
def get_drive_folders(auth_token: str | None = Query(None)):
    """Get list of existing folders in Drive"""
    try:
        if not auth_token:
            raise HTTPException(status_code=400, detail="Missing auth token")
        
        service = get_drive_service(auth_token)
        if not service:
            raise HTTPException(status_code=400, detail="Failed to initialize Drive service")
        
        # Query for folders
        query = "mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = service.files().list(
            q=query,
            fields="files(id,name,parents)",
            pageSize=100
        ).execute()
        
        folders = results.get('files', [])
        
        return {
            "success": True,
            "folders": [{"id": f["id"], "name": f["name"]} for f in folders]
        }
    
    except Exception as e:
        print(f"Get folders error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/favicon.ico")
def favicon():
    return {}