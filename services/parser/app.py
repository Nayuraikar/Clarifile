# parser/app.py
from fastapi import FastAPI, HTTPException
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
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- Paths & constants ---
DB = "metadata_db/clarifile.db"
SAMPLE_DIR = "storage/sample_files"
ORGANIZED_DIR = "storage/organized_demo"
ALLOWED_EXTS = {
    ".txt", ".pdf", ".png", ".jpg", ".jpeg",
    ".mp3", ".wav", ".mp4", ".mkv", ".mpeg"
}

app = FastAPI()

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
        tags TEXT
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
    # Break long text into smaller chunks if > 2000 chars
    chunks = [long_text[i:i+2000] for i in range(0, len(long_text), 2000)]
    summary_chunks = []
    for c in chunks:
        try:
            out = summarizer(c, max_length=150, min_length=30, do_sample=False)
            summary_chunks.append(out[0].get("summary_text", "").strip())
        except:
            summary_chunks.append(c[:200])
    return " ".join(summary_chunks)


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
        vec = embed_texts([summary])[0]
        c = conn.cursor()
        c.execute("SELECT id, name, rep_vector FROM categories")
        rows = c.fetchall()
        best_sim, best_row = -1.0, None
        for cid, name, rep_vec_json in rows:
            rep_vec = np.array(json.loads(rep_vec_json), dtype=np.float32)
            sim = float(np.dot(vec, rep_vec))
            if sim > best_sim:
                best_sim, best_row = sim, (cid, name)
        if best_row and best_sim >= threshold:
            return best_row[0], best_row[1]
        name = compress_title(summary)
        c.execute("INSERT INTO categories(name, rep_summary, rep_vector) VALUES (?,?,?)",
                  (name, summary, json.dumps(vec.tolist())))
        conn.commit()
        return c.lastrowid, name
    except:
        return None, compress_title(summary)

# --- API ---
@app.post("/scan_folder")
def scan_folder():
    inserted_files_with_chunks = 0
    processed_files = []
    for fname in os.listdir(SAMPLE_DIR):
        path = os.path.join(SAMPLE_DIR, fname)
        if not os.path.isfile(path): continue
        ext = os.path.splitext(path)[1].lower()
        if ext not in ALLOWED_EXTS: continue

        fh = file_hash(path)
        size = os.path.getsize(path)
        cur = conn.cursor()
        cur.execute("""INSERT OR IGNORE INTO files
                       (file_path,file_name,file_hash,size,status,proposed_label)
                       VALUES (?,?,?,?,?,?)""",
                    (path, fname, fh, size, "new", "Uncategorized"))
        conn.commit()
        cur.execute("SELECT id FROM files WHERE file_hash=?", (fh,))
        row = cur.fetchone()
        if not row: continue
        file_id = row[0]

        text, summary, transcript, tags = "", "", "", []

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
        elif ext in {".mp4", ".mkv", ".mpeg"}:
            transcript, frame_tags = process_video_ffmpeg(path)
            text = transcript + " " + " ".join(frame_tags)
            tags.extend(frame_tags)

        if text.strip():
            chunks = chunk_text(text)
            for s, e, chunk in chunks:
                chash = hashlib.sha256((str(file_id)+str(s)+str(e)).encode()).hexdigest()
                cur.execute("""INSERT OR IGNORE INTO chunks
                            (file_id,chunk_text,chunk_hash,start_offset,end_offset)
                            VALUES (?,?,?,?,?)""",
                            (file_id, chunk, chash, s, e))
            conn.commit()
            inserted_files_with_chunks += 1

            summary = summarize_text(text)
            cat_id, cat_name = assign_category_from_summary(summary)
        else:
            cat_id, cat_name, summary = None, "Uncategorized", ""

        cur.execute("""UPDATE files
                       SET summary=?, proposed_label=?, category_id=?, transcript=?, tags=?
                       WHERE id=?""",
                    (summary, cat_name, cat_id, transcript, json.dumps(tags), file_id))
        conn.commit()
        processed_files.append({
            "file": fname,
            "category": cat_name,
            "summary": summary[:120],
            "transcript_len": len(transcript),
            "tags": tags
        })
    return {"status": "scanned", "chunks_inserted": inserted_files_with_chunks,
            "processed_files": processed_files}

@app.get("/list_proposals")
def list_proposals():
    c = conn.cursor()
    c.execute("""SELECT id, file_name, proposed_label, final_label 
                 FROM files 
                 WHERE summary IS NOT NULL AND summary != ''""")
    rows = c.fetchall()
    return [{"id": r[0], "file": r[1], "proposed": r[2], "final": r[3]} for r in rows]

@app.post("/approve")
def approve(file_id: int, final_label: str):
    cur = conn.cursor()
    cur.execute("UPDATE files SET final_label=? WHERE id=?", (final_label, file_id))
    conn.commit()
    return {"status": "approved", "file_id": file_id, "final_label": final_label}

@app.get("/categories")
def categories():
    c = conn.cursor()
    c.execute("SELECT id, name FROM categories")
    rows = c.fetchall()
    return [{"id": r[0], "name": r[1]} for r in rows]

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