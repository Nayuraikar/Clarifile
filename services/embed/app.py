# services/embed/app.py
from fastapi import FastAPI
import sqlite3, os, numpy as np
from sentence_transformers import SentenceTransformer

DB = "metadata_db/clarifile.db"
EMB_DIR = "storage/embeddings"
os.makedirs(EMB_DIR, exist_ok=True)

app = FastAPI()
# model will download at first run (CPU)
model = SentenceTransformer("all-mpnet-base-v2")

def get_db_conn():
    return sqlite3.connect(DB, check_same_thread=False)

@app.post("/embed_pending")
def embed_pending():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, chunk_text FROM chunks")
    rows = cur.fetchall()
    created = 0
    for r in rows:
        cid = r[0]
        txt = r[1]
        fn = os.path.join(EMB_DIR, f"{cid}.npy")
        if os.path.exists(fn):
            continue
        vec = model.encode(txt, convert_to_numpy=True)
        np.save(fn, vec)
        created += 1
    conn.close()
    return {"embedded": created}
