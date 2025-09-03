# services/dedup/app.py
from fastapi import FastAPI, HTTPException
import sqlite3, os, glob, numpy as np, shutil
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

DB = "metadata_db/clarifile.db"
EMB_DIR = "storage/embeddings"
ORGANIZED_DIR = "storage/organized_demo"
THRESHOLD = 0.92  # cosine similarity threshold for near-duplicates

app = FastAPI()

def get_conn():
    return sqlite3.connect(DB, check_same_thread=False)

@app.get("/duplicates")
def duplicates(threshold: float = THRESHOLD):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, file_path, file_hash, file_name FROM files")
    rows = cur.fetchall()
    files = [{"id":r[0],"path":r[1],"hash":r[2],"name":r[3]} for r in rows]

    # exact duplicates (group by file_hash)
    by_hash = {}
    for f in files:
        by_hash.setdefault(f["hash"], []).append(f)
    exact_pairs = []
    for h, group in by_hash.items():
        if len(group) > 1:
            # all pairs in group
            ids = [g["id"] for g in group]
            for i in range(len(ids)):
                for j in range(i+1, len(ids)):
                    exact_pairs.append({"file_a": ids[i], "file_b": ids[j], "method": "exact", "score": 1.0})

    # near duplicates via averaged embeddings
    # Build doc-level embeddings by averaging chunk embeddings for each file
    doc_ids = []
    doc_vecs = []
    for f in files:
        # find all chunk embedding files whose chunk_id belongs to this file
        # we use DB to find chunk ids for each file
        cur.execute("SELECT id FROM chunks WHERE file_id=?", (f["id"],))
        chunk_rows = cur.fetchall()
        chunk_ids = [r[0] for r in chunk_rows]
        vecs = []
        for cid in chunk_ids:
            fn = os.path.join(EMB_DIR, f"{cid}.npy")
            if os.path.exists(fn):
                vecs.append(np.load(fn))
        if len(vecs) > 0:
            doc_vec = np.mean(np.vstack(vecs), axis=0)
            doc_ids.append(f["id"])
            doc_vecs.append(doc_vec)
    if len(doc_vecs) >= 2:
        M = np.vstack(doc_vecs).astype('float32')
        M = normalize(M, axis=1)
        sims = cosine_similarity(M, M)  # symmetric
        near_pairs = []
        n = M.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                sim = float(sims[i, j])
                if sim >= threshold:
                    near_pairs.append({"file_a": int(doc_ids[i]), "file_b": int(doc_ids[j]), "method":"embedding", "score": sim})
    else:
        near_pairs = []

    conn.close()
    return {"exact": exact_pairs, "near": near_pairs}

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
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id,file_path,file_name,proposed_label FROM files WHERE id=?", (file_a,))
    a = cur.fetchone()
    cur.execute("SELECT id,file_path,file_name,proposed_label FROM files WHERE id=?", (file_b,))
    b = cur.fetchone()
    if not a or not b:
        raise HTTPException(status_code=404, detail="one of files not found")
    # actions
    results = {"copied":[]}
    if action == "keep_a":
        dest = os.path.join(ORGANIZED_DIR, a[3] or "Duplicates")
        os.makedirs(dest, exist_ok=True)
        shutil.copy2(a[1], os.path.join(dest, a[2]))
        cur.execute("UPDATE files SET final_label=?, status=? WHERE id=?", (a[3] or "Duplicates", "approved", a[0]))
        cur.execute("UPDATE files SET status=? WHERE id=?", ("ignored_duplicate", b[0]))
        results["copied"].append(a[0])
    elif action == "keep_b":
        dest = os.path.join(ORGANIZED_DIR, b[3] or "Duplicates")
        os.makedirs(dest, exist_ok=True)
        shutil.copy2(b[1], os.path.join(dest, b[2]))
        cur.execute("UPDATE files SET final_label=?, status=? WHERE id=?", (b[3] or "Duplicates", "approved", b[0]))
        cur.execute("UPDATE files SET status=? WHERE id=?", ("ignored_duplicate", a[0]))
        results["copied"].append(b[0])
    elif action == "keep_both":
        desta = os.path.join(ORGANIZED_DIR, a[3] or "Duplicates")
        destb = os.path.join(ORGANIZED_DIR, b[3] or "Duplicates")
        os.makedirs(desta, exist_ok=True)
        os.makedirs(destb, exist_ok=True)
        shutil.copy2(a[1], os.path.join(desta, a[2]))
        shutil.copy2(b[1], os.path.join(destb, b[2]))
        cur.execute("UPDATE files SET final_label=?, status=? WHERE id=?", (a[3] or "Duplicates", "approved", a[0]))
        cur.execute("UPDATE files SET final_label=?, status=? WHERE id=?", (b[3] or "Duplicates", "approved", b[0]))
        results["copied"].extend([a[0], b[0]])
    else:
        raise HTTPException(status_code=400, detail="unknown action")
    conn.commit()
    conn.close()
    return {"status":"resolved","details":results}
