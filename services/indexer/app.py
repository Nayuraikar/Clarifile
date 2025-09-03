# services/indexer/app.py
from fastapi import FastAPI, HTTPException
import numpy as np, os, glob
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

EMB_DIR = "storage/embeddings"
app = FastAPI()

_nn = None
_ids = None
_matrix = None

@app.post("/reindex")
def reindex():
    global _nn, _ids, _matrix
    files = glob.glob(os.path.join(EMB_DIR, "*.npy"))
    if not files:
        return {"error":"no embeddings found"}
    ids = []
    vecs = []
    for p in files:
        cid = int(os.path.splitext(os.path.basename(p))[0])
        v = np.load(p).astype('float32')
        ids.append(cid)
        vecs.append(v)
    matrix = np.vstack(vecs)
    # normalize to unit vectors for cosine similarity
    matrix = normalize(matrix, norm='l2', axis=1)
    nn = NearestNeighbors(n_neighbors=10, metric='cosine').fit(matrix)
    _nn = nn
    _ids = np.array(ids)
    _matrix = matrix
    return {"indexed": len(ids)}

@app.get("/knn")
def knn(chunk_id: int = 1, k: int = 5):
    global _nn, _ids, _matrix
    fn = os.path.join(EMB_DIR, f"{chunk_id}.npy")
    if not os.path.exists(fn):
        raise HTTPException(status_code=404, detail="embedding for chunk not found")
    vec = np.load(fn).astype('float32').reshape(1, -1)
    vec = normalize(vec, norm='l2')
    if _nn is None:
        # attempt to build index in-memory if not built yet
        reindex()
    dists, idxs = _nn.kneighbors(vec, n_neighbors=k)
    # sklearn's 'cosine' returns cosine distance = 1 - cosine_similarity
    neighbors = []
    for dist, idx in zip(dists[0], idxs[0]):
        cid = int(_ids[idx])
        sim = 1.0 - float(dist)  # similarity
        neighbors.append({"chunk_id": cid, "similarity": sim})
    return {"neighbors": neighbors}
