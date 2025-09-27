# services/embed/enhanced_app.py
"""
Enhanced embedding service with integrated smart categorization.
Supports both the original embedding functionality and the new chunked, TF-IDF weighted approach.
"""

from fastapi import FastAPI, HTTPException
import sqlite3, os, numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List, Optional
import json

# Import our enhanced categorization
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from smart_categorize_v2 import EnhancedCategorizer, chunk_text, extract_text

DB = "metadata_db/clarifile.db"
EMB_DIR = "storage/embeddings"
MODELS_DIR = "storage/categorization_models"
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

app = FastAPI(title="Enhanced Clarifile Embedding Service", version="2.0")

# Initialize models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Consistent with categorization
enhanced_categorizer = EnhancedCategorizer("all-MiniLM-L6-v2")

class TextsRequest(BaseModel):
    texts: List[str]

class CategorizeRequest(BaseModel):
    content: str
    content_type: Optional[str] = "document"
    use_enhanced: Optional[bool] = True

class FileProcessRequest(BaseModel):
    file_path: str
    extract_chunks: Optional[bool] = True
    categorize: Optional[bool] = True

class BatchCategorizeRequest(BaseModel):
    file_paths: List[str]
    output_dir: str
    k: Optional[int] = None
    chunk_size: Optional[int] = 2000
    overlap: Optional[int] = 200
    use_hdbscan: Optional[bool] = False

def get_db_conn():
    return sqlite3.connect(DB, check_same_thread=False)

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "embedding_model": embedding_model._first_module().config.get('name_or_path', 'unknown'),
        "enhanced_categorizer": "loaded" if enhanced_categorizer.model else "not_loaded"
    }

@app.post("/embed_pending")
def embed_pending():
    """Original embedding functionality for backward compatibility."""
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
        vec = embedding_model.encode(txt, convert_to_numpy=True)
        np.save(fn, vec)
        created += 1
    conn.close()
    return {"embedded": created}

@app.post("/embed_texts")
def embed_texts(req: TextsRequest):
    """Original text embedding functionality."""
    vecs = embedding_model.encode(req.texts, convert_to_numpy=True, normalize_embeddings=True)
    return {"vectors": vecs.tolist()}

@app.post("/categorize_content")
def categorize_content(req: CategorizeRequest):
    """
    Enhanced content categorization endpoint.
    Supports both original and enhanced categorization methods.
    """
    try:
        if req.use_enhanced:
            # Use the enhanced categorizer with chunking and TF-IDF weighting
            category = enhanced_categorizer.categorize_content(req.content, req.content_type)
        else:
            # Fallback to simpler semantic analysis
            category = enhanced_categorizer._analyze_content_semantically(
                req.content[:2000], req.content.lower()
            )
        
        return {
            "category": category,
            "method": "enhanced" if req.use_enhanced else "semantic",
            "content_length": len(req.content)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Categorization failed: {str(e)}")

@app.post("/process_file")
def process_file(req: FileProcessRequest):
    """
    Process a single file: extract text, create chunks, generate embeddings, and categorize.
    """
    try:
        if not os.path.exists(req.file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Extract text
        text = extract_text(req.file_path)
        if not text:
            return {
                "file_path": req.file_path,
                "error": "No text could be extracted from file"
            }
        
        result = {
            "file_path": req.file_path,
            "text_length": len(text),
            "text_preview": text[:200] + "..." if len(text) > 200 else text
        }
        
        # Create chunks if requested
        if req.extract_chunks:
            chunks = chunk_text(text, chunk_size=2000, overlap=200)
            result["chunks"] = {
                "count": len(chunks),
                "chunks": chunks[:3]  # Return first 3 chunks as preview
            }
            
            # Generate embeddings for chunks
            if chunks:
                chunk_embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
                result["embeddings"] = {
                    "shape": chunk_embeddings.shape,
                    "sample": chunk_embeddings[0][:5].tolist()  # First 5 dimensions of first chunk
                }
        
        # Categorize if requested
        if req.categorize:
            category = enhanced_categorizer.categorize_content(text)
            result["category"] = category
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.post("/batch_categorize")
def batch_categorize(req: BatchCategorizeRequest):
    """
    Perform batch categorization on multiple files using the enhanced pipeline.
    """
    try:
        # Validate input files
        valid_files = [f for f in req.file_paths if os.path.exists(f)]
        if not valid_files:
            raise HTTPException(status_code=400, detail="No valid files found")
        
        # Import the main categorization function
        from smart_categorize_v2 import main as categorize_main, build_file_embeddings, cluster_embeddings, name_clusters_by_tfidf, save_model_outputs
        
        # Create output directory
        os.makedirs(req.output_dir, exist_ok=True)
        
        # Build file embeddings
        file_embeddings, file_chunks, all_chunks, vec_chunks = build_file_embeddings(
            valid_files, embedding_model, 
            chunk_size=req.chunk_size, overlap=req.overlap
        )
        
        # Filter valid embeddings
        valid_files_filtered = []
        valid_embeddings = []
        valid_chunks = []
        
        for i, (f, emb, chunks) in enumerate(zip(valid_files, file_embeddings, file_chunks)):
            if np.any(emb != 0):
                valid_files_filtered.append(f)
                valid_embeddings.append(emb)
                valid_chunks.append(chunks)
        
        if not valid_files_filtered:
            return {"error": "No files with extractable content found"}
        
        valid_embeddings = np.array(valid_embeddings)
        
        # Cluster embeddings
        labels, km = cluster_embeddings(
            valid_embeddings, 
            use_hdbscan=req.use_hdbscan, 
            k=req.k
        )
        
        # Generate cluster names
        docs_per_file = [" ".join(chs) if chs else "" for chs in valid_chunks]
        cluster_terms = name_clusters_by_tfidf(docs_per_file, labels, top_n=3)
        
        # Save results
        mapping = {}
        for fpath, lab in zip(valid_files_filtered, labels):
            if lab == -1:
                labname = "noise_uncategorized"
            else:
                terms = cluster_terms.get(lab, ["cat"])
                labname = f"cat{int(lab)}_" + "_".join(terms[:3])
            
            folder = os.path.join(req.output_dir, labname)
            os.makedirs(folder, exist_ok=True)
            
            # Copy file to category folder
            import shutil
            shutil.copy2(fpath, folder)
            mapping[fpath] = labname
        
        # Save metadata
        with open(os.path.join(req.output_dir, "categories_map.json"), "w") as f:
            json.dump(mapping, f, indent=2)
        
        np.save(os.path.join(req.output_dir, "file_embeddings.npy"), valid_embeddings)
        save_model_outputs(req.output_dir, valid_files_filtered, labels, valid_embeddings, 
                          clusterer=km, tfidf_vectorizer=vec_chunks)
        
        return {
            "success": True,
            "total_files": len(req.file_paths),
            "processed_files": len(valid_files_filtered),
            "categories": len(set(labels)),
            "cluster_terms": cluster_terms,
            "output_dir": req.output_dir,
            "mapping": mapping
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch categorization failed: {str(e)}")

@app.post("/load_categorization_model")
def load_categorization_model(model_dir: str):
    """Load a previously saved categorization model."""
    try:
        enhanced_categorizer.load_saved_model(model_dir)
        return {
            "success": True,
            "model_dir": model_dir,
            "centroids_loaded": enhanced_categorizer.centroids is not None,
            "tfidf_loaded": enhanced_categorizer.tfidf_vectorizer is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/list_models")
def list_models():
    """List available categorization models."""
    models = []
    if os.path.exists(MODELS_DIR):
        for item in os.listdir(MODELS_DIR):
            item_path = os.path.join(MODELS_DIR, item)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, "classification_metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        models.append({
                            "name": item,
                            "path": item_path,
                            "metadata": metadata
                        })
                    except:
                        models.append({
                            "name": item,
                            "path": item_path,
                            "metadata": None
                        })
    return {"models": models}

@app.get("/model_info")
def model_info():
    """Get information about the currently loaded model."""
    return {
        "embedding_model": {
            "name": embedding_model._first_module().config.get('name_or_path', 'unknown'),
            "loaded": True
        },
        "categorization_model": {
            "loaded": enhanced_categorizer.model is not None,
            "centroids_available": enhanced_categorizer.centroids is not None,
            "tfidf_available": enhanced_categorizer.tfidf_vectorizer is not None,
            "n_centroids": len(enhanced_categorizer.centroids) if enhanced_categorizer.centroids is not None else 0
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
