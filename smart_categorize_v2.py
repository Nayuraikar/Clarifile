#!/usr/bin/env python3
"""
smart_categorize_v2.py
Enhanced content-based categorization with chunking, TF-IDF weighted embeddings, and intelligent clustering.
Integrates with Clarifile's existing architecture.

Usage:
  python smart_categorize_v2.py --source ./my_files --dest ./sorted
  python smart_categorize_v2.py --source ./my_files --dest ./sorted --k 5
"""

import os, argparse, json, shutil
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Text & file libs
import pdfplumber
import docx
from PIL import Image
import pytesseract

# Embedding & clustering
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# optional advanced
try:
    import umap
    import hdbscan
    UMAP_HDBSCAN = True
except Exception:
    UMAP_HDBSCAN = False

# optional pdf->images
try:
    from pdf2image import convert_from_path
    PDF2IMAGE = True
except:
    PDF2IMAGE = False

# optional FAISS for fast similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except:
    FAISS_AVAILABLE = False

# ---------- helpers ----------
def extract_text(path, max_chars=20000, ocr_if_empty=True):
    """Extract text from various file formats with OCR fallback."""
    ext = Path(path).suffix.lower()
    text = ""
    try:
        if ext in (".txt", ".md", ".csv", ".log"):
            with open(path, "r", encoding="utf8", errors="ignore") as f:
                text = f.read()
        elif ext == ".pdf":
            try:
                with pdfplumber.open(path) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                    text = "\n".join(pages).strip()
            except Exception:
                text = ""
            if (not text) and ocr_if_empty and PDF2IMAGE:
                try:
                    imgs = convert_from_path(path)
                    text = "\n".join(pytesseract.image_to_string(img) for img in imgs)
                except Exception as e:
                    print(f"OCR failed for {path}: {e}")
        elif ext in (".docx",):
            doc = docx.Document(path)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
            text = pytesseract.image_to_string(Image.open(path))
        else:
            # try reading chunk as text fallback
            with open(path, "rb") as f:
                raw = f.read(200000)
                try:
                    text = raw.decode("utf8", errors="ignore")
                except:
                    text = ""
    except Exception as e:
        print(f"Warn extract: {path} - {e}")
        text = ""
    text = " ".join(text.split())
    return text[:max_chars]

def chunk_text(text, chunk_size=2000, overlap=200):
    """Split text into overlapping chunks for better embedding quality."""
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks

# ---------- main pipeline ----------
def build_file_embeddings(files, model, chunk_size=2000, overlap=200):
    """
    Build file embeddings using chunking and TF-IDF weighted aggregation.
    This is the core improvement over the original approach.
    """
    docs = []
    file_chunks = []   # list of lists: chunks per file
    chunk_to_file = [] # mapping chunk idx -> file idx

    print("Extracting text & chunking...")
    for i, p in enumerate(tqdm(files)):
        text = extract_text(p)
        if not text:
            file_chunks.append([])
            continue
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        file_chunks.append(chunks)
        for _ in chunks:
            chunk_to_file.append(i)
        docs.extend(chunks)

    if len(docs) == 0:
        raise ValueError("No text extracted. Enable OCR or check files.")

    # TF-IDF on chunks -> chunk importance
    print("Computing TF-IDF on chunks...")
    vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))
    X_tfidf = vec.fit_transform(docs)
    # chunk importance as sum of tfidf weights (scalar)
    chunk_weights = X_tfidf.sum(axis=1).A1
    # avoid zeros
    chunk_weights = chunk_weights + 1e-6

    # embed chunks (batch)
    print(f"Embedding {len(docs)} chunks with model...")
    chunk_embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)

    # aggregate into file embeddings: weighted average of chunk embeddings per file
    n_files = len(files)
    emb_dim = chunk_embeddings.shape[1]
    file_embeddings = np.zeros((n_files, emb_dim), dtype=np.float32)
    file_weights = np.zeros(n_files, dtype=np.float32)
    
    for i, emb in enumerate(chunk_embeddings):
        fidx = chunk_to_file[i]
        w = chunk_weights[i]
        file_embeddings[fidx] += emb * w
        file_weights[fidx] += w
    
    # handle files without chunks (zeros): leave as zero and later filter
    nonzero = file_weights > 0
    file_embeddings[nonzero] /= file_weights[nonzero][:,None]
    
    return file_embeddings, file_chunks, docs, vec

def cluster_embeddings(embs, use_hdbscan=True, k=None):
    """
    Cluster embeddings using either HDBSCAN/UMAP (variable clusters) or KMeans (fixed K).
    """
    # If HDBSCAN/UMAP available & not forcing K, use it (variable clusters)
    if use_hdbscan and UMAP_HDBSCAN and k is None:
        print("Reducing dims with UMAP and clustering with HDBSCAN...")
        reducer = umap.UMAP(n_components=5, random_state=42)
        emb_low = reducer.fit_transform(embs)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
        labels = clusterer.fit_predict(emb_low)
        # labels == -1 => noise/unclustered
        return labels, None
    else:
        # fallback KMeans. If k not given choose with silhouette search (2..min(10,n-1))
        if k is None:
            k = choose_k_auto(embs, min_k=2, max_k=min(10, max(2, len(embs)-1)))
        print(f"Clustering with KMeans k={k}")
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embs)
        return km.labels_, km

def choose_k_auto(embeddings, min_k=2, max_k=10):
    """Automatically choose optimal K using silhouette score."""
    n = len(embeddings)
    if n < 3:
        return 1 if n==1 else 2
    max_k = min(max_k, n-1)
    best_k = min_k
    best_score = -1.0
    
    print(f"Choosing optimal K between {min_k} and {max_k}...")
    for k in range(min_k, max_k+1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        try:
            score = silhouette_score(embeddings, labels)
        except:
            score = -1
        print(f"  K={k}: silhouette={score:.3f}")
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

def name_clusters_by_tfidf(docs_per_file, labels, top_n=3):
    """Generate meaningful cluster names using TF-IDF top terms."""
    # Filter out empty docs
    valid_docs = [doc for doc in docs_per_file if doc.strip()]
    if not valid_docs:
        return {c: ["empty"] for c in set(labels)}
    
    vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))
    X = vec.fit_transform(valid_docs)
    terms = np.array(vec.get_feature_names_out())
    cluster_terms = {}
    
    # Create mapping from original index to valid doc index
    doc_mapping = {}
    valid_idx = 0
    for i, doc in enumerate(docs_per_file):
        if doc.strip():
            doc_mapping[i] = valid_idx
            valid_idx += 1
    
    for c in sorted(set(labels)):
        if c == -1:
            cluster_terms[c] = ["noise"]
            continue
        
        # Get indices for this cluster, but only those with valid docs
        idxs = [doc_mapping[i] for i, l in enumerate(labels) if l == c and i in doc_mapping]
        
        if not idxs:
            cluster_terms[c] = ["empty"]
            continue
            
        mean_tfidf = X[idxs].mean(axis=0).A1
        top_idx = mean_tfidf.argsort()[::-1][:top_n]
        cluster_terms[c] = [t.replace(" ", "_") for t in terms[top_idx].tolist()]
    
    return cluster_terms

# ---------- incremental classification helper ----------
def save_model_outputs(dst, files, labels, file_embeddings, clusterer=None, tfidf_vectorizer=None):
    """Save centroids and metadata for incremental classification."""
    os.makedirs(dst, exist_ok=True)
    mapping = {}
    for f, lab in zip(files, labels):
        mapping[f] = int(lab)
    
    # compute centroids for each cluster (ignore -1)
    unique = [c for c in sorted(set(labels)) if c != -1]
    centroids = []
    cluster_map = {}
    
    for c in unique:
        idxs = [i for i, l in enumerate(labels) if l == c]
        if idxs:
            cent = file_embeddings[idxs].mean(axis=0)
            centroids.append(cent)
            cluster_map[c] = len(centroids) - 1  # maps original label->centroid index
    
    centroids = np.stack(centroids) if centroids else np.zeros((0, file_embeddings.shape[1]))
    
    # Save centroids and metadata
    np.save(os.path.join(dst, "centroids.npy"), centroids)
    
    metadata = {
        "mapping": mapping,
        "cluster_map": cluster_map,
        "n_clusters": len(unique),
        "total_files": len(files)
    }
    
    with open(os.path.join(dst, "classification_metadata.json"), "w", encoding="utf8") as f:
        json.dump(metadata, f, indent=2)
    
    # Save TF-IDF vectorizer if provided (for chunk weighting in incremental classification)
    if tfidf_vectorizer:
        import joblib
        joblib.dump(tfidf_vectorizer, os.path.join(dst, "tfidf_vectorizer.pkl"))
    
    print(f"Saved centroids and metadata to {dst}")

def classify_new_file(path, model, centroids, vec_for_weights=None, chunk_size=2000, overlap=200, similarity_threshold=0.65):
    """
    Classify a new file using saved centroids.
    Returns classification result with confidence score.
    """
    text = extract_text(path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    
    if not chunks:
        return {"label": "no_text", "score": 0.0, "path": path}
    
    # compute chunk importance using provided vectorizer if given
    if vec_for_weights:
        try:
            X = vec_for_weights.transform(chunks)
            chunk_weights = X.sum(axis=1).A1 + 1e-6
        except:
            chunk_weights = np.ones(len(chunks))
    else:
        chunk_weights = np.ones(len(chunks))
    
    # Generate embeddings for chunks
    chunk_embs = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    
    # Aggregate to file embedding using weighted average
    file_emb = (chunk_embs * chunk_weights[:, None]).sum(axis=0) / chunk_weights.sum()
    
    if centroids.shape[0] == 0:
        return {"label": "no_centroids", "score": 0.0, "path": path}
    
    # Find best matching centroid
    sims = cosine_similarity(file_emb.reshape(1, -1), centroids).ravel()
    best_i = int(sims.argmax())
    best_score = float(sims[best_i])
    
    if best_score < similarity_threshold:
        return {"label": "review", "score": best_score, "path": path}
    else:
        return {"label": int(best_i), "score": best_score, "path": path}

# ---------- Enhanced Clarifile Integration ----------
class EnhancedCategorizer:
    """
    Enhanced categorizer that integrates with Clarifile's existing architecture.
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.centroids = None
        self.tfidf_vectorizer = None
        self.cluster_terms = {}
        
    def load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            print(f"Loading embedding model ({self.model_name})...")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully!")
    
    def load_saved_model(self, model_dir):
        """Load previously saved centroids and metadata."""
        self.load_model()
        
        centroids_path = os.path.join(model_dir, "centroids.npy")
        metadata_path = os.path.join(model_dir, "classification_metadata.json")
        tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        
        if os.path.exists(centroids_path):
            self.centroids = np.load(centroids_path)
            print(f"Loaded {len(self.centroids)} centroids")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                print(f"Loaded metadata for {metadata.get('total_files', 0)} files")
        
        if os.path.exists(tfidf_path):
            import joblib
            self.tfidf_vectorizer = joblib.load(tfidf_path)
            print("Loaded TF-IDF vectorizer")
    
    def categorize_content(self, content, content_type="document"):
        """
        Enhanced content categorization using the new pipeline.
        Maintains compatibility with existing Clarifile API.
        """
        if not content.strip():
            return "Uncategorized: General"
        
        self.load_model()
        
        # If we have saved centroids, use incremental classification
        if self.centroids is not None:
            # Create a temporary file-like object for classification
            chunks = chunk_text(content, chunk_size=2000, overlap=200)
            if not chunks:
                return "Uncategorized: General"
            
            # Use TF-IDF weights if available
            if self.tfidf_vectorizer:
                try:
                    X = self.tfidf_vectorizer.transform(chunks)
                    chunk_weights = X.sum(axis=1).A1 + 1e-6
                except:
                    chunk_weights = np.ones(len(chunks))
            else:
                chunk_weights = np.ones(len(chunks))
            
            # Generate embeddings and aggregate
            chunk_embs = self.model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
            file_emb = (chunk_embs * chunk_weights[:, None]).sum(axis=0) / chunk_weights.sum()
            
            # Find best matching centroid
            sims = cosine_similarity(file_emb.reshape(1, -1), self.centroids).ravel()
            best_i = int(sims.argmax())
            best_score = float(sims[best_i])
            
            if best_score > 0.6:  # Threshold for confident classification
                # Map back to category name if available
                if hasattr(self, 'cluster_terms') and best_i in self.cluster_terms:
                    terms = self.cluster_terms[best_i]
                    return f"Category_{best_i}: {' '.join(terms[:2])}"
                else:
                    return f"Category_{best_i}: Clustered"
        
        # Fallback to semantic analysis (existing method)
        return self._analyze_content_semantically(content[:2000], content.lower())
    
    def _analyze_content_semantically(self, content, content_lower):
        """Enhanced semantic analysis with better categorization."""
        
        # Academic/Research Paper Detection
        academic_indicators = {
            'structure': ['abstract', 'introduction', 'methodology', 'results', 'discussion', 'conclusion', 'references'],
            'keywords': ['research', 'study', 'analysis', 'experiment', 'hypothesis', 'findings', 'literature review'],
            'academic_terms': ['peer-reviewed', 'journal', 'conference', 'publication', 'citation', 'doi'],
        }
        
        academic_score = 0
        for category, terms in academic_indicators.items():
            matches = sum(1 for term in terms if term in content_lower)
            academic_score += matches * (2 if category == 'structure' else 1)
        
        if academic_score >= 3:
            if any(term in content_lower for term in ['software', 'programming', 'algorithm', 'computer']):
                return "Computer Science: Research Paper"
            elif any(term in content_lower for term in ['medical', 'clinical', 'patient', 'treatment']):
                return "Medical: Research Paper"
            elif any(term in content_lower for term in ['business', 'management', 'marketing', 'finance']):
                return "Business: Research Paper"
            else:
                return "Academic: Research Paper"
        
        # Financial Document Detection
        financial_terms = ['invoice', 'bill', 'payment', 'amount', 'total', 'due', 'budget', 'expense']
        financial_score = sum(1 for term in financial_terms if term in content_lower)
        
        if financial_score >= 2:
            if any(term in content_lower for term in ['invoice', 'bill']):
                return "Finance: Invoice"
            else:
                return "Finance: Documents"
        
        # Business/Work Document Detection
        business_terms = ['meeting', 'minutes', 'agenda', 'project', 'task', 'team', 'report']
        business_score = sum(1 for term in business_terms if term in content_lower)
        
        if business_score >= 2:
            if any(term in content_lower for term in ['meeting', 'minutes']):
                return "Work: Meeting"
            else:
                return "Work: Document"
        
        # Technical Documentation
        tech_terms = ['code', 'function', 'api', 'documentation', 'technical', 'programming']
        tech_score = sum(1 for term in tech_terms if term in content_lower)
        
        if tech_score >= 2:
            return "Technical: Documentation"
        
        # Personal Documents
        personal_terms = ['personal', 'diary', 'journal', 'note', 'private']
        personal_score = sum(1 for term in personal_terms if term in content_lower)
        
        if personal_score >= 1:
            return "Personal: Notes"
        
        return "General: Document"

# ---------- driver ----------
def main(args):
    src = args.source
    dst = args.dest
    os.makedirs(dst, exist_ok=True)

    # 1) collect files
    all_files = []
    for root, _, fnames in os.walk(src):
        for f in fnames:
            all_files.append(os.path.join(root, f))
    all_files = sorted(all_files)
    
    if not all_files:
        print("No files found")
        return

    print(f"Found {len(all_files)} files to categorize")

    # 2) build embeddings (chunking + tfidf-weighted aggregation)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    file_embeddings, file_chunks, all_chunks, vec_chunks = build_file_embeddings(
        all_files, model, chunk_size=args.chunk_size, overlap=args.overlap
    )

    # Filter out files with no embeddings
    valid_files = []
    valid_embeddings = []
    valid_chunks = []
    
    for i, (f, emb, chunks) in enumerate(zip(all_files, file_embeddings, file_chunks)):
        if np.any(emb != 0):  # File has valid embedding
            valid_files.append(f)
            valid_embeddings.append(emb)
            valid_chunks.append(chunks)
    
    if not valid_files:
        print("No valid files with extractable content found")
        return
    
    valid_embeddings = np.array(valid_embeddings)
    print(f"Processing {len(valid_files)} files with valid content")

    # 3) prepare doc text per file (join chunks back)
    docs_per_file = [" ".join(chs) if chs else "" for chs in valid_chunks]

    # 4) cluster
    labels, km = cluster_embeddings(
        valid_embeddings, 
        use_hdbscan=args.use_hdbscan, 
        k=(args.k if args.k > 0 else None)
    )
    
    print("Cluster distribution:", {int(x): int((labels == x).sum()) for x in set(labels)})

    # 5) name clusters
    cluster_terms = name_clusters_by_tfidf(docs_per_file, labels, top_n=3)
    print("Cluster terms:", cluster_terms)

    # 6) copy files into cluster folders
    mapping = {}
    for fpath, lab in zip(valid_files, labels):
        if lab == -1:
            labname = "noise_uncategorized"
        else:
            terms = cluster_terms.get(lab, ["cat"])
            labname = f"cat{int(lab)}_" + "_".join(terms[:3])
        
        folder = os.path.join(dst, labname)
        os.makedirs(folder, exist_ok=True)
        shutil.copy2(fpath, folder)
        mapping[fpath] = labname

    # 7) save metadata and centroids for incremental classify
    with open(os.path.join(dst, "categories_map.json"), "w", encoding="utf8") as fh:
        json.dump(mapping, fh, indent=2)
    
    np.save(os.path.join(dst, "file_embeddings.npy"), valid_embeddings)
    save_model_outputs(dst, valid_files, labels, valid_embeddings, clusterer=km, tfidf_vectorizer=vec_chunks)
    
    print(f"Done. Results saved to {dst}")
    print(f"Categorized {len(valid_files)} files into {len(set(labels))} categories")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced document categorization with chunking and TF-IDF weighting")
    parser.add_argument("--source", "-s", required=True, help="Source directory with files to categorize")
    parser.add_argument("--dest", "-d", required=True, help="Destination directory for categorized output")
    parser.add_argument("--k", type=int, default=0, help="Force K for KMeans (0=auto)")
    parser.add_argument("--chunk_size", type=int, default=2000, help="Chunk size for text splitting")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks")
    parser.add_argument("--use_hdbscan", action="store_true", help="Use UMAP+HDBSCAN when available")
    args = parser.parse_args()
    main(args)
