#!/usr/bin/env python3
"""
smart_categorizer.py
Smart content-based categorization using embeddings and clustering.
Integrates with Clarifile's existing parser service.
"""

import os
import json
import shutil
import argparse
from tqdm import tqdm
import numpy as np

# Text & file libs
import pdfplumber
import docx
from PIL import Image
import pytesseract

# Embedding & clustering
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# Optional: if you installed pdf2image and want OCR fallback for scanned PDFs
try:
    from pdf2image import convert_from_path
    PDF2IMAGE = True
except Exception:
    PDF2IMAGE = False

class SmartCategorizer:
    """Smart content-based file categorizer using embeddings and clustering."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the categorizer with the embedding model."""
        self.model_name = model_name
        self.model = None
        self.embeddings_cache = {}

    def load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            print("DEBUG: Loading embedding model (all-MiniLM-L6-v2) ...")
            try:
                self.model = SentenceTransformer(self.model_name)
                print("DEBUG: Model loaded successfully!")
            except Exception as e:
                print(f"DEBUG: Error loading model: {e}")
                raise

    def extract_text(self, path, max_chars=15000, ocr_if_empty=True):
        """Extract text from a file based on its type."""
        ext = os.path.splitext(path)[1].lower()
        text = ""

        try:
            if ext in (".txt", ".md", ".csv", ".log"):
                with open(path, "r", encoding="utf8", errors="ignore") as f:
                    text = f.read()
            elif ext == ".pdf":
                try:
                    with pdfplumber.open(path) as pdf:
                        pages_text = [p.extract_text() or "" for p in pdf.pages]
                        text = "\n".join(pages_text).strip()
                except Exception:
                    text = ""
                # fallback to OCR if empty and pdf2image available
                if (not text) and ocr_if_empty and PDF2IMAGE:
                    imgs = convert_from_path(path)
                    text = "\n".join(pytesseract.image_to_string(img) for img in imgs)
            elif ext in (".docx",):
                doc = docx.Document(path)
                text = "\n".join([p.text for p in doc.paragraphs])
            elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
                text = pytesseract.image_to_string(Image.open(path))
            else:
                # try reading first chunk as text
                with open(path, "rb") as f:
                    raw = f.read(200000)
                    try:
                        text = raw.decode("utf8", errors="ignore")
                    except:
                        text = ""
        except Exception as e:
            print(f"Warning reading {path}: {e}")

        text = " ".join(text.split())
        return text[:max_chars]

    def choose_k_auto(self, embeddings, min_k=2, max_k=10):
        """Automatically choose the optimal number of clusters using silhouette score."""
        n = len(embeddings)
        if n < 3:
            return 1 if n==1 else 2

        max_k = min(max_k, n-1)
        best_k = None
        best_score = -1.0

        for k in range(min_k, max_k+1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(embeddings)
            try:
                score = silhouette_score(embeddings, labels)
            except:
                score = -1
            if score > best_score:
                best_score = score
                best_k = k

        return best_k or min_k

    def top_terms_by_cluster(self, docs, labels, top_n=3):
        """Generate cluster names based on top TF-IDF terms."""
        vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))
        X = vec.fit_transform(docs)
        terms = np.array(vec.get_feature_names_out())
        cluster_terms = {}

        for c in sorted(set(labels)):
            idxs = [i for i,l in enumerate(labels) if l==c]
            if not idxs:
                cluster_terms[c] = ["empty"]
                continue
            mean_tfidf = X[idxs].mean(axis=0).A1
            top_idx = mean_tfidf.argsort()[::-1][:top_n]
            cluster_terms[c] = [t.replace(" ", "_") for t in terms[top_idx].tolist()]

        return cluster_terms

    def categorize_files(self, file_paths, output_dir, k=None):
        """
        Categorize files based on their content.

        Args:
            file_paths: List of file paths to categorize
            output_dir: Directory to save categorized files
            k: Number of clusters (optional, auto-determined if None)

        Returns:
            Dictionary mapping file paths to category names
        """
        self.load_model()

        # Extract text from files
        docs = []
        good_files = []

        print("Extracting text from files...")
        for path in tqdm(file_paths):
            text = self.extract_text(path)
            if text.strip():
                docs.append(text)
                good_files.append(path)
            else:
                print(f"Skipped (no text): {path}")

        if not docs:
            print("No extractable text found.")
            return {}

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.model.encode(docs, show_progress_bar=True, convert_to_numpy=True)

        # Choose number of clusters
        if k and k > 0:
            k = min(k, len(docs))
        else:
            print("Choosing number of clusters automatically...")
            k = self.choose_k_auto(embeddings, min_k=2, max_k=10)
            print(f"Chosen k = {k}")

        # Perform clustering
        print("Clustering embeddings with KMeans ...")
        if k == 1:
            labels = [0] * len(docs)
            km = None
        else:
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings)
            labels = km.labels_

        # Generate cluster names
        cluster_terms = self.top_terms_by_cluster(docs, labels, top_n=3)

        # Create output directories and copy files
        os.makedirs(output_dir, exist_ok=True)
        mapping = {}

        print("Saving categorized files...")
        for path, label in zip(good_files, labels):
            terms = cluster_terms.get(label, ["cat"])
            folder_name = f"cat{label}_" + "_".join(terms[:3])
            folder_path = os.path.join(output_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            # Copy file to category folder
            shutil.copy2(path, folder_path)
            mapping[path] = folder_name

        # Save metadata
        metadata = {
            "mapping": mapping,
            "cluster_terms": cluster_terms,
            "k": k,
            "total_files": len(good_files)
        }

        with open(os.path.join(output_dir, "categorization_metadata.json"), "w", encoding="utf8") as f:
            json.dump(metadata, f, indent=2)

        # Save embeddings for future incremental classification
        np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)

        print("Done. Categories saved to:", output_dir)
        print("Cluster -> top terms:", cluster_terms)

        return mapping

    def categorize_content(self, content, content_type="document"):
        """
        Categorize a single piece of content using transformer-based semantic analysis.

        Args:
            content: Text content to categorize
            content_type: Type of content (document, text, etc.)

        Returns:
            Category name in format "Category: Subcategory"
        """
        print(f"DEBUG: categorize_content called with content length: {len(content) if content else 0}")
        
        if not content.strip():
            print("DEBUG: No content provided, returning Uncategorized")
            return "Uncategorized: General"

        try:
            self.load_model()
        except Exception as e:
            print(f"DEBUG: Failed to load model, using fallback: {e}")
            return self._keyword_based_categorization(content.lower())
        
        # Clean and prepare content for analysis
        analysis_text = content.strip()[:2000]  # Use first 2000 chars for analysis
        text_lower = analysis_text.lower()
        
        print(f"DEBUG: Analyzing content: {analysis_text[:100]}...")
        
        # Generate semantic embedding for the content
        try:
            embedding = self.model.encode([analysis_text], convert_to_numpy=True)
            print(f"DEBUG: Generated embedding with shape: {embedding.shape}")
        except Exception as e:
            print(f"DEBUG: Error generating embedding: {e}")
            # Fall back to keyword-based analysis
            return self._keyword_based_categorization(text_lower)
        
        # Advanced content analysis using multiple signals
        category = self._analyze_content_semantically(analysis_text, text_lower, embedding)
        
        print(f"DEBUG: Final categorization: {category}")
        return category
    
    def _analyze_content_semantically(self, content, content_lower, embedding):
        """
        Perform semantic analysis using multiple signals to determine category.
        """
        
        # 1. Academic/Research Paper Detection (Enhanced)
        academic_indicators = {
            'structure': ['abstract', 'introduction', 'methodology', 'results', 'discussion', 'conclusion', 'references'],
            'keywords': ['research', 'study', 'analysis', 'experiment', 'hypothesis', 'findings', 'literature review'],
            'academic_terms': ['peer-reviewed', 'journal', 'conference', 'publication', 'citation', 'doi'],
            'research_methods': ['survey', 'interview', 'statistical', 'qualitative', 'quantitative', 'empirical']
        }
        
        academic_score = 0
        for category, terms in academic_indicators.items():
            matches = sum(1 for term in terms if term in content_lower)
            academic_score += matches * (2 if category == 'structure' else 1)
        
        if academic_score >= 3:
            # Determine specific academic field
            if any(term in content_lower for term in ['software', 'programming', 'algorithm', 'computer', 'machine learning', 'ai', 'neural']):
                return "Computer Science: Research Paper"
            elif any(term in content_lower for term in ['medical', 'clinical', 'patient', 'treatment', 'diagnosis']):
                return "Medical: Research Paper"
            elif any(term in content_lower for term in ['business', 'management', 'marketing', 'finance', 'economics']):
                return "Business: Research Paper"
            elif any(term in content_lower for term in ['psychology', 'behavior', 'cognitive', 'mental']):
                return "Psychology: Research Paper"
            else:
                return "Academic: Research Paper"
        
        # 2. Financial Document Detection
        financial_indicators = {
            'invoice_terms': ['invoice', 'bill', 'payment', 'amount', 'total', 'due', 'customer'],
            'financial_terms': ['budget', 'expense', 'revenue', 'profit', 'cost', 'financial'],
            'accounting_terms': ['debit', 'credit', 'balance', 'account', 'transaction']
        }
        
        financial_score = 0
        for category, terms in financial_indicators.items():
            matches = sum(1 for term in terms if term in content_lower)
            financial_score += matches * (3 if category == 'invoice_terms' else 1)
        
        if financial_score >= 2:
            if any(term in content_lower for term in ['invoice', 'bill']):
                return "Finance: Invoice"
            elif any(term in content_lower for term in ['budget', 'expense']):
                return "Finance: Budget"
            else:
                return "Finance: Documents"
        
        # 3. Business/Work Document Detection
        business_indicators = {
            'meeting_terms': ['meeting', 'minutes', 'agenda', 'attendees', 'action items'],
            'work_terms': ['project', 'task', 'deadline', 'team', 'manager', 'report'],
            'corporate_terms': ['company', 'department', 'employee', 'policy', 'procedure']
        }
        
        business_score = 0
        for category, terms in business_indicators.items():
            matches = sum(1 for term in terms if term in content_lower)
            business_score += matches * (3 if category == 'meeting_terms' else 1)
        
        if business_score >= 2:
            if any(term in content_lower for term in ['meeting', 'minutes', 'agenda']):
                return "Work: Meeting"
            elif any(term in content_lower for term in ['report', 'analysis']):
                return "Work: Report"
            else:
                return "Work: Document"
        
        # 4. Legal Document Detection
        legal_indicators = ['contract', 'agreement', 'terms', 'legal', 'law', 'clause', 'party', 'liability']
        legal_score = sum(1 for term in legal_indicators if term in content_lower)
        
        if legal_score >= 2:
            return "Legal: Contract"
        
        # 5. Technical Documentation Detection
        tech_indicators = {
            'programming': ['code', 'function', 'class', 'variable', 'algorithm', 'programming'],
            'documentation': ['api', 'documentation', 'manual', 'guide', 'tutorial'],
            'technical': ['technical', 'specification', 'architecture', 'design', 'implementation']
        }
        
        tech_score = 0
        for category, terms in tech_indicators.items():
            matches = sum(1 for term in terms if term in content_lower)
            tech_score += matches
        
        if tech_score >= 2:
            return "Technical: Documentation"
        
        # 6. Personal Document Detection
        personal_indicators = ['personal', 'diary', 'journal', 'note', 'reminder', 'todo', 'private']
        personal_score = sum(1 for term in personal_indicators if term in content_lower)
        
        if personal_score >= 1:
            return "Personal: Notes"
        
        # 7. Use semantic similarity for final classification
        # This is where the transformer embedding really helps
        return self._semantic_similarity_classification(content, embedding)
    
    def _semantic_similarity_classification(self, content, embedding):
        """
        Use semantic similarity to classify content against known categories.
        """
        # Define category prototypes with example content
        category_prototypes = {
            "Academic: Research Paper": "research study methodology results analysis findings academic paper",
            "Finance: Documents": "invoice payment bill financial budget expense cost money",
            "Work: Meeting": "meeting agenda minutes discussion team project work business",
            "Legal: Contract": "contract agreement legal terms conditions law clause party",
            "Technical: Documentation": "technical documentation code programming software development",
            "Personal: Notes": "personal notes diary journal thoughts ideas private"
        }
        
        try:
            # Generate embeddings for category prototypes
            prototype_texts = list(category_prototypes.values())
            prototype_embeddings = self.model.encode(prototype_texts, convert_to_numpy=True)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(embedding, prototype_embeddings)[0]
            
            # Find best match
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            
            category_names = list(category_prototypes.keys())
            best_category = category_names[best_match_idx]
            
            print(f"Best semantic match: {best_category} (similarity: {best_similarity:.3f})")
            
            # Only use semantic match if similarity is high enough
            if best_similarity > 0.3:
                return best_category
                
        except Exception as e:
            print(f"Error in semantic similarity classification: {e}")
        
        # Final fallback
        return self._keyword_based_categorization(content.lower())
    
    def _keyword_based_categorization(self, text_lower):
        """
        Fallback keyword-based categorization.
        """
        # Enhanced keyword matching with scoring
        category_keywords = {
            "Academic: Research Paper": ['academic', 'research', 'paper', 'study', 'thesis', 'analysis', 'methodology'],
            "Computer Science: Technical": ['software', 'programming', 'code', 'algorithm', 'computer', 'technical'],
            "Work: Meeting": ['meeting', 'minutes', 'agenda', 'report', 'business', 'team'],
            "Finance: Documents": ['invoice', 'payment', 'financial', 'budget', 'bill', 'cost', 'expense'],
            "Legal: Contracts": ['contract', 'legal', 'agreement', 'terms', 'law', 'clause']
        }
        
        best_category = "General: Document"
        best_score = 0
        
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category if best_score > 0 else "General: Document"

# Standalone usage
def main():
    """Main function for standalone usage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s", required=True, help="Source folder with files")
    parser.add_argument("--dest", "-d", required=True, help="Destination folder for categorized output")
    parser.add_argument("--k", type=int, default=0, help="(Optional) number of clusters. Omit for auto")
    args = parser.parse_args()

    categorizer = SmartCategorizer()

    # Gather files
    files = []
    for root, _, fnames in os.walk(args.source):
        for f in fnames:
            path = os.path.join(root, f)
            files.append(path)
    files = sorted(files)

    if not files:
        print("No files found in", args.source)
        return

    categorizer.categorize_files(files, args.dest, args.k)

if __name__ == "__main__":
    main()
