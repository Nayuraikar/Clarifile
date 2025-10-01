#!/usr/bin/env python3
"""
smart_categorizer.py
Smart content-based categorization using embeddings and clustering.
Integrates with Clarifile's existing parser service.
"""

import os
import re
import json
import numpy as np
import argparse
import joblib
import warnings
import mimetypes

# Text & file processing
import pdfplumber
import docx
from PIL import Image
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='`resume_download`')  # Suppress specific download warning
warnings.filterwarnings('ignore', category=UserWarning)  # Suppress other user warnings

# Check for PDF to image conversion support
try:
    PDF2IMAGE = True
except Exception:
    PDF2IMAGE = False

class SmartCategorizer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the categorizer with the embedding model."""
        self.model_name = model_name
        self.model = None

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
        try:
            # Check if file exists
            if not os.path.exists(path):
                print(f"File not found: {path}")
                return ""

            # Check file size
            file_size = os.path.getsize(path)
            if file_size == 0:
                return ""

            # Get file extension and MIME type
            _, ext = os.path.splitext(path.lower())
            mime_type, _ = mimetypes.guess_type(path)
            
            # Enhanced Audio/Video file handling
            if mime_type and (mime_type.startswith(('audio/', 'video/')) or ext in ['.mp3', '.wav', '.ogg', '.m4a', '.aac', '.mp4', '.mov', '.avi', '.mkv']):
                file_info = {
                    'type': 'audio' if (mime_type and mime_type.startswith('audio/')) or ext in ['.mp3', '.wav', '.ogg', '.m4a', '.aac'] else 'video',
                    'mime_type': mime_type or 'application/octet-stream',
                    'size': file_size,
                    'name': os.path.basename(path),
                    'path': path,
                    'extension': ext,
                    'last_modified': os.path.getmtime(path)
                }
                print(f"DEBUG: Created file info for {path}: {file_info}")
                return json.dumps(file_info, indent=2)  # Pretty print for better debugging

            # Handle specific document types
            if ext == '.docx':
                try:
                    doc = docx.Document(path)
                    return "\n".join([p.text for p in doc.paragraphs])
                except Exception as e:
                    print(f"Error reading DOCX {path}: {e}")
                    return ""

            # Handle images with OCR if available
            if ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
                if PDF2IMAGE:
                    try:
                        img = Image.open(path)
                        return pytesseract.image_to_string(img)[:max_chars]
                    except Exception as e:
                        print(f"OCR failed for {path}: {e}")
                return ""

            # Text files
            if mime_type and mime_type.startswith('text/'):
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        return f.read(max_chars)
                except Exception as e:
                    print(f"Error reading text file {path}: {e}")
                    return ""

            # Try to read as binary text
            try:
                with open(path, 'rb') as f:
                    content = f.read(max_chars)
                    try:
                        return content.decode('utf-8', errors='ignore')
                    except UnicodeDecodeError:
                        return ""
            except Exception as e:
                print(f"Error reading binary file {path}: {e}")
                return ""
                
        except Exception as e:
            print(f"Unexpected error processing {path}: {e}")
            return ""

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
        print(f"\n=== DEBUG: categorize_content called ===")
        print(f"Content type: {content_type}")
        print(f"Content length: {len(content) if content else 0}")
        print(f"First 100 chars: {content[:100] if content else 'N/A'}")
        
        if not content or not content.strip():
            print("DEBUG: No content provided, returning Uncategorized")
            return "Uncategorized: General"

        # Check if content is a JSON string (for audio/video files)
        try:
            file_info = json.loads(content)
            if isinstance(file_info, dict):
                print("DEBUG: Detected JSON content, processing as file metadata...")
                return self._analyze_content_semantically(content, content.lower(), None)
        except (json.JSONDecodeError, TypeError):
            pass  # Not a JSON string, continue with normal processing

        try:
            print("DEBUG: Loading model for semantic analysis...")
            self.load_model()
            print("DEBUG: Model loaded successfully")
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
        # Check if content is a JSON string with file metadata
        try:
            file_info = json.loads(content)
            if isinstance(file_info, dict):
                file_type = file_info.get('type', '').lower()
                mime_type = file_info.get('mime_type', '').lower()
                file_name = file_info.get('name', '').lower()
                
                print(f"DEBUG: Processing file metadata - Type: {file_type}, MIME: {mime_type}, Name: {file_name}")
                
                if file_type == 'audio' or 'audio/' in mime_type or any(ext in file_name for ext in ['.mp3', '.wav', '.ogg', '.m4a', '.aac']):
                    # Try to determine audio type from filename
                    if any(term in file_name for term in ['meeting', 'call', 'interview', 'conversation']):
                        return "Media: Audio - Meeting/Interview"
                    elif any(term in file_name for term in ['lecture', 'presentation', 'talk']):
                        return "Media: Audio - Lecture/Presentation"
                    elif any(term in file_name for term in ['music', 'song', 'track']):
                        return "Media: Audio - Music"
                    else:
                        return "Media: Audio File"
                elif file_type == 'video' or 'video/' in mime_type or any(ext in file_name for ext in ['.mp4', '.mov', '.avi', '.mkv']):
                    return "Media: Video File"
                elif file_type == 'binary' or not mime_type.startswith(('text/', 'application/')):
                    return "File: Binary Data"
                
                print(f"DEBUG: File metadata processed but no specific type matched: {file_info}")
                
        except (json.JSONDecodeError, TypeError) as e:
            print(f"DEBUG: Not a JSON string or invalid format: {e}")
            pass  # Not a JSON string, continue with normal processing
        
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
        
        # 6. Scientific/Chemical Analysis Detection (Enhanced)
        scientific_indicators = {
            'analysis_terms': ['analysis', 'analyze', 'analyzed', 'analysed', 'qualitative', 'quantitative', 'assay', 'test', 'examination', 'evaluation'],
            'scientific_terms': ['sulphur', 'sulfur', 'chemical', 'compound', 'element', 'sample', 'specimen', 'laboratory', 'lab', 'research', 'study', 'data'],
            'methods': ['spectroscopy', 'chromatography', 'titration', 'reaction', 'synthesis', 'preparation', 'reconstruction', 'imaging', 'microscopy', 'spectrometry'],
            'measurements': ['concentration', 'purity', 'yield', 'percentage', 'ratio', 'measurement', 'matrix', 'structural', 'parameter'],
            'visualization': ['photo', 'image', 'micrograph', 'micrography', 'microscopy', 'sketch', 'diagram', 'figure', 'graph', 'plot']
        }
        
        scientific_score = 0
        matched_terms = set()
        
        for category, terms in scientific_indicators.items():
            category_matches = [term for term in terms if term in content_lower]
            matches_count = len(category_matches)
            matched_terms.update(category_matches)
            
            # Weight different categories differently
            if category == 'analysis_terms':
                scientific_score += matches_count * 1.5
            elif category in ['scientific_terms', 'methods']:
                scientific_score += matches_count * 1.3
            else:
                scientific_score += matches_count
        
        print(f"DEBUG: Scientific analysis score: {scientific_score:.1f}")
        print(f"DEBUG: Matched scientific terms: {', '.join(matched_terms) if matched_terms else 'None'}")
        
        if scientific_score >= 2:  # Lowered threshold to catch more scientific content
            # Enhanced type detection
            if any(term in content_lower for term in ['sulphur', 'sulfur']):
                if any(term in content_lower for term in ['photo', 'image', 'micrograph', 'sketch']):
                    return "Science: Imaging - Sulphur Analysis"
                return "Science: Chemical Analysis - Sulphur"
            elif any(term in content_lower for term in ['photo', 'image', 'micrograph', 'sketch']):
                if any(term in content_lower for term in ['reconstruct', 'reconstruction', '3d']):
                    return "Science: Reconstructed Imaging Analysis"
                return "Science: Image Analysis"
            elif any(term in content_lower for term in ['matrix', 'structural', 'quantitative']):
                return "Science: Structural Analysis"
            elif any(term in content_lower for term in ['organic', 'compound', 'molecule']):
                return "Science: Organic Chemical Analysis"
            elif any(term in content_lower for term in ['metal', 'inorganic', 'mineral']):
                return "Science: Inorganic Chemical Analysis"
            else:
                return "Science: General Analysis"
        
        # 7. Personal Document Detection
        personal_indicators = ['personal', 'diary', 'journal', 'note', 'reminder', 'todo', 'private']
        personal_score = sum(1 for term in personal_indicators if term in content_lower)
        
        if personal_score >= 1:
            return "Personal: Notes"
        
        # 8. Use semantic similarity for final classification
        # This is where the transformer embedding really helps
        return self._semantic_similarity_classification(content, embedding)
    
    def _semantic_similarity_classification(self, content, embedding):
        """
        Use semantic similarity to classify content against known categories.
        """
        # Define category prototypes with example content (Enhanced with more specific categories)
        category_prototypes = {
            "Academic: Research Paper": "research study methodology results analysis findings academic paper",
            "Science: Chemical Analysis - Sulphur": "sulphur sulfur chemical analysis experiment lab test results data compound quantitative measurement",
            "Science: Chemical Analysis - Organic": "organic compound molecule analysis chemical structure formula synthesis reaction",
            "Science: Chemical Analysis - Inorganic": "inorganic metal mineral analysis chemical composition structure properties",
            "Science: Imaging Analysis": "photo image micrograph microscopy analysis visualization structure composition",
            "Science: Reconstructed Imaging": "3d reconstruction imaging analysis model structure visualization tomography",
            "Science: Structural Analysis": "matrix structural analysis quantitative measurement parameters properties material",
            "Science: Laboratory Report": "laboratory experiment test results methodology analysis data findings conclusion",
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
            print(f"DEBUG: Best semantic match: {best_category} (similarity: {best_similarity:.3f})")
            
            # Adjust threshold based on content length and confidence
            threshold = 0.25  # Lower threshold to catch more cases
            if len(content.strip().split()) < 20:  # For very short content
                threshold = 0.2
                
            if best_similarity > threshold:
                return best_category
                
        except Exception as e:
            print(f"Error in semantic similarity classification: {e}")
        
        # Final fallback
        return self._keyword_based_categorization(content.lower())
    
    def _keyword_based_categorization(self, text_lower):
        """
        Enhanced keyword-based categorization with better scientific term coverage.
        """
        # Enhanced keyword matching with scoring and more specific categories
        category_keywords = {
            "Academic: Research Paper": ['academic', 'research', 'paper', 'study', 'thesis', 'analysis', 'methodology', 'literature'],
            "Science: Chemical Analysis - Sulphur": ['sulphur', 'sulfur', 'sulfate', 'sulphate', 'thiol', 'thio', 'sulfide'],
            "Science: Chemical Analysis - Organic": ['organic', 'compound', 'molecule', 'polymer', 'carbon', 'hydrocarbon'],
            "Science: Chemical Analysis - Inorganic": ['inorganic', 'metal', 'alloy', 'mineral', 'oxide', 'salt'],
            "Science: Imaging Analysis": ['photo', 'image', 'micrograph', 'microscopy', 'microscope', 'visualization'],
            "Science: Reconstructed Imaging": ['reconstruct', 'reconstruction', '3d', 'tomography', 'volume', 'render'],
            "Science: Structural Analysis": ['matrix', 'structural', 'quantitative', 'morphology', 'crystal', 'lattice'],
            "Science: Laboratory Report": ['experiment', 'laboratory', 'lab', 'results', 'data', 'findings', 'conclusion', 'method', 'procedure', 'observation', 'measurement'],
            "Science: General Analysis": ['analysis', 'analyze', 'analyzed', 'analysed', 'qualitative', 'quantitative', 'assay', 'test', 'examination', 'evaluation', 'characterization'],
            "Computer Science: Technical": ['software', 'programming', 'code', 'algorithm', 'computer', 'technical', 'computation', 'simulation'],
            "Work: Meeting": ['meeting', 'minutes', 'agenda', 'report', 'business', 'team', 'project', 'discussion'],
            "Finance: Documents": ['invoice', 'payment', 'financial', 'budget', 'bill', 'cost', 'expense', 'transaction'],
            "Legal: Contracts": ['contract', 'legal', 'agreement', 'terms', 'law', 'clause', 'party', 'signature']
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
