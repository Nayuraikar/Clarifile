#This module provides search functionality including visual search, text search and search with typos

import os
import re
import json
import hashlib
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import spacy
import torch
import textdistance
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util
from difflib import get_close_matches

logger = logging.getLogger(__name__)

def load_spacy_model():
    """Load spaCy model with proper error handling and download if needed."""
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("Successfully loaded spaCy model in search_utils")
        return nlp
    except OSError:
        try:
            logger.info("Downloading spaCy model 'en_core_web_sm' for search_utils...")
            import subprocess
            import sys
            # ensure the model is properly installed in the environment
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
            logger.info("Successfully downloaded and loaded spaCy model in search_utils")
            return nlp
        except Exception as e:
            logger.error(f"Failed to download or load spaCy model in search_utils: {str(e)}")
            return None

#NLP components
nlp = load_spacy_model()
if nlp is None:
    logger.warning("spaCy model not available in search_utils. Some features may be limited.")
    from spacy.lang.en import English
    nlp = English()  

model = SentenceTransformer('all-MiniLM-L6-v2')

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
model = model.to(device)

# Cache configuration
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.search_cache')
EMBEDDING_CACHE_DIR = os.path.join(CACHE_DIR, 'embeddings')
CHUNK_CACHE_DIR = os.path.join(CACHE_DIR, 'chunks')
SUMMARY_CACHE_DIR = os.path.join(CACHE_DIR, 'summaries')
CACHE_EXPIRY_DAYS = 30 

# cache directories
for directory in [EMBEDDING_CACHE_DIR, CHUNK_CACHE_DIR, SUMMARY_CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# File type mappings
FILE_TYPES = {
    'document': ['document', 'doc', 'docx', 'text', 'txt', 'rtf', 'odt'],
    'spreadsheet': ['spreadsheet', 'excel', 'xls', 'xlsx', 'csv', 'ods'],
    'presentation': ['presentation', 'powerpoint', 'ppt', 'pptx', 'key', 'odp'],
    'image': ['image', 'photo', 'picture', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg', 'webp'],
    'video': ['video', 'movie', 'mp4', 'mov', 'avi', 'mkv', 'webm', 'wmv'],
    'audio': ['audio', 'music', 'sound', 'mp3', 'wav', 'ogg', 'm4a', 'flac'],
    'archive': ['archive', 'zip', 'rar', '7z', 'tar', 'gz', 'bz2'],
    'code': ['code', 'program', 'script', 'py', 'js', 'jsx', 'ts', 'tsx', 'java', 'c', 'cpp', 'h', 'hpp', 'cs', 'go', 'rb', 'php', 'swift', 'kt'],
    'pdf': ['pdf'],
    'folder': ['folder', 'directory']
}

def get_cache_key(text: str, prefix: str = '') -> str:
    #Generate a cache key for the given text and prefix
    return hashlib.md5(f"{prefix}_{text}".encode('utf-8')).hexdigest()

def get_cache_path(cache_dir: str, key: str) -> str:
    #Get the full cache file path for a given key
    return os.path.join(cache_dir, f"{key}.pkl")

def is_cache_valid(cache_path: str) -> bool:
    #Check if a cache file exists and hasn't expired
    if not os.path.exists(cache_path):
        return False
    
    cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    return datetime.now() - cache_time < timedelta(days=CACHE_EXPIRY_DAYS)

def load_from_cache(cache_path: str) -> Any:
    #Load data from cache file
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except (pickle.PickleError, EOFError, FileNotFoundError):
        return None

def save_to_cache(data: Any, cache_path: str) -> None:
    #Save data to cache file
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    except (pickle.PickleError, IOError) as e:
        logger.warning(f"Failed to save cache: {e}")

def clear_old_cache() -> None:
    #Remove cache files older than the expiry time
    for cache_dir in [EMBEDDING_CACHE_DIR, CHUNK_CACHE_DIR, SUMMARY_CACHE_DIR]:
        if not os.path.exists(cache_dir):
            continue
            
        for filename in os.listdir(cache_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(cache_dir, filename)
                if not is_cache_valid(filepath):
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass

clear_old_cache()

class EmbeddingCache:
    #Cache for document embeddings    
    @staticmethod
    def get_embedding(text: str, model) -> np.ndarray:
        key = get_cache_key(text, 'emb')
        cache_path = get_cache_path(EMBEDDING_CACHE_DIR, key)
        
        if is_cache_valid(cache_path):
            cached = load_from_cache(cache_path)
            if cached is not None:
                return cached
        embedding = model.encode(text, convert_to_tensor=False)
        save_to_cache(embedding, cache_path)
        return embedding

class ChunkCache:
    #Cache for document chunks    
    @staticmethod
    def get_chunks(text: str, chunk_func, *args, **kwargs) -> List[str]:
        key = get_cache_key(text, 'chunk')
        cache_path = get_cache_path(CHUNK_CACHE_DIR, key)
        
        if is_cache_valid(cache_path):
            cached = load_from_cache(cache_path)
            if cached is not None:
                return cached
        
        chunks = list(chunk_func(text, *args, **kwargs))
        save_to_cache(chunks, cache_path)
        return chunks

class SummaryCache:
    #Cache for document summaries    
    @staticmethod
    def get_summary(text: str, summary_func, *args, **kwargs) -> str:
        key = get_cache_key(text, 'sum')
        cache_path = get_cache_path(SUMMARY_CACHE_DIR, key)
        
        if is_cache_valid(cache_path):
            cached = load_from_cache(cache_path)
            if cached is not None:
                return cached
        summary = summary_func(text, *args, **kwargs)
        if summary: 
            save_to_cache(summary, cache_path)
        return summary

def preprocess_query(query: str) -> str:
    #Preprocess the search query by lowercasing and removing extra whitespace
    return ' '.join(query.lower().split())

def extract_file_types(query: str) -> Tuple[List[str], str]:
    #Extract file types from query if specified.
    query_lower = query.lower()
    cleaned_query = query
    found_types = set()
    
    ext_pattern = r'\.([a-z0-9]+)(?:\s|$)'
    for match in re.finditer(ext_pattern, query_lower):
        ext = match.group(1).lower()
        ext_mapping = {
            'doc': 'docx',
            'ppt': 'pptx',
            'jpg': 'jpeg',
            'tif': 'tiff'
        }
        ext = ext_mapping.get(ext, ext)
        found_types.add(f'.{ext}')
    
    type_keywords = {
        'pdf': ['.pdf'],
        'text': ['.txt'],
        'word': ['.docx', '.doc'],
        'excel': ['.xlsx', '.xls', '.csv'],
        'spreadsheet': ['.xlsx', '.xls', '.csv', '.ods'],
        'powerpoint': ['.pptx', '.ppt'],
        'presentation': ['.pptx', '.ppt', '.key', '.odp'],
        'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp'],
        'photo': ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
        'video': ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv'],
        'audio': ['.mp3', '.wav', '.ogg', '.m4a', '.flac'],
        'archive': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'],
        'code': ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp', '.cs', '.go', '.rb', '.php', '.swift', '.kt']
    }
    
    for keyword, extensions in type_keywords.items():
        if re.search(rf'\b{re.escape(keyword)}\b', query_lower):
            found_types.update(extensions)
    
    for ext in found_types:
        cleaned_query = re.sub(
            rf'(?i)\b{re.escape(ext[1:])}\b', 
            '',
            cleaned_query
        ).strip()
    
    # Remove common file type phrases
    cleaned_query = re.sub(
        r'\b(?:file|document|type|format)(?:s|es)?\b',
        '',
        cleaned_query,
        flags=re.IGNORECASE
    ).strip()
    
    cleaned_query = ' '.join(cleaned_query.split())
    
    return list(found_types), cleaned_query

def extract_search_terms(query: str) -> List[str]:
    # Remove stop words and get lemmas
    doc = nlp(query)
    terms = []
    
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            # Use lemma for normalization
            term = token.lemma_.lower()
            if len(term) > 2: 
                terms.append(term)
    
    return terms

def get_semantic_embedding(text: str) -> np.ndarray:
    return model.encode([text])[0]

def find_similar_terms(term: str, vocabulary: List[str], top_n: int = 3) -> List[str]:
    return get_close_matches(term, vocabulary, n=top_n, cutoff=0.7)

def expand_query_with_synonyms(query: str) -> str:
    # Simple synonym mapping - can be expanded
    synonyms = {
        'doc': 'document',
        'file': 'document',
        'img': 'image',
        'pic': 'image',
        'photo': 'image',
        'ppt': 'presentation',
        'sheet': 'spreadsheet',
        'xls': 'excel',
        'movie': 'video',
        'song': 'audio',
        'music': 'audio',
        'zip': 'archive',
        'code': 'source',
        'src': 'source',
        'folder': 'directory'
    }
    
    words = query.split()
    expanded = []
    
    for word in words:
        expanded.append(word)
        if word in synonyms:
            expanded.append(synonyms[word])
    
    return ' '.join(expanded)

def fuzzy_match(term: str, text: str, threshold: float = 0.8) -> bool:
    #Check if term is in text with fuzzy matching
    from difflib import SequenceMatcher
    
    if term.lower() in text.lower():
        return True
    
    words = re.findall(r'\w+', text.lower())
    term_words = re.findall(r'\w+', term.lower())
    
    if len(term_words) > 1:
        return all(any(SequenceMatcher(None, tw, w).ratio() >= threshold 
                      for w in words) for tw in term_words)
    
    term = term_words[0]
    return any(SequenceMatcher(None, term, w).ratio() >= threshold for w in words)

def parse_negations(query: str) -> Tuple[List[str], str]:
    # Patterns to match negations
    patterns = [
        r'(?:\bnot\b|\bwithout\b|\bexclude\b|\bexcept\b|\bno\b|\-\s*)(?:\s+the\s+)?([^,.]+?)(?=\s+(?:and|or|but|,|$))',
        r'\bexclud(?:ing|es|e)\s+([^,.]+?)(?=\s+(?:and|or|but|,|$))',
    ]
    
    neg_terms = []
    cleaned_query = query
    
    for pattern in patterns:
        matches = re.finditer(pattern, query, flags=re.IGNORECASE)
        for match in matches:
            term = match.group(1).strip()
            if term and len(term) > 1: 
                neg_terms.append(term)
                cleaned_query = cleaned_query.replace(match.group(0), '')
    
    # Clean up the query
    cleaned_query = ' '.join(cleaned_query.split())
    return neg_terms, cleaned_query

def filter_by_file_type(documents: List[Dict[str, Any]], file_types: List[str]) -> List[Dict[str, Any]]:
    
    #Filter documents by file extension with support for fuzzy matching and common aliases.

    if not file_types or not documents:
        return documents
    
    normalized_ft = {ft.lstrip('.').lower() for ft in file_types}
    
    alias_map = {
        'doc': 'docx',
        'ppt': 'pptx',
        'jpg': 'jpeg',
        'tif': 'tiff',
        'text': 'txt',
        'ps': 'postscript',
        'xls': 'xlsx',
        'mpeg': 'mpg',
        'jpeg': 'jpg', 
        'tiff': 'tif', 
        'docx': 'doc', 
        'pptx': 'ppt',  
        'xlsx': 'xls'  
    }
    
    # Add aliases to the file types set
    additional_ft = set()
    for ft in normalized_ft:
        if ft in alias_map:
            additional_ft.add(alias_map[ft])
    normalized_ft.update(additional_ft)
    
    filtered = []
    for doc in documents:
        if not isinstance(doc, dict) or 'metadata' not in doc:
            continue
            
        # Get file extension from metadata or try to extract from filename
        ext = ''
        metadata = doc.get('metadata', {})
        
        if 'file_extension' in metadata:
            ext = str(metadata['file_extension']).lstrip('.').lower()
        elif 'filename' in metadata:
            filename = str(metadata['filename']).lower()
            if '.' in filename:
                ext = filename.rsplit('.', 1)[1].lower()
        
        if ext and (ext in normalized_ft or 
                   any(alias == ext for alias in alias_map.get(ext, [])) or
                   any(alias_map.get(a) == ext for a in alias_map if a in normalized_ft)):
            filtered.append(doc)
    
    return filtered

def filter_negations(documents: List[Dict[str, Any]], neg_terms: List[str]) -> List[Dict[str, Any]]:
    
    #Filter out documents containing negated terms with fuzzy matching.
    if not neg_terms or not documents:
        return documents
    neg_terms = list({term.lower().strip() for term in neg_terms if term.strip()})
    if not neg_terms:
        return documents
    exact_patterns = []
    fuzzy_terms = []
    
    for term in neg_terms:
        if re.match(r'^\w+$', term):
            exact_patterns.append(re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE))
        else:
            fuzzy_terms.append(term)
    
    filtered = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue
            
        text = ''
        if 'text' in doc:
            text = str(doc['text'])
        
        if 'metadata' in doc and isinstance(doc['metadata'], dict):
            filename = doc['metadata'].get('filename', '')
            if filename:
                text = f"{filename} {text}"
        
        text = text.lower()
        
        has_negation = any(pattern.search(text) for pattern in exact_patterns)
        
        if not has_negation and fuzzy_terms:
            has_negation = any(fuzzy_match(term, text) for term in fuzzy_terms)
        
        if not has_negation:
            filtered.append(doc)
    
    return filtered

def parse_search_query(query: str) -> Dict[str, Any]:
    
    if not query or not isinstance(query, str):
        return {
            'original_query': '',
            'cleaned_query': '',
            'file_types': [],
            'negation_terms': [],
            'search_terms': [],
            'semantic_embedding': np.array([]),
            'expanded_query': '',
            'has_filters': False
        }
    original_query = query.strip()
    file_types, query_text = extract_file_types(original_query)
    neg_terms, query_text = parse_negations(query_text)
    cleaned_query = ' '.join(query_text.split()) 
    search_terms = extract_search_terms(cleaned_query)
    try:
        semantic_embedding = get_semantic_embedding(cleaned_query or ' ')
    except Exception as e:
        logger.error(f"Error generating semantic embedding: {e}")
        semantic_embedding = np.array([])
    expanded_query = cleaned_query
    if search_terms:
        try:
            expanded_query = expand_query_with_synonyms(cleaned_query)
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
    
    has_filters = len(file_types) > 0 or len(neg_terms) > 0
    return {
        'original_query': original_query,
        'cleaned_query': cleaned_query,
        'file_types': file_types,
        'negation_terms': neg_terms,
        'search_terms': search_terms,
        'semantic_embedding': semantic_embedding,
        'expanded_query': expanded_query,
        'has_filters': has_filters
    }
class QueryExpander:    
    #A class to handle query expansion and enhancement for semantic search.    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info(f"Initialized QueryExpander with model '{model_name}' on device '{self.device}'")
    
    def get_synonyms(self, word: str, pos: str = None) -> List[str]:
        try:
            synsets = wordnet.synsets(word, pos=pos) if pos else wordnet.synsets(word)
            synonyms = set()
            
            for syn in synsets:
                for lemma in syn.lemmas():
                    # Only add single-word synonyms that are alphanumeric
                    synonym = lemma.name().replace('_', ' ').lower()
                    if synonym.isalpha() and synonym != word.lower():
                        synonyms.add(synonym)
            
            return list(synonyms)
        except Exception as e:
            logger.warning(f"Error getting synonyms for '{word}': {str(e)}")
            return []
    
    def expand_query_terms(self, query_terms: List[str]) -> Dict[str, List[str]]:
        expanded = {}
        for term in query_terms:
            # Skip very short terms
            if len(term) < 3:
                expanded[term] = []
                continue
                
            # Get noun and verb synonyms
            synonyms = set()
            synonyms.update(self.get_synonyms(term, 'n'))  # noun synonyms
            synonyms.update(self.get_synonyms(term, 'v'))  # verb synonyms
            
            # Filter out synonyms that are too different in length
            max_len_diff = 2
            filtered_synonyms = [
                s for s in synonyms 
                if abs(len(s) - len(term)) <= max_len_diff
            ]
            
            expanded[term] = filtered_synonyms
            
        return expanded
    
    def find_similar_terms(self, term: str, corpus_terms: List[str], 
                          threshold: float = 0.85) -> List[Tuple[str, float]]:
        similar = []
        for t in corpus_terms:
            if t.lower() == term.lower():
                continue
                
            similarity = textdistance.jaro_winkler(term.lower(), t.lower())
            if similarity >= threshold:
                similar.append((t, similarity))
        
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar
    
    def expand_query(self, query: str, max_expansions: int = 5) -> Dict[str, Any]:
        # Tokenize and clean the query
        terms = [t.lower() for t in re.findall(r'\b\w+\b', query)]
        
        expanded_terms = self.expand_query_terms(terms)
        
        all_terms = set(terms)
        for term in terms:
            similar = self.find_similar_terms(term, list(all_terms))
            for t, _ in similar[:max_expansions]:
                all_terms.add(t)
        term_list = list(all_terms)
        if not term_list:
            return {
                'original_query': query,
                'expanded_terms': {},
                'expanded_queries': [query],
                'embeddings': None
            }
        
        with torch.no_grad():
            embeddings = self.model.encode(
                term_list, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
        
        return {
            'original_query': query,
            'expanded_terms': expanded_terms,
            'expanded_queries': term_list,
            'embeddings': embeddings
        }
    
    def semantic_search_with_expansion(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        min_score: float = 0.2,
        chunk_size: int = 3,
        overlap: int = 1,
        apply_filters: bool = True
    ) -> List[Dict[str, Any]]:
        
        from services.parser.semantic_search import semantic_search
        expanded = self.expand_query(query)
        if not expanded['expanded_queries'] or expanded['expanded_queries'] == [query]:
            return semantic_search(
                query=query,
                documents=documents,
                top_k=top_k,
                min_score=min_score,
                chunk_size=chunk_size,
                overlap=overlap,
                apply_filters=apply_filters
            )
        all_results = []
        for q in expanded['expanded_queries']:
            results = semantic_search(
                query=q,
                documents=documents,
                top_k=top_k * 2,  
                min_score=min_score,
                chunk_size=chunk_size,
                overlap=overlap,
                apply_filters=apply_filters
            )
            all_results.extend(results)
        seen_docs = set()
        unique_results = []
        
        for result in sorted(all_results, key=lambda x: x['score'], reverse=True):
            doc_id = result.get('metadata', {}).get('id', '')
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                unique_results.append(result)
                if len(unique_results) >= top_k:
                    break
        
        return unique_results
# Global instance 
try:
    query_expander = QueryExpander()
except Exception as e:
    logger.error(f"Failed to initialize QueryExpander: {str(e)}")
    query_expander = None
