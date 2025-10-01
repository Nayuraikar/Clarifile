import re
import logging
import spacy
from typing import Tuple, Optional, List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from difflib import get_close_matches

# Set up logging
logger = logging.getLogger(__name__)

def load_spacy_model():
    """Load spaCy model with proper error handling and download if needed."""
    try:
        # First try loading the model normally
        nlp = spacy.load("en_core_web_sm")
        logger.info("Successfully loaded spaCy model in search_utils")
        return nlp
    except OSError:
        try:
            logger.info("Downloading spaCy model 'en_core_web_sm' for search_utils...")
            import subprocess
            import sys
            # Use subprocess to ensure the model is properly installed in the environment
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
            logger.info("Successfully downloaded and loaded spaCy model in search_utils")
            return nlp
        except Exception as e:
            logger.error(f"Failed to download or load spaCy model in search_utils: {str(e)}")
            return None

# Initialize spaCy model
nlp = load_spacy_model()
if nlp is None:
    logger.warning("spaCy model not available in search_utils. Some features may be limited.")
    # Create a minimal nlp object with just the tokenizer
    from spacy.lang.en import English
    nlp = English()  # This provides basic tokenization but no statistical model

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# File type mappings
FILE_TYPES = {
    'document': ['document', 'doc', 'docx', 'text', 'txt', 'pdf', 'rtf', 'odt'],
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

def preprocess_query(query: str) -> str:
    """Preprocess the search query by lowercasing and removing extra whitespace."""
    return ' '.join(query.lower().split())

def extract_file_types(query: str) -> Tuple[List[str], str]:
    """
    Extract file types from query if specified.
    Returns (list_of_file_types, cleaned_query)
    """
    query_lower = query.lower()
    cleaned_query = query
    found_types = set()
    
    # First, look for explicit file extensions (e.g., .pdf, .txt)
    ext_pattern = r'\.([a-z0-9]+)(?:\s|$)'
    for match in re.finditer(ext_pattern, query_lower):
        ext = match.group(1).lower()
        # Map common aliases to standard extensions
        ext_mapping = {
            'doc': 'docx',
            'ppt': 'pptx',
            'jpg': 'jpeg',
            'tif': 'tiff'
        }
        ext = ext_mapping.get(ext, ext)
        found_types.add(f'.{ext}')
    
    # Then look for file type keywords (e.g., "pdf files", "word documents")
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
    
    # Remove file type terms from the query
    for ext in found_types:
        cleaned_query = re.sub(
            rf'(?i)\b{re.escape(ext[1:])}\b',  # Remove without the dot
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
    
    # Clean up any extra spaces
    cleaned_query = ' '.join(cleaned_query.split())
    
    return list(found_types), cleaned_query

def extract_search_terms(query: str) -> List[str]:
    """Extract meaningful search terms from the query."""
    # Remove stop words and get lemmas
    doc = nlp(query)
    terms = []
    
    for token in doc:
        # Skip stop words, punctuation, and spaces
        if not token.is_stop and not token.is_punct and not token.is_space:
            # Use lemma for normalization
            term = token.lemma_.lower()
            if len(term) > 2:  # Skip very short terms
                terms.append(term)
    
    return terms

def get_semantic_embedding(text: str) -> np.ndarray:
    """Get the semantic embedding for a text."""
    return model.encode([text])[0]

def find_similar_terms(term: str, vocabulary: List[str], top_n: int = 3) -> List[str]:
    """Find similar terms using string similarity."""
    return get_close_matches(term, vocabulary, n=top_n, cutoff=0.7)

def expand_query_with_synonyms(query: str) -> str:
    """Expand query with synonyms for better matching."""
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
    """Check if term is in text with fuzzy matching."""
    from difflib import SequenceMatcher
    
    # Simple direct match first for performance
    if term.lower() in text.lower():
        return True
    
    # Check for partial matches in longer text
    words = re.findall(r'\w+', text.lower())
    term_words = re.findall(r'\w+', term.lower())
    
    # If it's a multi-word term, check for all words existing in text
    if len(term_words) > 1:
        return all(any(SequenceMatcher(None, tw, w).ratio() >= threshold 
                      for w in words) for tw in term_words)
    
    # For single word terms, check for best match
    term = term_words[0]
    return any(SequenceMatcher(None, term, w).ratio() >= threshold for w in words)

def parse_negations(query: str) -> Tuple[List[str], str]:
    """Extract negation terms from query."""
    # Patterns to match negations
    patterns = [
        r'(?:\bnot\b|\bwithout\b|\bexclude\b|\bexcept\b|\bno\b|\b-\s*)(?:\s+the\s+)?([^,.]+?)(?=\s+(?:and|or|but|,|$))',
        r'\bexclud(?:ing|es|e)\s+([^,.]+?)(?=\s+(?:and|or|but|,|$))',
    ]
    
    neg_terms = []
    cleaned_query = query
    
    for pattern in patterns:
        matches = re.finditer(pattern, query, flags=re.IGNORECASE)
        for match in matches:
            term = match.group(1).strip()
            if term and len(term) > 1:  # Ignore single character terms
                neg_terms.append(term)
                # Remove the negation term from the query
                cleaned_query = cleaned_query.replace(match.group(0), '')
    
    # Clean up the query
    cleaned_query = ' '.join(cleaned_query.split())
    return neg_terms, cleaned_query

def filter_by_file_type(documents: List[Dict[str, Any]], file_types: List[str]) -> List[Dict[str, Any]]:
    """
    Filter documents by file extension with support for fuzzy matching and common aliases.
    
    Args:
        documents: List of document dictionaries with metadata
        file_types: List of file extensions (with or without leading dot)
        
    Returns:
        Filtered list of documents matching any of the specified file types
    """
    if not file_types or not documents:
        return documents
    
    # Normalize file types (remove leading dots and convert to lowercase)
    normalized_ft = {ft.lstrip('.').lower() for ft in file_types}
    
    # Map common aliases to standard extensions
    alias_map = {
        'doc': 'docx',
        'ppt': 'pptx',
        'jpg': 'jpeg',
        'tif': 'tiff',
        'text': 'txt',
        'ps': 'postscript',
        'xls': 'xlsx',
        'mpeg': 'mpg',
        'jpeg': 'jpg',  # Add reverse mapping
        'tiff': 'tif',  # Add reverse mapping
        'docx': 'doc',  # Add reverse mapping
        'pptx': 'ppt',  # Add reverse mapping
        'xlsx': 'xls'   # Add reverse mapping
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
        
        # Try to get extension from file_extension field first
        if 'file_extension' in metadata:
            ext = str(metadata['file_extension']).lstrip('.').lower()
        # Otherwise extract from filename
        elif 'filename' in metadata:
            filename = str(metadata['filename']).lower()
            if '.' in filename:
                ext = filename.rsplit('.', 1)[1].lower()
        
        # Check for direct match or alias match
        if ext and (ext in normalized_ft or 
                   any(alias == ext for alias in alias_map.get(ext, [])) or
                   any(alias_map.get(a) == ext for a in alias_map if a in normalized_ft)):
            filtered.append(doc)
    
    return filtered

def filter_negations(documents: List[Dict[str, Any]], neg_terms: List[str]) -> List[Dict[str, Any]]:
    """
    Filter out documents containing negated terms with fuzzy matching.
    
    Args:
        documents: List of document dictionaries with text and metadata
        neg_terms: List of terms that should not appear in the documents
        
    Returns:
        Filtered list of documents that don't contain any of the negated terms
    """
    if not neg_terms or not documents:
        return documents
    
    # Pre-process negation terms (lowercase and remove duplicates)
    neg_terms = list({term.lower().strip() for term in neg_terms if term.strip()})
    if not neg_terms:
        return documents
    
    # Compile regex patterns for exact matching (faster than fuzzy for exact matches)
    exact_patterns = []
    fuzzy_terms = []
    
    for term in neg_terms:
        # Check if term is a simple word (alphanumeric with no spaces)
        if re.match(r'^\w+$', term):
            exact_patterns.append(re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE))
        else:
            fuzzy_terms.append(term)
    
    filtered = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue
            
        # Get text content from document
        text = ''
        if 'text' in doc:
            text = str(doc['text'])
        
        # Also include filename in the search text
        if 'metadata' in doc and isinstance(doc['metadata'], dict):
            filename = doc['metadata'].get('filename', '')
            if filename:
                text = f"{filename} {text}"
        
        text = text.lower()
        
        # First check for exact matches using regex (faster)
        has_negation = any(pattern.search(text) for pattern in exact_patterns)
        
        # Only do fuzzy matching if no exact matches found and we have fuzzy terms
        if not has_negation and fuzzy_terms:
            has_negation = any(fuzzy_match(term, text) for term in fuzzy_terms)
        
        if not has_negation:
            filtered.append(doc)
    
    return filtered

def parse_search_query(query: str) -> Dict[str, Any]:
    """
    Parse the search query and extract structured information with enhanced file type
    and negation handling.
    
    Args:
        query: The raw search query string
        
    Returns:
        Dictionary containing:
        {
            'original_query': str,
            'cleaned_query': str,  # Query with file types and negations removed
            'file_types': List[str],  # Extracted file extensions (without leading dots)
            'negation_terms': List[str],  # Terms to exclude from results
            'search_terms': List[str],  # Normalized search terms
            'semantic_embedding': np.ndarray,  # Vector representation of the query
            'expanded_query': str,  # Query expanded with synonyms
            'has_filters': bool  # True if file types or negations were found
        }
    """
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
    
    # Store original query
    original_query = query.strip()
    
    # Step 1: Extract file types first (before cleaning)
    file_types, query_text = extract_file_types(original_query)
    
    # Step 2: Extract negation terms
    neg_terms, query_text = parse_negations(query_text)
    
    # Step 3: Clean the remaining query text
    cleaned_query = ' '.join(query_text.split())  # Normalize whitespace
    
    # Step 4: Extract search terms from cleaned query
    search_terms = extract_search_terms(cleaned_query)
    
    # Step 5: Generate semantic embedding (empty array if generation fails)
    try:
        semantic_embedding = get_semantic_embedding(cleaned_query or ' ')
    except Exception as e:
        print(f"Error generating semantic embedding: {e}")
        semantic_embedding = np.array([])
    
    # Step 6: Expand query with synonyms if we have search terms
    expanded_query = cleaned_query
    if search_terms:
        try:
            expanded_query = expand_query_with_synonyms(cleaned_query)
        except Exception as e:
            print(f"Error expanding query: {e}")
    
    # Determine if any filters were applied
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
