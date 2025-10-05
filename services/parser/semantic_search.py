from typing import List, Dict, Any, Optional, Tuple, Union, Literal, Set, Callable
import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from enum import Enum
import torch
from collections import defaultdict
import logging
import functools
import textdistance
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Import the query expander using absolute import
from services.parser.search_utils_combined import QueryExpander, query_expander
from services.parser.search_utils_combined import EmbeddingCache, ChunkCache

def smart_chunks(
    text: str, 
    max_words: int = 5, 
    min_length: int = 1,
    **kwargs
) -> List[str]:
    """
    Extract meaningful chunks from text using spaCy's noun chunks.
    Falls back to simple chunking if spaCy fails or no noun chunks are found.
    
    Args:
        text: Input text to extract chunks from
        max_words: Maximum number of words in a chunk (default: 5)
        min_length: Minimum number of words in a chunk (default: 1)
        **kwargs: Additional keyword arguments (ignored, for compatibility)
        
    Returns:
        List of extracted chunks, or [text] if no valid chunks found
    """
    # Input validation
    if not text or not isinstance(text, str) or not text.strip():
        return [text] if text else []
        
    # Ensure min_length is at least 1 and max_words is sensible
    min_length = max(1, int(min_length))
    max_words = max(min_length, min(20, int(max_words)))  # Cap at 20 words
    
    try:
        # Process text with spaCy
        doc = nlp(text)
        chunks = []
        seen_chunks = set()  # For deduplication
        
        # Extract noun chunks and filter by length
        for chunk in doc.noun_chunks:
            words = [t.text for t in chunk if not t.is_punct and not t.is_space]
            if min_length <= len(words) <= max_words:
                chunk_text = ' '.join(words)
                if chunk_text and chunk_text not in seen_chunks:
                    chunks.append(chunk_text)
                    seen_chunks.add(chunk_text)
        
        # Fallback to simple chunking if no noun chunks found
        if not chunks:
            words = [t.text for t in doc if not t.is_punct and not t.is_space]
            if words:
                chunks = [' '.join(words[i:i + max_words]) 
                         for i in range(0, len(words), max_words)]
        
        return chunks if chunks else [text]
    
    except Exception as e:
        # Log a cleaner error message if it's a spaCy-specific error
        error_msg = str(e)
        if 'E029' in error_msg or 'noun_chunks' in error_msg:
            logger.debug("spaCy model not properly loaded for noun chunks. Using simple chunking.")
        else:
            logger.warning(f"Error in smart_chunks: {error_msg[:200]}")
        
        # Fallback to simple whitespace-based chunking
        words = text.split()
        return [' '.join(words[i:i + max_words]) 
                for i in range(0, len(words), max_words)]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data with fallback for punkt/punkt_tab
def ensure_nltk_resources():
    """Ensure required NLTK resources are available with fallback handling."""
    resources_to_download = []
    
    # Check punkt_tab (new) or punkt (old) for sentence tokenization
    punkt_available = False
    for punkt_name in ['punkt_tab', 'punkt']:
        try:
            nltk.data.find(f'tokenizers/{punkt_name}')
            punkt_available = True
            logger.info(f"Found NLTK resource: {punkt_name}")
            break
        except LookupError:
            continue
    
    if not punkt_available:
        # Try to download punkt_tab first, then punkt as fallback
        for punkt_name in ['punkt_tab', 'punkt']:
            try:
                logger.info(f"Downloading NLTK resource: {punkt_name}")
                nltk.download(punkt_name, quiet=True)
                punkt_available = True
                break
            except Exception as e:
                logger.warning(f"Failed to download {punkt_name}: {e}")
                continue
        
        if not punkt_available:
            logger.error("Failed to download punkt tokenizer. Sentence tokenization may not work properly.")
    
    # Check other required resources
    for resource in ['stopwords', 'wordnet']:
        try:
            if resource == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif resource == 'wordnet':
                nltk.data.find('corpora/wordnet')
        except LookupError:
            try:
                logger.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download {resource}: {e}")

# Initialize NLTK resources
ensure_nltk_resources()

# Initialize stop words
STOP_WORDS = set(stopwords.words('english'))

# Device configuration
device = "mps" if torch.backends.mps.is_available() else "cpu"
if device == "mps" and not torch.backends.mps.is_available():
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model with device
try:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    logger.info(f"Using device: {device}")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise

def load_spacy_model(disable=None):
    """
    Load spaCy model with proper error handling and download if needed.
    
    Args:
        disable: List of pipeline components to disable (default: None, loads all components)
    """
    try:
        # First try loading the model with all components
        nlp = spacy.load("en_core_web_sm", disable=disable or [])
        logger.info("Successfully loaded spaCy model")
        return nlp
    except OSError:
        try:
            logger.info("Downloading spaCy model 'en_core_web_sm'...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm", disable=disable or [])
            logger.info("Successfully downloaded and loaded spaCy model")
            return nlp
        except Exception as e:
            logger.error(f"Failed to download or load spaCy model: {str(e)}")
            return None

# Initialize spaCy model with all components for noun chunking
nlp = load_spacy_model()
if nlp is None:
    logger.warning("Falling back to simple tokenization. Some features may be limited.")
    # Create a minimal fallback tokenizer if spaCy fails to load
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    logger.info("Initialized minimal spaCy tokenizer as fallback")

class SearchOperator(Enum):
    OR = "or"
    AND = "and"
    DEFAULT = "or"

def retry_on_failure(max_retries: int = 3, default_return=None):
    """Decorator to retry a function on failure."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        logger.error(f"All {max_retries} attempts failed")
                        return default_return
        return wrapper
    return decorator

class QueryIntent:
    def __init__(self, text: str, operator: SearchOperator = None):
        self.original_text = text or ""
        self.text = self._clean_text(self.original_text)
        self.operator = operator or self._detect_operator()
        self.negation_terms = self._extract_negation_terms()
        self.key_phrases = self._extract_key_phrases()
        self.topics = self._extract_topics()
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize the query text."""
        if not text or not isinstance(text, str):
            return ""
            
        # Convert to lowercase and normalize whitespace
        text = ' '.join(text.lower().split())
        
        # Remove common search phrases and noise
        text = re.sub(
            r"(?:find|show|me|documents?|files?|about|related to|that contain|talks? about|"
            r"contains? info(?:rmation)? about|what is|search for|look for|get|please|can you|would you|"
            r"give me|i need|i want|return|locate|the|a|an|that|which|where|what|when|how|is|are|was|were|do|does|did|can|could|will|would|should|has|have|had)\b",
            " ",
            text,
            flags=re.IGNORECASE
        )
        
        # Remove special characters except spaces, hyphens, and quotes
        text = re.sub(r"[^\w\s'-]", ' ', text)
        
        # Remove extra whitespace and normalize
        return ' '.join(text.split())
        
    def _detect_operator(self) -> SearchOperator:
        """Detect if the query uses AND or OR logic."""
        text = f" {self.text.lower()} "
        and_count = text.count(" and ")
        or_count = text.count(" or ")
        
        if and_count > or_count:
            return SearchOperator.AND
        return SearchOperator.OR
    
    @retry_on_failure(default_return=[])
    def _extract_negation_terms(self) -> List[str]:
        """Extract negation terms from the query."""
        if not self.text:
            return []
            
        negation_terms = []
        negation_patterns = [
            r'\bnot\s+(\w+)',  # not word
            r'\bno\s+(\w+)',   # no word
            r'\bwithout\s+(\w+)',  # without word
            r'\bexclude\s+(?:the\s+)?(\w+)',  # exclude (the) word
            r'\b-\s*(\w+)'  # -word (minus sign syntax)
        ]
        
        for pattern in negation_patterns:
            matches = re.finditer(pattern, self.text, re.IGNORECASE)
            for match in matches:
                term = match.group(1).lower()
                if len(term) > 2 and term not in STOP_WORDS:  # Skip very short terms and stopwords
                    negation_terms.append(term)
        
        return list(set(negation_terms))  # Remove duplicates

    def _extract_key_phrases(self) -> List[str]:
        """Extract key phrases using NLP with multiple fallback strategies."""
        if not self.text or not isinstance(self.text, str):
            return []
            
        # Remove negation terms from the text before extracting key phrases
        clean_text = self.text.lower()
        for term in self.negation_terms:
            clean_text = re.sub(r'\b' + re.escape(term) + r'\b', '', clean_text)
        
        phrases = set()
        
        # Try using spaCy if available
        if nlp is not None:
            try:
                doc = nlp(clean_text)
                # Extract noun phrases
                for chunk in doc.noun_chunks:
                    phrase = chunk.text.lower().strip()
                    if len(phrase.split()) > 1:  # Only consider multi-word phrases
                        phrases.add(phrase)
                
                # Extract named entities
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                        phrases.add(ent.text.lower())
                        
            except Exception as e:
                logger.warning(f"spaCy phrase extraction failed: {str(e)}")
        
        # Fallback 1: Simple n-gram extraction
        try:
            words = [w for w in word_tokenize(clean_text) if w.isalnum()]
            
            # Add bigrams
            for i in range(len(words) - 1):
                if words[i] not in STOP_WORDS and words[i+1] not in STOP_WORDS:
                    phrase = f"{words[i]} {words[i+1]}"
                    if len(phrase) > 3 and not any(nt in phrase for nt in self.negation_terms):
                        phrases.add(phrase)
            
            # Add trigrams
            for i in range(len(words) - 2):
                if (words[i] not in STOP_WORDS and 
                    words[i+2] not in STOP_WORDS and
                    words[i+1] not in STOP_WORDS):
                    phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                    if len(phrase) > 6 and not any(nt in phrase for nt in self.negation_terms):
                        phrases.add(phrase)
        except Exception as e:
            logger.warning(f"N-gram extraction failed: {str(e)}")
        
        # Fallback 2: Simple word-based extraction
        if not phrases:
            words = [w for w in word_tokenize(clean_text) 
                    if w.isalpha() and w not in STOP_WORDS and len(w) > 2]
            phrases.update(words)
        
        # Remove any phrases that contain negation terms
        filtered_phrases = []
        for phrase in phrases:
            if not any(nt in phrase.lower() for nt in self.negation_terms):
                filtered_phrases.append(phrase)
        
        return filtered_phrases[:15]  # Increased limit to 15 for better coverage
        
    def _extract_topics(self) -> List[str]:
        """Extract distinct topics from the query with improved phrase handling."""
        if not self.text:
            return []
            
        # First, extract key phrases
        topics = self.key_phrases.copy()
        
        # Then process the remaining text
        remaining_text = self.text
        for phrase in topics:
            remaining_text = remaining_text.replace(phrase, '')
            
        # Split remaining text on operators and clean
        for part in re.split(r"\s*(?:or|and|,)\s*", remaining_text):
            part = re.sub(r"[^\w\s-]", ' ', part).strip()
            words = [w for w in word_tokenize(part) if w.lower() not in STOP_WORDS and len(w) > 1]
            if words:
                topics.append(' '.join(words))
        
        # If no meaningful topics found, use the cleaned query
        if not topics and self.text:
            words = [w for w in word_tokenize(self.text) if w.lower() not in STOP_WORDS and len(w) > 1]
            if words:
                topics.append(' '.join(words))
        
        return list(set(topics))
    
    def get_search_operator(self) -> SearchOperator:
        return self.operator
    
    def __str__(self):
        return f"QueryIntent(text='{self.text}', operator={self.operator}, topics={self.topics}, key_phrases={self.key_phrases})"

def clean_query(query: str) -> str:
    """
    Clean and preprocess the search query by removing noise and normalizing.
    Returns a cleaned version of the query.
    """
    if not query or not isinstance(query, str):
        return ""
    
    # Create a QueryIntent to handle cleaning
    intent = QueryIntent(query)
    return intent.text

def preprocess_text(text: str, chunk_size: int = 3, overlap: int = 1) -> List[str]:
    """
    Split text into overlapping chunks of sentences for better context.
    
    Args:
        text: Input text to be chunked
        chunk_size: Number of sentences per chunk
        overlap: Number of sentences to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
        
    # Split into sentences
    sentences = sent_tokenize(text)
    
    # Handle case with fewer sentences than chunk size
    if len(sentences) <= chunk_size:
        return [' '.join(sentences)] if sentences else []
    
    # Create overlapping chunks
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(sentences) - overlap, step):
        chunk = sentences[i:i + chunk_size]
        chunks.append(' '.join(chunk))
    
    return chunks

def process_document(document: Dict[str, Any], chunk_size: int = 3, overlap: int = 1) -> List[Dict[str, Any]]:
    """
    Process a single document into chunks with embeddings and metadata.
    
    Args:
        document: Dictionary containing 'text' and 'metadata' keys
        chunk_size: Number of sentences per chunk
        overlap: Number of sentences to overlap between chunks
        
    Returns:
        List of chunk dictionaries with text, embedding, and metadata
    """
    text = document.get('text', '')
    if not text:
        return []
    
    # Get document metadata
    metadata = document.get('metadata', {})
    doc_id = metadata.get('id', str(id(document)))
    
    # Split document into chunks
    chunks = preprocess_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return []
    
    try:
        # Encode chunks in batches for better performance
        chunk_texts = [f"passage: {chunk}" for chunk in chunks]
        chunk_embeddings = model.encode(
            chunk_texts,
            batch_size=32,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Create chunk objects with metadata and embeddings
        chunk_objects = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            chunk_objects.append({
                'text': chunk_text,
                'embedding': embedding,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'document_id': doc_id,
                'metadata': metadata
            })
            
        return chunk_objects
        
    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {str(e)}")
        return []

@retry_on_failure(default_return=[])
def process_document_chunk(chunk: Dict[str, Any], query_embeddings: torch.Tensor, model) -> Dict[str, Any]:
    """Process a single document chunk and compute similarity scores."""
    try:
        # Get or generate chunk embedding
        chunk_text = chunk.get('text', '')
        chunk_embedding = EmbeddingCache.get_embedding(chunk_text, model)
        
        # Convert to tensor if needed
        if not isinstance(chunk_embedding, torch.Tensor):
            chunk_embedding = torch.tensor(chunk_embedding, device=model.device)
        
        # Calculate similarity with query
        chunk_embedding = chunk_embedding.unsqueeze(0)  # Add batch dimension
        similarities = util.cos_sim(query_embeddings, chunk_embedding)[0]
        
        return {
            'chunk': chunk,
            'similarity': float(torch.max(similarities).item()),
            'scores': similarities.tolist()
        }
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        return None

def semantic_search(
    query: Union[str, QueryIntent], 
    documents: List[Dict[str, Any]],
    top_k: int = 5, 
    min_score: float = 0.2,
    chunk_size: int = 3,
    overlap: int = 1,
    apply_filters: bool = True,
    expand_query: bool = True,
    use_cache: bool = True,
    batch_size: int = 32,
    max_workers: int = 4
) -> List[Dict[str, Any]]:
    """
    Perform semantic search on documents with improved error handling and performance.
    
    Args:
        query: The search query or QueryIntent object
        documents: List of documents with 'text' and 'metadata' keys
        top_k: Maximum number of results to return
        min_score: Minimum similarity score (0-1) for results
        chunk_size: Number of sentences per chunk
        overlap: Number of sentences to overlap between chunks
        apply_filters: Whether to apply file type and negation filters
        expand_query: Whether to expand the query with synonyms and similar terms
        
    Returns:
        List of matching documents with similarity scores and metadata
    """
    
    # Use query expansion if available and enabled
    if expand_query and query_expander and isinstance(query, str):
        return query_expander.semantic_search_with_expansion(
            query=query,
            documents=documents,
            top_k=top_k,
            min_score=min_score,
            chunk_size=chunk_size,
            overlap=overlap,
            apply_filters=apply_filters,
            use_cache=use_cache,
            batch_size=batch_size,
            max_workers=max_workers
        )
        
    if not documents or not query:
        return []
    if not documents or not query:
        return []
        
    from search_utils_combined import parse_search_query, filter_by_file_type, filter_negations
    
    # Get query text and generate embeddings
    if isinstance(query, str):
        query_intent = QueryIntent(query)
    else:
        query_intent = query
        
    query_text = query_intent.text if hasattr(query_intent, 'text') else str(query)
    synonyms = query_intent.get_synonyms() if hasattr(query_intent, 'get_synothers') else []
    
    query_embeddings = model.encode(
        [query_text] + synonyms,
        convert_to_tensor=True,
        show_progress_bar=False
    )
    
    # Process documents in batches
    results = []
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        
        # Process documents in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for doc in batch_docs:
                # Get or generate chunks
                doc_text = doc.get('text', '')
                # Use the smart_chunks function with fallback to simple chunking
                chunks = ChunkCache.get_chunks(
                    doc_text,
                    smart_chunks,  # Will use our fallback if not defined
                    max_words=chunk_size * 100,
                    stride=overlap * 50
                )
                
                # Process each chunk
                for chunk in chunks:
                    futures.append(executor.submit(
                        process_document_chunk,
                        {'text': chunk, 'metadata': doc.get('metadata', {})},
                        query_embeddings,
                        model
                    ))
            
            # Process completed futures
            for future in as_completed(futures):
                result = future.result()
                if result and result['similarity'] >= min_score:
                    results.append(result)
    
    # Sort and return top-k results
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return [{
        'score': r['similarity'],
        'text': r['chunk']['text'],
        'metadata': r['chunk'].get('metadata', {}),
        'scores': r['scores']
    } for r in results[:top_k]]
    
    # Ensure query_intent is properly initialized
    if not hasattr(query_intent, 'text'):
        query_intent = QueryIntent(str(query))
        
    # Get attributes with fallbacks
    query_text = query_intent.text
    key_phrases = getattr(query_intent, 'key_phrases', [])
    file_types = getattr(query_intent, 'file_types', [])
    negation_terms = getattr(query_intent, 'negation_terms', [])
    
    # Apply file type filtering if requested and file types are specified
    if apply_filters and file_types and hasattr(query_intent, 'file_types'):
        from search_utils_combined import filter_by_file_type
        documents = filter_by_file_type(documents, file_types)
        if not documents:
            logger.info(f"No documents match the specified file types: {file_types}")
            return []
    
    # Apply negation filtering at the document level if requested
    if apply_filters and negation_terms:
        from search_utils_combined import filter_negations
        documents = filter_negations(documents, negation_terms)
    
    if not query_text:
        logger.warning("Empty query text")
        return []
    
    # Check if we have topics to search for
    if not getattr(query_intent, 'topics', []) and not getattr(query_intent, 'key_phrases', []):
        logger.warning("No searchable topics or key phrases found in query")
        return []
    
    logger.info(f"Processing query: {query_intent}")
    logger.debug(f"File types filter: {file_types}")
    logger.debug(f"Negation terms: {negation_terms}")
    
    # Prepare query embeddings for all topics and key phrases
    try:
        search_terms = query_intent.topics + query_intent.key_phrases
        topic_embeddings = []
        
        for term in search_terms:
            query_with_prefix = f"query: {term}"
            embedding = model.encode(
                query_with_prefix,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            topic_embeddings.append({
                'embedding': embedding,
                'term': term,
                'is_key_phrase': term in query_intent.key_phrases
            })
    except Exception as e:
        logger.error(f"Error encoding query topics: {str(e)}")
        return []
    
    # Process all documents in batches
    all_chunks = []
    doc_chunk_indices = {}
    
    for doc_idx, doc in enumerate(documents):
        # Skip documents that don't match file type filters
        if apply_filters and file_types:
            doc_ext = ''
            if 'metadata' in doc and 'filename' in doc['metadata']:
                filename = doc['metadata']['filename']
                if '.' in filename:
                    doc_ext = filename.rsplit('.', 1)[1].lower()
            
            if doc_ext and f".{doc_ext}" not in file_types and doc_ext not in file_types:
                continue
        
        # Skip documents containing negation terms
        if apply_filters and negation_terms:
            doc_content = f"{doc.get('text', '')} {doc.get('metadata', {}).get('filename', '')}".lower()
            if any(nt.lower() in doc_content for nt in negation_terms):
                continue
        
        # Process document into chunks
        doc_id = doc.get('metadata', {}).get('id', f'doc_{doc_idx}')
        chunks = process_document(doc, chunk_size=chunk_size, overlap=overlap)
        
        # Add filename to chunk metadata if not present
        for chunk in chunks:
            if 'metadata' not in chunk:
                chunk['metadata'] = {}
            if 'filename' not in chunk['metadata'] and 'filename' in doc.get('metadata', {}):
                chunk['metadata']['filename'] = doc['metadata']['filename']
        
        start_idx = len(all_chunks)
        all_chunks.extend(chunks)
        doc_chunk_indices[doc_id] = (start_idx, len(all_chunks))
    
    if not all_chunks:
        logger.info("No chunks available for search after filtering")
        return []
    
    # Process chunks in batches to save memory
    batch_size = 32
    doc_scores = defaultdict(list)
    
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i+batch_size]
        chunk_embeddings = torch.stack([chunk['embedding'] for chunk in batch_chunks])
        
        # Calculate similarity with each search term
        for term_info in topic_embeddings:
            topic_emb = term_info['embedding']
            topic_idx = search_terms.index(term_info['term'])
            
            # Expand topic embedding to match batch size
            topic_emb_expanded = topic_emb.unsqueeze(0)  # [1, dim]
            
            # Calculate cosine similarity: [1, dim] @ [batch_size, dim].T -> [1, batch_size]
            similarities = util.cos_sim(topic_emb_expanded, chunk_embeddings)[0]
            
            # Update document scores
            for chunk_idx, score in enumerate(similarities):
                chunk = batch_chunks[chunk_idx]
                doc_id = chunk.get('document_id', chunk['metadata'].get('id', ''))
                
                # Skip if this chunk contains any negation terms
                if apply_filters and negation_terms:
                    chunk_content = f"{chunk.get('text', '')} {chunk.get('metadata', {}).get('filename', '')}".lower()
                    if any(nt.lower() in chunk_content for nt in negation_terms):
                        continue
                
                doc_scores[doc_id].append({
                    'score': score.item(),
                    'topic_idx': topic_idx,
                    'chunk': chunk,
                    'is_key_phrase': term_info['is_key_phrase']
                })
    
    # Process results based on search operator
    results = []
    operator = query_intent.operator if hasattr(query_intent, 'operator') else SearchOperator.DEFAULT
    
    for doc_id, scores in doc_scores.items():
        # Group scores by topic
        topic_scores = defaultdict(list)
        for score_info in scores:
            topic_scores[score_info['topic_idx']].append(score_info)
        
        # For each topic, keep only the best matching chunk
        best_topic_matches = []
        for topic_idx, topic_scores_list in topic_scores.items():
            if topic_scores_list:
                best_match = max(topic_scores_list, key=lambda x: x['score'])
                best_topic_matches.append(best_match)
        
        # Apply operator logic
        required_terms = len(query_intent.topics)  # Only require topics, not key phrases
        if operator == SearchOperator.AND and len([m for m in best_topic_matches if not m['is_key_phrase']]) < required_terms:
            continue  # Skip if not all required topics matched for AND queries
        
        if not best_topic_matches:
            continue
            
        # Calculate overall score
        if operator == SearchOperator.AND and required_terms > 0:
            # Use the minimum score of required topics to ensure all are well-matched
            required_matches = [m for m in best_topic_matches if not m['is_key_phrase']]
            if required_matches:  # Only if we have required matches
                overall_score = min(match['score'] for match in required_matches)
            else:
                overall_score = max(match['score'] for match in best_topic_matches)
        else:
            # Use the maximum score for OR queries or if no required terms
            overall_score = max(match['score'] for match in best_topic_matches)
        
        if overall_score < min_score:
            continue
            
        # Get the best matching chunk for the highest scoring topic
        best_match = max(best_topic_matches, key=lambda x: x['score'])
        
        # Find the original document
        doc_metadata = next((d.get('metadata', {}) for d in documents 
                           if d.get('metadata', {}).get('id') == doc_id), {})
        
        # Add filename to metadata if not present
        if 'filename' not in doc_metadata and 'filename' in best_match['chunk'].get('metadata', {}):
            doc_metadata['filename'] = best_match['chunk']['metadata']['filename']
        
        results.append({
            'metadata': doc_metadata,
            'score': float(overall_score),
            'semantic_score': float(overall_score),
            'keyword_score': 0.0,
            'best_chunk': best_match['chunk']['text'],
            'match_type': 'semantic',
            'filename': doc_metadata.get('filename', ''),
            'matched_terms': [{
                'term': search_terms[match['topic_idx']],
                'score': float(match['score']),
                'is_key_phrase': match['is_key_phrase']
            } for match in best_topic_matches],
            'chunk_scores': [{
                'term': search_terms[match['topic_idx']],
                'score': float(match['score']),
                'text': match['chunk']['text'],
                'is_key_phrase': match['is_key_phrase']
            } for match in best_topic_matches if match['score'] >= min_score]
        })
    
    # Sort results by score (highest first) and apply top_k
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Final negation check (just to be safe)
    if apply_filters and negation_terms:
        filtered_results = []
        for result in results:
            result_content = f"{result.get('best_chunk', '')} {result.get('filename', '')}".lower()
            if not any(nt.lower() in result_content for nt in negation_terms):
                filtered_results.append(result)
        results = filtered_results
    
    return results[:top_k]

@retry_on_failure(default_return=[])
def hybrid_search(
    query: Union[str, QueryIntent], 
    documents: List[Dict[str, Any]],
    semantic_weight: float = 0.7, 
    top_k: int = 5, 
    min_score: float = 0.2,
    chunk_size: int = 3,
    overlap: int = 1
) -> List[Dict[str, Any]]:
    """
    Combine semantic search with keyword search for improved results.
    
    Args:
        query: The search query or QueryIntent object
        documents: List of documents with 'text' and 'metadata' keys
        semantic_weight: Weight for semantic score (0-1)
        top_k: Maximum number of results to return
        min_score: Minimum combined score (0-1) for results
        chunk_size: Number of sentences per chunk for semantic search
        overlap: Number of sentences to overlap between chunks
        
    Returns:
        List of matching documents with combined scores and metadata
    """
    if not documents or not query:
        return []
    
    # Convert query to QueryIntent if it's a string
    if isinstance(query, str):
        try:
            query_intent = QueryIntent(query)
        except Exception as e:
            logger.error(f"Failed to create QueryIntent: {str(e)}")
            return []
    else:
        query_intent = query
    
    # Get semantic search results with a lower threshold
    semantic_results = []
    try:
        semantic_results = semantic_search(
            query_intent, 
            documents, 
            top_k=len(documents), 
            min_score=min_score * 0.7,  # Lower threshold for semantic search
            chunk_size=chunk_size,
            overlap=overlap
        )
    except Exception as e:
        logger.error(f"Semantic search failed: {str(e)}")
        # Continue with empty semantic results if semantic search fails
    
    # Create a map of document ID to semantic score and best chunk
    semantic_info = {}
    for result in semantic_results:
        try:
            doc_id = result.get('metadata', {}).get('id', id(result))
            semantic_info[doc_id] = {
                'score': float(result.get('score', 0)),
                'best_chunk': result.get('best_chunk', ''),
                'matched_topics': list(result.get('matched_topics', [])),
                'chunk_scores': list(result.get('chunk_scores', []))
            }
        except Exception as e:
            logger.warning(f"Error processing semantic result: {str(e)}")
    
    # Calculate keyword scores for all documents with phrase support
    keyword_scores = []
    clean_q = clean_query(query_intent.text if hasattr(query_intent, 'text') else str(query))
    
    # Extract both individual terms and phrases for matching
    query_terms = {term for term in clean_q.split() if len(term) > 2}  # Ignore very short terms
    key_phrases = getattr(query_intent, 'key_phrases', [])
    
    for doc in documents:
        try:
            text = str(doc.get('text', '')).lower()
            doc_id = doc.get('metadata', {}).get('id', id(doc))
            
            # Calculate term matches
            term_matches = sum(1 for term in query_terms if term in text)
            
            # Calculate phrase matches (weighted higher)
            phrase_matches = sum(3 for phrase in key_phrases if phrase.lower() in text)
            
            # Combine scores with higher weight for phrases
            total_matches = term_matches + phrase_matches
            max_possible = len(query_terms) + (3 * len(key_phrases)) or 1  # Avoid division by zero
            keyword_score = min(total_matches / max_possible, 1.0)  # Cap at 1.0
            
            keyword_scores.append((doc_id, keyword_score))
        except Exception as e:
            logger.warning(f"Error calculating keyword score: {str(e)}")
            continue
    
    # Normalize keyword scores to 0-1 range
    keyword_scores_dict = {}
    if keyword_scores:
        max_kw_score = max(score for _, score in keyword_scores) or 1
        keyword_scores_dict = {doc_id: score/max_kw_score for doc_id, score in keyword_scores}
    
    # Combine scores and prepare results
    results = []
    
    # Get all unique document IDs from both searches
    all_doc_ids = set(semantic_info.keys())
    all_doc_ids.update(doc_id for doc_id in keyword_scores_dict)
    
    for doc_id in all_doc_ids:
        try:
            # Get document metadata (from semantic results or original documents)
            doc_metadata = next(
                (result['metadata'] for result in semantic_results 
                 if result.get('metadata', {}).get('id') == doc_id),
                next((doc.get('metadata', {}) for doc in documents 
                      if doc.get('metadata', {}).get('id') == doc_id), {})
            )
            
            # Get scores from both searches with defaults
            semantic_info_doc = semantic_info.get(doc_id, {
                'score': 0.0, 
                'best_chunk': '',
                'matched_topics': [],
                'chunk_scores': []
            })
            
            semantic_score = float(semantic_info_doc.get('score', 0))
            keyword_score = float(keyword_scores_dict.get(doc_id, 0))
            
            # Combine scores using weighted average
            combined_score = (semantic_weight * semantic_score) + ((1 - semantic_weight) * keyword_score)
            
            if combined_score >= min_score:
                result = {
                    'metadata': doc_metadata,
                    'score': combined_score,
                    'semantic_score': semantic_score,
                    'keyword_score': keyword_score,
                    'best_chunk': semantic_info_doc.get('best_chunk', ''),
                    'matched_topics': semantic_info_doc.get('matched_topics', []),
                    'chunk_scores': semantic_info_doc.get('chunk_scores', []),
                    'match_type': 'hybrid'
                }
                
                # If we have a good keyword match but no semantic match, include some context
                if semantic_score < min_score * 0.5 and keyword_score > 0.5:
                    # Find the document text
                    doc_text = next((str(d.get('text', '')) for d in documents 
                                   if d.get('metadata', {}).get('id') == doc_id), '')
                    
                    if doc_text:
                        # Get a snippet around the first matching term/phrase
                        snippet = _get_context_snippet(doc_text, query_terms.union(key_phrases))
                        if snippet:
                            result['best_chunk'] = snippet
                
                results.append(result)
        except Exception as e:
            logger.warning(f"Error processing document {doc_id}: {str(e)}")
            continue
    
    # Sort by combined score (highest first) and apply top_k
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]
    return results[:top_k]


def _get_context_snippet(text: str, search_terms: Set[str], window_size: int = 150) -> str:
    """
    Extract a context window around the first occurrence of any search term.
    
    Args:
        text: The full text to search in
        search_terms: Set of terms or phrases to look for
        window_size: Number of characters to include around the match
        
    Returns:
        A string containing the context window, or empty string if no match
    """
    if not text or not search_terms:
        return ""
    
    text_lower = text.lower()
    
    # Find the earliest occurrence of any search term
    first_pos = len(text)
    for term in search_terms:
        pos = text_lower.find(term.lower())
        if 0 <= pos < first_pos:
            first_pos = pos
    
    if first_pos >= len(text):
        return ""  # No match found
    
    # Calculate window bounds
    start = max(0, first_pos - (window_size // 2))
    end = min(len(text), first_pos + len(term) + (window_size // 2))
    
    # Extract the window
    snippet = text[start:end]
    
    # Add ellipsis if needed
    if start > 0:
        snippet = '...' + snippet
    if end < len(text):
        snippet = snippet + '...'
    
    return snippet
