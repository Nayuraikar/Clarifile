"""
Search caching module for semantic search and document processing.

This module provides caching functionality for embeddings, document chunks,
and API responses to improve search performance.
"""

import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

# Cache configuration
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.search_cache')
EMBEDDING_CACHE_DIR = os.path.join(CACHE_DIR, 'embeddings')
CHUNK_CACHE_DIR = os.path.join(CACHE_DIR, 'chunks')
SUMMARY_CACHE_DIR = os.path.join(CACHE_DIR, 'summaries')
CACHE_EXPIRY_DAYS = 30  # Number of days before cache expires

# Ensure cache directories exist
for directory in [EMBEDDING_CACHE_DIR, CHUNK_CACHE_DIR, SUMMARY_CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

def get_cache_key(text: str, prefix: str = '') -> str:
    """Generate a cache key for the given text and prefix."""
    return hashlib.md5(f"{prefix}_{text}".encode('utf-8')).hexdigest()

def get_cache_path(cache_dir: str, key: str) -> str:
    """Get the full cache file path for a given key."""
    return os.path.join(cache_dir, f"{key}.pkl")

def is_cache_valid(cache_path: str) -> bool:
    """Check if a cache file exists and hasn't expired."""
    if not os.path.exists(cache_path):
        return False
    
    # Check if cache is older than expiry time
    cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    return datetime.now() - cache_time < timedelta(days=CACHE_EXPIRY_DAYS)

def load_from_cache(cache_path: str) -> Any:
    """Load data from cache file."""
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except (pickle.PickleError, EOFError, FileNotFoundError):
        return None

def save_to_cache(data: Any, cache_path: str) -> None:
    """Save data to cache file."""
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    except (pickle.PickleError, IOError) as e:
        print(f"Warning: Failed to save cache: {e}")

class EmbeddingCache:
    """Cache for document embeddings."""
    
    @staticmethod
    def get_embedding(text: str, model) -> np.ndarray:
        """Get embedding from cache or generate if not found."""
        key = get_cache_key(text, 'emb')
        cache_path = get_cache_path(EMBEDDING_CACHE_DIR, key)
        
        # Try to load from cache
        if is_cache_valid(cache_path):
            cached = load_from_cache(cache_path)
            if cached is not None:
                return cached
        
        # Generate and cache if not found
        embedding = model.encode(text, convert_to_tensor=False)
        save_to_cache(embedding, cache_path)
        return embedding

class ChunkCache:
    """Cache for document chunks."""
    
    @staticmethod
    def get_chunks(text: str, chunk_func, *args, **kwargs) -> List[str]:
        """Get chunks from cache or generate if not found."""
        key = get_cache_key(text, 'chunk')
        cache_path = get_cache_path(CHUNK_CACHE_DIR, key)
        
        # Try to load from cache
        if is_cache_valid(cache_path):
            cached = load_from_cache(cache_path)
            if cached is not None:
                return cached
        
        # Generate and cache if not found
        chunks = list(chunk_func(text, *args, **kwargs))
        save_to_cache(chunks, cache_path)
        return chunks

class SummaryCache:
    """Cache for document summaries."""
    
    @staticmethod
    def get_summary(text: str, summary_func, *args, **kwargs) -> str:
        """Get summary from cache or generate if not found."""
        key = get_cache_key(text, 'sum')
        cache_path = get_cache_path(SUMMARY_CACHE_DIR, key)
        
        # Try to load from cache
        if is_cache_valid(cache_path):
            cached = load_from_cache(cache_path)
            if cached is not None:
                return cached
        
        # Generate and cache if not found
        summary = summary_func(text, *args, **kwargs)
        if summary:  # Only cache if we got a valid summary
            save_to_cache(summary, cache_path)
        return summary

def clear_old_cache() -> None:
    """Remove cache files older than the expiry time."""
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

# Clean up old cache on import
clear_old_cache()
