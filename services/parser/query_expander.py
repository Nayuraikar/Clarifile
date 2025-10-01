"""
Query expansion and enhancement utilities for semantic search.

This module provides functionality to expand search queries with synonyms,
handle typos, and improve search results through semantic matching.
"""

import re
from typing import List, Set, Dict, Any, Optional, Tuple
from nltk.corpus import wordnet
import numpy as np
import textdistance
from sentence_transformers import SentenceTransformer, util
import torch
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class QueryExpander:
    """
    A class to handle query expansion and enhancement for semantic search.
    
    This class provides methods to expand queries with synonyms, handle typos,
    and perform semantic search with enhanced queries.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
        """
        Initialize the QueryExpander with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
            device: Device to run the model on ('cuda', 'mps', or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info(f"Initialized QueryExpander with model '{model_name}' on device '{self.device}'")
    
    def get_synonyms(self, word: str, pos: str = None) -> List[str]:
        """
        Get synonyms for a word using WordNet.
        
        Args:
            word: The word to find synonyms for
            pos: Part of speech filter (e.g., 'n' for noun, 'v' for verb)
            
        Returns:
            List of synonyms
        """
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
        """
        Expand a list of query terms with synonyms.
        
        Args:
            query_terms: List of terms to expand
            
        Returns:
            Dictionary mapping original terms to their expansions
        """
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
        """
        Find terms in the corpus that are similar to the input term.
        
        Args:
            term: The term to find similar terms for
            corpus_terms: List of terms to compare against
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of (term, similarity_score) tuples
        """
        similar = []
        for t in corpus_terms:
            # Skip the term itself
            if t.lower() == term.lower():
                continue
                
            # Calculate Jaro-Winkler similarity (good for typos)
            similarity = textdistance.jaro_winkler(term.lower(), t.lower())
            if similarity >= threshold:
                similar.append((t, similarity))
        
        # Sort by similarity score (highest first)
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar
    
    def expand_query(self, query: str, max_expansions: int = 5) -> Dict[str, Any]:
        """
        Expand a search query with synonyms and similar terms.
        
        Args:
            query: The original search query
            max_expansions: Maximum number of expansions per term
            
        Returns:
            Dictionary containing expanded query information
        """
        # Tokenize and clean the query
        terms = [t.lower() for t in re.findall(r'\b\w+\b', query)]
        
        # Expand each term with synonyms
        expanded_terms = self.expand_query_terms(terms)
        
        # Find similar terms for each original term
        all_terms = set(terms)
        for term in terms:
            similar = self.find_similar_terms(term, list(all_terms))
            # Add top similar terms
            for t, _ in similar[:max_expansions]:
                all_terms.add(t)
        
        # Generate embeddings for all terms
        term_list = list(all_terms)
        if not term_list:
            return {
                'original_query': query,
                'expanded_terms': {},
                'expanded_queries': [query],
                'embeddings': None
            }
        
        # Get embeddings for all terms
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
        """
        Perform semantic search with query expansion.
        
        Args:
            query: The search query
            documents: List of documents to search
            top_k: Maximum number of results to return
            min_score: Minimum similarity score (0-1)
            chunk_size: Number of sentences per chunk
            overlap: Number of sentences to overlap between chunks
            apply_filters: Whether to apply file type and negation filters
            
        Returns:
            List of search results
        """
        from services.parser.semantic_search import semantic_search
        
        # Expand the query
        expanded = self.expand_query(query)
        
        # If no expansions, fall back to regular semantic search
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
        
        # Perform search with each expanded query
        all_results = []
        for q in expanded['expanded_queries']:
            results = semantic_search(
                query=q,
                documents=documents,
                top_k=top_k * 2,  # Get more results to account for merging
                min_score=min_score,
                chunk_size=chunk_size,
                overlap=overlap,
                apply_filters=apply_filters
            )
            all_results.extend(results)
        
        # Deduplicate and sort results
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

# Global instance for convenience
try:
    query_expander = QueryExpander()
except Exception as e:
    logger.error(f"Failed to initialize QueryExpander: {str(e)}")
    query_expander = None
