# services/parser/nlp.py
"""
NLU helpers using Google Generative Language (Gemini) via API key.
- Gemini calls are deterministic (temperature=0).
- If context is long we chunk by sentence boundaries and query each chunk.
- If Gemini can't find an answer, it will be instructed to return "I don't know".
"""

import os
import time
import requests
import nltk
import re
import logging
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'parser.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# sentence tokenizer
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

# --- CONFIG ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
if not GEMINI_API_KEYS and GEMINI_API_KEY:
    GEMINI_API_KEYS = [GEMINI_API_KEY]

# Optional on-disk config: if no env keys, read from a local keys file
if not GEMINI_API_KEYS:
    default_keys_file = os.getenv("GEMINI_KEYS_FILE", os.path.join(os.path.dirname(__file__), "gemini_keys.txt"))
    try:
        if os.path.exists(default_keys_file):
            with open(default_keys_file, "r", encoding="utf-8") as f:
                raw = f.read()
            parsed = []
            for line in raw.strip().split("\n"):
                s = line.strip()
                if s:
                    parsed.append(s)
            if parsed:
                GEMINI_API_KEYS = parsed
                print(f"Loaded {len(GEMINI_API_KEYS)} API keys from {default_keys_file}")
    except Exception as _e:
        # If file read fails, keep empty and error later at call-site
        pass
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

REQUEST_TIMEOUT = 45
RETRY_COUNT = 3
RETRY_BACKOFF = 2.0
RATE_LIMIT_BACKOFF = 10.0  # Increased from 5.0 to 10.0 seconds
API_CALL_DELAY = 1.0  # Delay between API calls to prevent rate limiting

# Round-robin index for key rotation (in-memory)
_KEY_INDEX = 0
_KEY_FAILURES = {}  # Track failures per key index
_KEY_LAST_USED = {}  # Track last usage time per key index
_RATE_LIMITED_KEYS = set()  # Track which keys are currently rate limited


# --- Utilities ---
def _safe_post_with_key_rotation(payload: dict):
    """POST with retries and API key rotation; returns JSON (or raises).
    - Tries current key, on 401/403/429 or network errors rotates to next key.
    - Persists next starting key via module-level _KEY_INDEX.
    """
    global _KEY_INDEX
    if not GEMINI_API_KEYS:
        raise RuntimeError("No Gemini API keys configured. Set GEMINI_API_KEYS or GEMINI_API_KEY.")

    last_exc = None
    total_keys = len(GEMINI_API_KEYS)
    current_time = time.time()

    # Try each key at most once per attempt (with backoff between attempts)
    for attempt in range(RETRY_COUNT + 1):
        for i in range(total_keys):
            key_index = _KEY_INDEX % total_keys
            key = GEMINI_API_KEYS[key_index]

            # Skip rate-limited keys for at least 60 seconds
            if key_index in _RATE_LIMITED_KEYS:
                if current_time - _KEY_LAST_USED.get(key_index, 0) < 60:
                    _KEY_INDEX = (_KEY_INDEX + 1) % total_keys
                    continue
                else:
                    _RATE_LIMITED_KEYS.discard(key_index)

            _KEY_INDEX = (_KEY_INDEX + 1) % total_keys
            _KEY_LAST_USED[key_index] = current_time

            headers = {"Content-Type": "application/json", "X-goog-api-key": key}
            try:
                # Add delay between API calls to prevent rate limiting
                time.sleep(API_CALL_DELAY)

                r = requests.post(GEMINI_ENDPOINT, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
                if r.status_code in (401, 403):
                    # Invalid key - mark as failed and try next
                    last_exc = requests.HTTPError(f"Gemini HTTP {r.status_code}")
                    _KEY_FAILURES[key_index] = _KEY_FAILURES.get(key_index, 0) + 1
                    continue
                elif r.status_code == 429:
                    # Rate limited - mark this key as rate limited and try next
                    last_exc = requests.HTTPError(f"Gemini HTTP {r.status_code} - Rate limited")
                    _RATE_LIMITED_KEYS.add(key_index)
                    _KEY_FAILURES[key_index] = _KEY_FAILURES.get(key_index, 0) + 1
                    # Longer backoff for rate limits
                    time.sleep(RATE_LIMIT_BACKOFF * 2)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.exceptions.Timeout:
                last_exc = requests.HTTPError("Request timeout")
                _KEY_FAILURES[key_index] = _KEY_FAILURES.get(key_index, 0) + 1
                continue
            except requests.exceptions.ConnectionError:
                last_exc = requests.HTTPError("Connection error")
                _KEY_FAILURES[key_index] = _KEY_FAILURES.get(key_index, 0) + 1
                continue
            except Exception as e:
                last_exc = e
                _KEY_FAILURES[key_index] = _KEY_FAILURES.get(key_index, 0) + 1
                continue

        # If we get here, all keys failed for this attempt
        if attempt < RETRY_COUNT:
            # Exponential backoff between attempts
            backoff_time = RETRY_BACKOFF * (2 ** attempt)
            print(f"All API keys failed attempt {attempt + 1}/{RETRY_COUNT + 1}. Retrying in {backoff_time}s...")
            time.sleep(backoff_time)

    raise last_exc or RuntimeError("All API keys failed")


def _extract_generated_text(resp_json: dict) -> str:
    """
    Extract text from Gemini response safely.
    """
    if not resp_json:
        return ""

    try:
        return resp_json["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        pass

    # Fallback: search for any "text" recursively
    def find_texts(obj):
        found = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "text" and isinstance(v, str):
                    found.append(v)
                else:
                    found.extend(find_texts(v))
        elif isinstance(obj, list):
            for el in obj:
                found.extend(find_texts(el))
        return found

    all_texts = find_texts(resp_json)
    return " ".join(all_texts).strip()


# --- Gemini call ---
def gemini_generate(prompt: str, temperature: float = 0.0, max_output_tokens: int = 256) -> str:
    """
    Call Gemini generateContent endpoint using API key.
    Returns the generated text or empty string on failure.
    """
    if not GEMINI_API_KEYS:
        print("❌ No Gemini API keys configured")
        return ""

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {          # ✅ FIXED: correct place for params
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        }
    }

    try:
        resp_json = _safe_post_with_key_rotation(payload)
        return _extract_generated_text(resp_json)
    except requests.HTTPError as e:
        print(f"Gemini API HTTP error: {repr(e)}")
        try:
            print("Response text:", e.response.text)
        except Exception:
            pass
        return ""
    except RuntimeError as e:
        print(f"Gemini API runtime error: {e}")
        return ""
    except Exception as e:
        print(f"Gemini API error: {repr(e)}")
        return ""


# --- Chunking helper ---
def smart_chunks(text: str, max_words: int = 600, stride: int = 120):
    """
    Split text into overlapping chunks made of whole sentences.
    """
    if not text:
        return

    sentences = sent_tokenize(text)
    chunk_words = []
    count = 0
    for sent in sentences:
        w = sent.split()
        if count + len(w) > max_words and chunk_words:
            yield " ".join(chunk_words)
            if stride > 0 and len(chunk_words) > stride:
                chunk_words = chunk_words[-stride:]
                count = len(chunk_words)
            else:
                chunk_words = []
                count = 0
        chunk_words.extend(w)
        count += len(w)
    if chunk_words:
        yield " ".join(chunk_words)


# --- Main QA using Gemini ---
def best_answer_or_summary(question: str, context_text: str) -> Dict:
    """
    Use Gemini to answer a question from context_text.
    Now tuned to be less strict and give fuller answers.
    """
    if not question or not context_text:
        return {"answer": "", "score": 0.0, "context": ""}

    base_instruction = (
        "You are a helpful assistant. Use the TEXT below to answer the QUESTION. "
        "Prefer giving the best possible answer from the text, even if partially relevant. "
        "If the text truly contains nothing useful, then reply exactly: I don't know.\n\n"
    )

    if len(context_text.split()) <= 1200:
        prompt = f"{base_instruction}TEXT:\n{context_text}\n\nQUESTION:\n{question}\n\nANSWER:"
        generated = gemini_generate(prompt, temperature=0.2, max_output_tokens=512)
        return {"answer": generated.strip(), "score": 1.0 if generated else 0.0, "context": context_text}

    # For long contexts → chunk
    candidate_answers = []
    for chunk in smart_chunks(context_text, max_words=600, stride=120):
        prompt = f"{base_instruction}TEXT:\n{chunk}\n\nQUESTION:\n{question}\n\nANSWER:"
        generated = gemini_generate(prompt, temperature=0.2, max_output_tokens=300)
        g = (generated or "").strip()
        if g and not re.fullmatch(r"(?i)i\s*do(n'?| no)t know", g):
            candidate_answers.append((g, chunk))

    if candidate_answers:
        # pick longest, most detailed answer
        candidate_answers.sort(key=lambda x: len(x[0]), reverse=True)
        best_ans, best_chunk = candidate_answers[0]
        return {"answer": best_ans, "score": 1.0, "context": best_chunk}

    # fallback
    prompt = f"{base_instruction}TEXT:\n{context_text}\n\nQUESTION:\n{question}\n\nANSWER:"
    final_generated = gemini_generate(prompt, temperature=0.2, max_output_tokens=400)
    return {"answer": final_generated.strip(), "score": 1.0 if final_generated else 0.0, "context": context_text}


# Backward compatibility
best_answer = best_answer_or_summary


def extract_entities(text: str):
    """
    Very lightweight entity extractor.
    """
    if not text:
        return []
    candidates = re.findall(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b", text)
    uniques, seen = [], set()
    for c in candidates:
        if c.lower() not in seen:
            seen.add(c.lower())
            uniques.append({"name": c, "type": "ENTITY"})
    return uniques

def summarize_with_gemini(long_text: str, max_tokens: int = 512) -> str:
    if not long_text.strip():
        return ""

    # Try Gemini API first
    try:
        prompt = (
            "Summarize the following text into a concise, clear paragraph. "
            "Capture the key points without losing important details:\n\n"
            f"{long_text}\n\nSUMMARY:"
        )
        result = gemini_generate(prompt, temperature=0.3, max_output_tokens=max_tokens)
        if result.strip():
            return result.strip()
    except Exception as e:
        print(f"Gemini summarization failed: {e}")

    # Fallback: simple text summarization
    print("Using fallback summarization...")
    sentences = long_text.strip().split('.')
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 2:
        return long_text[:500] + "..." if len(long_text) > 500 else long_text

    # Take first sentence + key sentences
    summary_parts = [sentences[0]]
    if len(sentences) > 1:
        summary_parts.append(sentences[len(sentences)//2])  # middle sentence
    if len(sentences) > 2:
        summary_parts.append(sentences[-1])  # last sentence

    summary = '. '.join(summary_parts) + '.'
    return summary[:500] + "..." if len(summary) > 500 else summary

def classify_with_gemini(content: str) -> str:
    """
    Use Gemini to classify file content into a detailed and specific category.
    The categories are designed to be specific and useful for document organization.
    """
    if not content.strip():
        return "Uncategorized"

    # Extract the first 2000 characters for analysis (to avoid token limits)
    analysis_text = content[:2000]
    text_lower = analysis_text.lower()
    
    # ===== EXPLICIT ACADEMIC PAPER DETECTION =====
    # Check for academic paper indicators first (before Gemini)
    academic_keywords = [
        'abstract', 'introduction', 'methodology', 'experiment', 'results', 'discussion',
        'conclusion', 'references', 'bibliography', 'citation', 'literature review',
        'peer-reviewed', 'academic', 'research', 'study', 'paper', 'thesis', 'dissertation',
        'conference', 'journal', 'publication', 'author', 'affiliation', 'university',
        'institute', 'icse', 'ieee', 'acm', 'springer', 'elsevier', 'scientific', 'empirical',
        'related work', 'background', 'experimental results', 'evaluation', 'findings',
        'contribution', 'limitation', 'future work', 'appendix'
    ]
    
    # Count academic indicators in the text
    academic_score = sum(1 for word in academic_keywords if word in text_lower)
    
    # If we have multiple academic indicators, it's definitely a research paper
    if academic_score >= 2:
        print(f"Academic paper detected with score {academic_score}")
        # Further refine the category based on content
        if any(word in text_lower for word in ['software', 'programming', 'code', 'algorithm', 'system', 'engineering', 'icse']):
            return "Computer Science: Research Paper"
        elif any(word in text_lower for word in ['medical', 'health', 'clinical', 'patient', 'disease']):
            return "Medical: Research Paper"
        elif any(word in text_lower for word in ['mathematics', 'maths', 'algebra', 'calculus']):
            return "Mathematics: Research Paper"
        elif any(word in text_lower for word in ['physics', 'quantum', 'mechanics', 'relativity']):
            return "Physics: Research Paper"
        else:
            return "Academic: Research Paper"
    
    # Check for PDFs with academic structure
    if '.pdf' in content.lower() and len(analysis_text) > 500:
        # Look for section headers that indicate academic content
        section_headers = ['abstract', 'introduction', 'method', 'result', 'conclusion', 'reference']
        section_count = sum(1 for section in section_headers if f"\n{section}" in text_lower)
        
        if section_count >= 2:  # At least 2 academic sections found
            print(f"PDF with academic structure detected: {section_count} sections found")
            return "Academic: Research Paper"
    
    # ===== GEMINI CLASSIFICATION =====
    # Only try Gemini if we haven't already classified it as an academic paper
    try:
        prompt = (
            "You are an advanced document classification system. Analyze the following content "
            "and determine the MOST SPECIFIC category that best describes it. Follow these rules:\n\n"
            "1. If the content appears to be an academic paper, research article, or scholarly work, "
               "classify it as 'Academic: Research Paper' or a more specific academic category.\n"
            "2. For academic papers, try to identify the specific field (e.g., 'Computer Science', 'Physics', 'Biology').\n\n"
            "3. For other documents, use the most specific category possible from these main categories:\n"
            "   - Academic: [Research Paper, Lecture Notes, Course Material]\n"
            "   - Computer Science: [Research Paper, Documentation, Code]\n"
            "   - Work: [Meeting Notes, Report, Presentation]\n"
            "   - Finance: [Invoice, Receipt, Statement]\n"
            "   - Legal: [Contract, Agreement, Terms]\n"
            "   - Personal: [Journal, Notes, Correspondence]\n\n"
            "4. Always respond in the format: 'Category: Subcategory'\n\n"
            f"CONTENT TO CLASSIFY:\n{analysis_text}\n\n"
            "ANALYZE THE CONTENT AND RESPOND WITH ONLY THE CATEGORY IN THIS FORMAT: 'Category: Subcategory'\n"
            "CATEGORY:"
        )

        result = gemini_generate(prompt, temperature=0.0, max_output_tokens=50)
        result = result.strip()

        # Basic validation and cleaning of the response
        if result and len(result) > 2 and len(result) <= 100:
            # If the response doesn't contain a colon, try to format it
            if ':' not in result:
                # If it's a single word, make it a main category
                if ' ' not in result:
                    return f"{result}: General"
                # Otherwise, take the first word as category, rest as subcategory
                parts = result.split()
                return f"{parts[0]}: {' '.join(parts[1:])}"

            return result

        print(f"Gemini returned invalid result: '{result}' for content: {analysis_text[:100]}...")

    except Exception as e:
        print(f"Gemini classification failed: {e}")
        
    # ===== FALLBACK CLASSIFICATION =====
    print("Using enhanced fallback classification...")
    
    # Define document type keywords and their categories
    doc_keywords = {
        'Academic: Research Paper': [
            'abstract', 'introduction', 'methodology', 'experiment', 'results', 'discussion', 
            'conclusion', 'references', 'bibliography', 'citation', 'literature review',
            'peer-reviewed', 'academic', 'research', 'study', 'paper', 'thesis', 'dissertation',
            'conference', 'journal', 'publication', 'author', 'affiliation', 'university',
            'institute', 'icse', 'ieee', 'acm', 'springer', 'elsevier', 'scientific', 'empirical',
            'related work', 'background', 'experimental results', 'evaluation', 'findings',
            'contribution', 'limitation', 'future work', 'appendix'
        ],
        'Computer Science: Research Paper': [
            'software', 'programming', 'code', 'algorithm', 'system', 'engineering',
            'computer science', 'software engineering', 'data structure', 'artificial intelligence',
            'machine learning', 'computer vision', 'natural language processing', 'icse', 'ieee', 'acm'
        ],
        'Finance: Documents': [
            'invoice', 'bill', 'payment', 'receipt', 'financial', 'budget', 'expense', 'tax',
            'bank', 'transaction', 'statement', 'balance', 'revenue', 'profit', 'loss'
        ],
        'Legal: Contracts': [
            'contract', 'legal', 'agreement', 'terms', 'law', 'court', 'litigation',
            'clause', 'party', 'signature', 'agreement', 'terms and conditions'
        ],
        'Work: Meetings': [
            'meeting', 'agenda', 'minutes', 'presentation', 'report', 'memo',
            'action items', 'attendees', 'discussion points', 'meeting notes'
        ]
    }
    
    # Count keyword matches for each category
    category_scores = {}
    for category, keywords in doc_keywords.items():
        score = sum(1 for word in keywords if word in text_lower)
        if score > 0:
            category_scores[category] = score
    
    # If we have matches, return the category with the highest score
    if category_scores:
        best_category = max(category_scores.items(), key=lambda x: x[1])[0]
        print(f"Fallback classification: {best_category} (score: {category_scores[best_category]})")
        return best_category
    
    # Check for PDFs with academic structure
    if '.pdf' in content.lower() and len(analysis_text) > 500:
        section_headers = ['abstract', 'introduction', 'method', 'result', 'conclusion', 'reference']
        section_count = sum(1 for section in section_headers if f"\n{section}" in text_lower)
        if section_count >= 2:
            print(f"PDF with academic structure detected: {section_count} sections")
            return "Academic: Research Paper"
    
    # Final fallback based on content length
    if len(analysis_text) > 500:  # Substantial content
        return "General: Document"
    elif '.pdf' in content.lower():
        return "Academic: Research Paper"  # Default for PDFs we're not sure about
    
    return "Uncategorized: General"