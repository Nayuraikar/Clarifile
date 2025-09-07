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
from typing import Dict

# sentence tokenizer
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

# --- CONFIG ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

REQUEST_TIMEOUT = 60
RETRY_COUNT = 2
RETRY_BACKOFF = 0.6


# --- Utilities ---
def _safe_post(payload: dict, headers: dict):
    """POST with retries and return JSON (or raise)."""
    last_exc = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            r = requests.post(GEMINI_ENDPOINT, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            time.sleep(RETRY_BACKOFF * (attempt + 1))
    raise last_exc


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
def gemini_generate(prompt: str, temperature: float = 0.0, max_output_tokens: int = 512) -> str:
    """
    Call Gemini generateContent endpoint using API key.
    Returns the generated text or empty string on failure.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured. Set environment variable GEMINI_API_KEY.")

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY,
    }

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
        resp_json = _safe_post(payload, headers)
        return _extract_generated_text(resp_json)
    except requests.HTTPError as e:
        print(f"Gemini API HTTP error: {repr(e)}")
        try:
            print("Response text:", e.response.text)
        except Exception:
            pass
        return ""
    except Exception as e:
        print("Gemini API error:", repr(e))
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
    Use Gemini to answer question strictly from context_text.
    """
    if not question or not context_text:
        return {"answer": "", "score": 0.0, "context": ""}

    base_instruction = (
        "You are a strict assistant. Use ONLY the information in the TEXT below to answer the QUESTION. "
        "If the TEXT does not contain the requested info, reply exactly: I don't know.\n\n"
    )

    if len(context_text.split()) <= 1200:
        prompt = f"{base_instruction}TEXT:\n{context_text}\n\nQUESTION:\n{question}\n\nANSWER:"
        generated = gemini_generate(prompt, temperature=0.0, max_output_tokens=512)
        return {"answer": generated.strip(), "score": 1.0 if generated else 0.0, "context": context_text}

    # For long contexts
    candidate_answers = []
    for chunk in smart_chunks(context_text, max_words=600, stride=120):
        prompt = f"{base_instruction}TEXT:\n{chunk}\n\nQUESTION:\n{question}\n\nANSWER:"
        generated = gemini_generate(prompt, temperature=0.0, max_output_tokens=300)
        if not generated:
            continue
        g = generated.strip()
        if re.fullmatch(r"(?i)i\s*do(n'?| no)t know", g):
            continue
        candidate_answers.append((g, chunk))

    if candidate_answers:
        candidate_answers.sort(key=lambda x: len(x[0]), reverse=True)
        best_ans, best_chunk = candidate_answers[0]
        return {"answer": best_ans, "score": 1.0, "context": best_chunk}

    # Final attempt across whole text
    prompt = f"{base_instruction}TEXT:\n{context_text}\n\nQUESTION:\n{question}\n\nANSWER:"
    final_generated = gemini_generate(prompt, temperature=0.0, max_output_tokens=400)
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


# --- NEW: Gemini summarizer ---
def summarize_with_gemini(long_text: str, max_tokens: int = 512) -> str:
    if not long_text.strip():
        return ""
    prompt = (
        "Summarize the following text into a concise, clear paragraph. "
        "Capture the key points without losing important details:\n\n"
        f"{long_text}\n\nSUMMARY:"
    )
    return gemini_generate(prompt, temperature=0.3, max_output_tokens=max_tokens).strip()

def classify_with_gemini(summary: str) -> str:
    """
    Use Gemini to classify a file summary into a broad category.
    Example categories: Finance, Legal, Medical, Work, Personal, Education, Technical, Media, Other.
    """
    if not summary.strip():
        return "Uncategorized"

    prompt = (
        "Classify the following text summary into ONE broad category label only. "
        "Possible categories include: Finance, Legal, Medical, Work, Personal, Education, Technical, Media, Other. "
        "Respond with ONLY the category name, nothing else.\n\n"
        f"SUMMARY:\n{summary}\n\nCATEGORY:"
    )
    result = gemini_generate(prompt, temperature=0.0, max_output_tokens=10)
    return result.strip() or "Uncategorized"