"""
FastAPI app for dedup service - optimized for Google Drive files
"""

import os
import logging
import hashlib
import re
import sqlite3
from fastapi import FastAPI
from typing import List, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

def is_google_drive_url(filepath):
    """Check if the path is a Google Drive URL"""
    if not filepath:
        return False
    return 'drive.google.com' in filepath or 'docs.google.com' in filepath

def compute_drive_content_hash(file_url, file_name):
    """Compute hash for Drive files based on URL and name"""
    try:
        if not file_url or not file_name:
            return None

        # Extract file ID from Google Drive URL if possible
        file_id_match = re.search(r'/file/d/([^/]+)/', file_url)
        if file_id_match:
            file_id = file_id_match.group(1)
            # Use file ID as the hash - identical files will have same ID
            return f"drive_{file_id}"

        # Fallback to URL hash if no file ID found
        return hashlib.sha256(f"{file_url}_{file_name}".encode('utf-8')).hexdigest()

    except Exception as e:
        logger.error(f"Error computing Drive hash for {file_name}: {e}")
        return None

def is_text_or_pdf_file(filepath):
    """Check if file is a text file or PDF"""
    if not filepath:
        return False

    # Check file extension
    text_extensions = {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv', '.log'}
    pdf_extensions = {'.pdf'}

    file_ext = os.path.splitext(filepath)[1].lower()

    if file_ext in text_extensions:
        return True
    elif file_ext in pdf_extensions:
        return True

    # For unknown extensions, try to detect if it's text
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(1024)
            # Check if it's likely text (contains mostly printable characters)
            text_chars = sum(1 for byte in chunk if 32 <= byte <= 126 or byte in {9, 10, 13})
            return (text_chars / len(chunk)) > 0.8 if chunk else False
    except:
        return False

def compute_exact_content_hash(filepath):
    """Compute hash of exact file content for precise duplicate detection"""
    try:
        if not os.path.exists(filepath):
            logger.warning(f"File does not exist: {filepath}")
            return None

        with open(filepath, "rb") as f:
            content = f.read()
            if not content:
                return "empty_file_hash"

        # For exact comparison, use the raw bytes
        return hashlib.sha256(content).hexdigest()

    except (IOError, OSError) as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return None

def compute_normalized_text_hash(filepath):
    """Compute hash of normalized text content (ignores whitespace differences)"""
    try:
        if not os.path.exists(filepath):
            logger.warning(f"File does not exist: {filepath}")
            return None

        with open(filepath, "rb") as f:
            content = f.read()
            if not content:
                return "empty_file_hash"

        # Try to normalize if it's text
        try:
            text_content = content.decode('utf-8', errors='ignore')
            normalized = normalize_text_content(text_content)
            return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.warning(f"Error normalizing text content for {filepath}: {e}")
            # If normalization fails, use original content
            return hashlib.sha256(content).hexdigest()

    except (IOError, OSError) as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return None

def normalize_text_content(content):
    """Normalize text content for comparison (remove whitespace, NULLs, control characters)"""
    if isinstance(content, bytes):
        try:
            content = content.decode('utf-8', errors='ignore')
        except:
            return content  # fallback

    # Remove NULL bytes and other control characters
    content = re.sub(r'[\x00-\x1F\x7F]', '', content)

    # Normalize whitespace and line endings
    normalized = re.sub(r'\s+', ' ', content.strip())
    return normalized

def get_db_connection():
    """Get database connection"""
    return sqlite3.connect("metadata_db/clarifile.db", check_same_thread=False)

app = FastAPI(title="Clarifile Dedup Service - Drive Optimized")

@app.get("/duplicates")
def find_duplicates():
    """Find duplicate files based on content hash - optimized for Drive files"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get all files with their paths
        cursor.execute("SELECT id, file_path, file_name, category_id, proposed_label, final_label FROM files WHERE file_path IS NOT NULL")
        files = cursor.fetchall()

        if not files:
            return {
                "summary": {
                    "duplicate_groups_found": 0,
                    "total_files_processed": 0
                },
                "duplicates": []
            }

        # Dictionary to store hash -> list of files
        hash_to_files = {}

        for file_id, file_path, file_name, category_id, proposed_label, final_label in files:
            if not file_path or not file_name:
                continue

            # Choose hash method based on file type
            if is_google_drive_url(file_path):
                # For Google Drive files, use Drive-specific hashing
                file_hash = compute_drive_content_hash(file_path, file_name)
                logger.info(f"Drive file {file_name}: using Drive hash")
            elif is_text_or_pdf_file(file_path):
                # For local text/PDF files, use exact content comparison
                file_hash = compute_exact_content_hash(file_path)
                logger.info(f"Local text/PDF file {file_name}: using exact hash")
            else:
                # For other local files, use normalized text hash
                file_hash = compute_normalized_text_hash(file_path)
                logger.info(f"Local binary file {file_name}: using normalized hash")

            if file_hash:
                if file_hash not in hash_to_files:
                    hash_to_files[file_hash] = []
                hash_to_files[file_hash].append({
                    "id": file_id,
                    "path": file_path,
                    "name": file_name,
                    "category": final_label or proposed_label or f"Category_{category_id}" if category_id else "Uncategorized",
                    "size": "Unknown"  # Drive files don't have local size
                })

        # Find groups with multiple files (duplicates)
        duplicate_groups = []
        group_id = 1

        for file_hash, file_list in hash_to_files.items():
            if len(file_list) > 1:
                duplicate_groups.append({
                    "group_id": f"group_{group_id}",
                    "file_count": len(file_list),
                    "files": file_list
                })
                group_id += 1

        conn.close()

        return {
            "summary": {
                "duplicate_groups_found": len(duplicate_groups),
                "total_files_processed": len(files)
            },
            "duplicates": duplicate_groups
        }

    except Exception as e:
        logger.error(f"Error finding duplicates: {e}")
        return {
            "error": str(e),
            "summary": {
                "duplicate_groups_found": 0,
                "total_files_processed": 0
            },
            "duplicates": []
        }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "dedup", "version": "drive_optimized"}
