#!/usr/bin/env python3
"""
Test script to debug the normalize_text_content function
"""

import os
import sys
import re
import hashlib

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

def test_normalize_function():
    """Test the normalize_text_content function with different inputs"""

    # Test 1: Normal text
    print("=== Test 1: Normal text ===")
    text = "Hello world! This is a test."
    normalized = normalize_text_content(text)
    print(f"Input: {repr(text)}")
    print(f"Output: {repr(normalized)}")
    print(f"Hash: {hashlib.sha256(normalized.encode('utf-8')).hexdigest()}")

    # Test 2: Text with special characters
    print("\n=== Test 2: Text with special characters ===")
    text = "Hello\nworld!\tThis is\ra test.\x00\x01\x02"
    normalized = normalize_text_content(text)
    print(f"Input: {repr(text)}")
    print(f"Output: {repr(normalized)}")
    print(f"Hash: {hashlib.sha256(normalized.encode('utf-8')).hexdigest()}")

    # Test 3: Bytes input
    print("\n=== Test 3: Bytes input ===")
    text_bytes = b"Hello world! This is a test."
    normalized = normalize_text_content(text_bytes)
    print(f"Input: {repr(text_bytes)}")
    print(f"Output: {repr(normalized)}")
    print(f"Type: {type(normalized)}")
    if isinstance(normalized, bytes):
        print("Output is bytes - this could be the issue!")
        print(f"String representation: {repr(str(normalized))}")

    # Test 4: Binary file content
    print("\n=== Test 4: Binary file content ===")
    # Create some fake binary content
    binary_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
    normalized = normalize_text_content(binary_content)
    print(f"Input: {repr(binary_content[:20])}...")
    print(f"Output: {repr(normalized)}")
    print(f"Type: {type(normalized)}")
    if isinstance(normalized, bytes):
        print("Output is bytes - this could be the issue!")

    # Test 5: Empty content
    print("\n=== Test 5: Empty content ===")
    empty_bytes = b""
    normalized = normalize_text_content(empty_bytes)
    print(f"Input: {repr(empty_bytes)}")
    print(f"Output: {repr(normalized)}")
    print(f"Type: {type(normalized)}")

if __name__ == "__main__":
    test_normalize_function()
