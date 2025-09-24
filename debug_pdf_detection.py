#!/usr/bin/env python3
"""
Debug script to test PDF detection and hashing
"""

import os
import sys
sys.path.append('/Users/nayana/Desktop/PROGRAMS/clarifile')

from services.dedup.app import is_text_or_pdf_file, compute_exact_content_hash, compute_normalized_text_hash

def test_pdf_detection():
    """Test PDF file detection"""
    print("=== Testing PDF Detection ===")

    # Test with a sample PDF path (you'll need to update this)
    test_pdf_path = "/Users/nayana/Desktop/PROGRAMS/clarifile/test.pdf"  # Update this path

    if os.path.exists(test_pdf_path):
        print(f"Testing file: {test_pdf_path}")
        is_pdf = is_text_or_pdf_file(test_pdf_path)
        print(f"Is PDF: {is_pdf}")

        if is_pdf:
            exact_hash = compute_exact_content_hash(test_pdf_path)
            normalized_hash = compute_normalized_text_hash(test_pdf_path)
            print(f"Exact hash: {exact_hash}")
            print(f"Normalized hash: {normalized_hash}")
        else:
            print("File not detected as PDF")
    else:
        print(f"Test file not found: {test_pdf_path}")
        print("Please update the test_pdf_path with an actual PDF file path")

    # Test with .pdf extension
    fake_pdf_path = "/fake/path/test.pdf"
    is_fake_pdf = is_text_or_pdf_file(fake_pdf_path)
    print(f"Fake PDF path detected: {is_fake_pdf}")

if __name__ == "__main__":
    test_pdf_detection()
