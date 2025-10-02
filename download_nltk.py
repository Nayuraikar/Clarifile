#!/usr/bin/env python3
"""
Script to download required NLTK resources with proper error handling.
"""

import nltk
import sys

def download_nltk_resources():
    """Download required NLTK resources with fallback handling."""
    resources_to_download = [
        ('punkt_tab', 'tokenizers/punkt_tab'),
        ('punkt', 'tokenizers/punkt'),  # fallback
        ('stopwords', 'corpora/stopwords'),
        ('wordnet', 'corpora/wordnet')
    ]
    
    punkt_downloaded = False
    
    print("Checking and downloading NLTK resources...")
    
    # Try punkt_tab first, then punkt as fallback
    for punkt_name, punkt_path in [('punkt_tab', 'tokenizers/punkt_tab'), ('punkt', 'tokenizers/punkt')]:
        try:
            nltk.data.find(punkt_path)
            print(f"✓ {punkt_name} already available")
            punkt_downloaded = True
            break
        except LookupError:
            try:
                print(f"Downloading {punkt_name}...")
                nltk.download(punkt_name, quiet=False)
                print(f"✓ {punkt_name} downloaded successfully")
                punkt_downloaded = True
                break
            except Exception as e:
                print(f"✗ Failed to download {punkt_name}: {e}")
                continue
    
    if not punkt_downloaded:
        print("✗ Failed to download punkt tokenizer")
        return False
    
    # Download other resources
    success = True
    for resource_name, resource_path in [('stopwords', 'corpora/stopwords'), ('wordnet', 'corpora/wordnet')]:
        try:
            nltk.data.find(resource_path)
            print(f"✓ {resource_name} already available")
        except LookupError:
            try:
                print(f"Downloading {resource_name}...")
                nltk.download(resource_name, quiet=False)
                print(f"✓ {resource_name} downloaded successfully")
            except Exception as e:
                print(f"✗ Failed to download {resource_name}: {e}")
                success = False
    
    return success

if __name__ == "__main__":
    try:
        success = download_nltk_resources()
        if success:
            print("\n✓ All NLTK resources are ready!")
            sys.exit(0)
        else:
            print("\n✗ Some NLTK resources failed to download")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
