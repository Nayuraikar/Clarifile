#!/usr/bin/env python3
"""
incremental_classifier.py
Classify new files using previously saved centroids and copy them into existing category folders.

Usage:
  python incremental_classifier.py --new_files ./new_docs --model_dir ./sorted_output --output ./classified --threshold 0.65
"""

import os
import argparse
import json
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from smart_categorize_v2 import classify_new_file, extract_text, chunk_text

def load_classification_model(model_dir):
    """Load saved centroids, metadata, and TF-IDF vectorizer."""
    centroids_path = os.path.join(model_dir, "centroids.npy")
    metadata_path = os.path.join(model_dir, "classification_metadata.json")
    tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    categories_path = os.path.join(model_dir, "categories_map.json")
    
    if not os.path.exists(centroids_path):
        raise FileNotFoundError(f"Centroids not found at {centroids_path}")
    
    centroids = np.load(centroids_path)
    
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    tfidf_vectorizer = None
    if os.path.exists(tfidf_path):
        try:
            import joblib
            tfidf_vectorizer = joblib.load(tfidf_path)
        except Exception as e:
            print(f"Warning: Could not load TF-IDF vectorizer: {e}")
    
    # Load category names from the original categorization
    category_names = {}
    if os.path.exists(categories_path):
        with open(categories_path, 'r') as f:
            categories_map = json.load(f)
            # Extract unique category names
            unique_categories = list(set(categories_map.values()))
            for i, cat_name in enumerate(unique_categories):
                category_names[i] = cat_name
    
    return centroids, metadata, tfidf_vectorizer, category_names

def classify_files_batch(files, model, centroids, tfidf_vectorizer=None, 
                        chunk_size=2000, overlap=200, similarity_threshold=0.65):
    """Classify multiple files in batch."""
    results = []
    
    print(f"Classifying {len(files)} files...")
    for file_path in tqdm(files):
        try:
            result = classify_new_file(
                file_path, model, centroids, 
                vec_for_weights=tfidf_vectorizer,
                chunk_size=chunk_size, 
                overlap=overlap,
                similarity_threshold=similarity_threshold
            )
            results.append(result)
        except Exception as e:
            print(f"Error classifying {file_path}: {e}")
            results.append({
                "label": "error", 
                "score": 0.0, 
                "path": file_path, 
                "error": str(e)
            })
    
    return results

def organize_classified_files(results, output_dir, category_names, model_dir):
    """Copy classified files into appropriate category folders."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create review folder for uncertain classifications
    review_dir = os.path.join(output_dir, "review")
    os.makedirs(review_dir, exist_ok=True)
    
    # Get existing category folder names from model_dir
    existing_categories = {}
    if os.path.exists(model_dir):
        for item in os.listdir(model_dir):
            item_path = os.path.join(model_dir, item)
            if os.path.isdir(item_path) and item.startswith("cat"):
                # Extract category number from folder name like "cat0_research_paper_analysis"
                try:
                    cat_num = int(item.split("_")[0][3:])  # Extract number after "cat"
                    existing_categories[cat_num] = item
                except:
                    pass
    
    stats = {
        "classified": 0,
        "review": 0,
        "error": 0,
        "no_text": 0
    }
    
    classification_log = []
    
    for result in results:
        file_path = result["path"]
        label = result["label"]
        score = result["score"]
        
        log_entry = {
            "file": os.path.basename(file_path),
            "full_path": file_path,
            "label": label,
            "score": score
        }
        
        try:
            if label == "review":
                # Copy to review folder
                dest_dir = review_dir
                shutil.copy2(file_path, dest_dir)
                stats["review"] += 1
                log_entry["destination"] = "review"
                
            elif label == "no_text":
                # Copy to review folder with special subfolder
                no_text_dir = os.path.join(review_dir, "no_text")
                os.makedirs(no_text_dir, exist_ok=True)
                shutil.copy2(file_path, no_text_dir)
                stats["no_text"] += 1
                log_entry["destination"] = "review/no_text"
                
            elif label == "error":
                # Copy to review folder with error subfolder
                error_dir = os.path.join(review_dir, "error")
                os.makedirs(error_dir, exist_ok=True)
                shutil.copy2(file_path, error_dir)
                stats["error"] += 1
                log_entry["destination"] = "review/error"
                
            elif isinstance(label, int):
                # Classified into existing category
                if label in existing_categories:
                    category_folder = existing_categories[label]
                    dest_dir = os.path.join(output_dir, category_folder)
                else:
                    # Create new category folder
                    category_folder = f"cat{label}_new_category"
                    dest_dir = os.path.join(output_dir, category_folder)
                
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(file_path, dest_dir)
                stats["classified"] += 1
                log_entry["destination"] = category_folder
                
            else:
                # Unknown label, send to review
                shutil.copy2(file_path, review_dir)
                stats["review"] += 1
                log_entry["destination"] = "review"
                
        except Exception as e:
            print(f"Error copying {file_path}: {e}")
            log_entry["error"] = str(e)
        
        classification_log.append(log_entry)
    
    # Save classification log
    with open(os.path.join(output_dir, "classification_log.json"), "w") as f:
        json.dump(classification_log, f, indent=2)
    
    return stats, classification_log

def main():
    parser = argparse.ArgumentParser(description="Classify new files using saved model")
    parser.add_argument("--new_files", "-n", required=True, 
                       help="Directory containing new files to classify")
    parser.add_argument("--model_dir", "-m", required=True,
                       help="Directory containing saved model (centroids, metadata)")
    parser.add_argument("--output", "-o", required=True,
                       help="Output directory for classified files")
    parser.add_argument("--threshold", "-t", type=float, default=0.65,
                       help="Similarity threshold for classification (default: 0.65)")
    parser.add_argument("--chunk_size", type=int, default=2000,
                       help="Chunk size for text processing")
    parser.add_argument("--overlap", type=int, default=200,
                       help="Overlap between chunks")
    parser.add_argument("--model_name", default="all-MiniLM-L6-v2",
                       help="Sentence transformer model name")
    
    args = parser.parse_args()
    
    # Load the classification model
    print("Loading classification model...")
    try:
        centroids, metadata, tfidf_vectorizer, category_names = load_classification_model(args.model_dir)
        print(f"Loaded model with {len(centroids)} categories")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load sentence transformer
    print(f"Loading sentence transformer model: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    
    # Collect files to classify
    new_files = []
    for root, _, fnames in os.walk(args.new_files):
        for f in fnames:
            new_files.append(os.path.join(root, f))
    
    if not new_files:
        print(f"No files found in {args.new_files}")
        return
    
    print(f"Found {len(new_files)} files to classify")
    
    # Classify files
    results = classify_files_batch(
        new_files, model, centroids, tfidf_vectorizer,
        chunk_size=args.chunk_size, overlap=args.overlap,
        similarity_threshold=args.threshold
    )
    
    # Organize classified files
    print("Organizing classified files...")
    stats, log = organize_classified_files(results, args.output, category_names, args.model_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("CLASSIFICATION SUMMARY")
    print("="*50)
    print(f"Total files processed: {len(new_files)}")
    print(f"Successfully classified: {stats['classified']}")
    print(f"Sent to review: {stats['review']}")
    print(f"No text extracted: {stats['no_text']}")
    print(f"Errors: {stats['error']}")
    print(f"\nResults saved to: {args.output}")
    print(f"Classification log: {os.path.join(args.output, 'classification_log.json')}")
    
    # Show some example classifications
    print("\nExample classifications:")
    for i, entry in enumerate(log[:5]):
        print(f"  {entry['file']} -> {entry['destination']} (score: {entry['score']:.3f})")
    
    if stats['review'] > 0:
        print(f"\nNote: {stats['review']} files were sent to 'review' folder for manual inspection.")
        print("Consider adjusting the similarity threshold or expanding your training set.")

if __name__ == "__main__":
    main()
