#!/usr/bin/env python3
"""
test_enhanced_categorization_v2.py
Test script for the enhanced categorization system with chunking and TF-IDF weighting.
"""

import os
import sys
import json
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

from smart_categorize_v2 import EnhancedCategorizer, extract_text, chunk_text
from incremental_classifier import classify_new_file, load_classification_model
from sentence_transformers import SentenceTransformer

def test_text_extraction():
    """Test text extraction from various file types."""
    print("="*60)
    print("TESTING TEXT EXTRACTION")
    print("="*60)
    
    test_files = [
        "test_content_files/invoice_sample.txt",
        "test_content_files/personal_journal.txt"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\nExtracting from: {file_path}")
            text = extract_text(file_path)
            print(f"Extracted {len(text)} characters")
            print(f"Preview: {text[:200]}...")
            
            # Test chunking
            chunks = chunk_text(text, chunk_size=500, overlap=50)
            print(f"Created {len(chunks)} chunks")
            if chunks:
                print(f"First chunk: {chunks[0][:100]}...")
        else:
            print(f"File not found: {file_path}")

def test_enhanced_categorization():
    """Test the enhanced categorization system."""
    print("\n" + "="*60)
    print("TESTING ENHANCED CATEGORIZATION")
    print("="*60)
    
    # Initialize the enhanced categorizer
    categorizer = EnhancedCategorizer()
    
    # Test content samples
    test_contents = [
        {
            "content": """
            Abstract: This research paper presents a comprehensive analysis of machine learning algorithms
            for document classification. The study employs various methodologies including supervised
            learning techniques and neural networks. Results indicate significant improvements in accuracy
            when using transformer-based models. The findings contribute to the literature on automated
            document processing and have implications for information retrieval systems.
            """,
            "expected_category": "Academic/Research"
        },
        {
            "content": """
            INVOICE #12345
            Date: 2024-01-15
            Bill To: John Doe
            123 Main Street
            
            Description: Software License
            Amount: $299.99
            Tax: $24.00
            Total: $323.99
            
            Payment due within 30 days.
            """,
            "expected_category": "Finance/Invoice"
        },
        {
            "content": """
            Meeting Minutes - Project Alpha
            Date: January 15, 2024
            Attendees: Alice, Bob, Carol
            
            Agenda:
            1. Project status update
            2. Budget review
            3. Next milestones
            
            Action Items:
            - Alice to complete design mockups by Friday
            - Bob to review technical specifications
            - Carol to schedule client meeting
            """,
            "expected_category": "Work/Meeting"
        },
        {
            "content": """
            Dear Diary,
            
            Today was an interesting day. I went for a walk in the park and reflected on my goals
            for this year. I'm thinking about learning a new language and maybe taking up photography.
            The weather was beautiful, and I felt grateful for the simple pleasures in life.
            
            Tomorrow I plan to start reading that book I bought last month.
            """,
            "expected_category": "Personal/Notes"
        }
    ]
    
    print("Testing content categorization:")
    for i, test_case in enumerate(test_contents, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Content preview: {test_case['content'][:100]}...")
        print(f"Expected: {test_case['expected_category']}")
        
        category = categorizer.categorize_content(test_case['content'])
        print(f"Predicted: {category}")
        
        # Simple match check
        expected_lower = test_case['expected_category'].lower()
        predicted_lower = category.lower()
        
        match_score = 0
        for word in expected_lower.split('/'):
            if word in predicted_lower:
                match_score += 1
        
        print(f"Match score: {match_score}/{len(expected_lower.split('/'))}")

def test_batch_categorization():
    """Test batch categorization on available test files."""
    print("\n" + "="*60)
    print("TESTING BATCH CATEGORIZATION")
    print("="*60)
    
    # Check if we have test files
    test_dir = "test_content_files"
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found. Skipping batch test.")
        return
    
    # Collect test files
    test_files = []
    for file in os.listdir(test_dir):
        file_path = os.path.join(test_dir, file)
        if os.path.isfile(file_path):
            test_files.append(file_path)
    
    if not test_files:
        print("No test files found. Skipping batch test.")
        return
    
    print(f"Found {len(test_files)} test files:")
    for f in test_files:
        print(f"  - {f}")
    
    # Run batch categorization
    try:
        from smart_categorize_v2 import main as categorize_main
        import argparse
        
        # Create a mock args object
        class MockArgs:
            def __init__(self):
                self.source = test_dir
                self.dest = "test_output_enhanced"
                self.k = 0  # Auto-determine
                self.chunk_size = 2000
                self.overlap = 200
                self.use_hdbscan = False  # Use KMeans for consistency
        
        args = MockArgs()
        
        print(f"\nRunning batch categorization...")
        print(f"Source: {args.source}")
        print(f"Destination: {args.dest}")
        
        # Import and run the main function
        from smart_categorize_v2 import main
        main(args)
        
        # Check results
        if os.path.exists(args.dest):
            print(f"\nResults saved to: {args.dest}")
            
            # List created categories
            categories = [d for d in os.listdir(args.dest) if os.path.isdir(os.path.join(args.dest, d))]
            print(f"Created {len(categories)} categories:")
            for cat in categories:
                cat_path = os.path.join(args.dest, cat)
                files_in_cat = [f for f in os.listdir(cat_path) if os.path.isfile(os.path.join(cat_path, f))]
                print(f"  - {cat}: {len(files_in_cat)} files")
            
            # Load and display metadata
            metadata_path = os.path.join(args.dest, "categories_map.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    mapping = json.load(f)
                print(f"\nFile mappings:")
                for file_path, category in mapping.items():
                    print(f"  {os.path.basename(file_path)} -> {category}")
        
    except Exception as e:
        print(f"Error in batch categorization: {e}")
        import traceback
        traceback.print_exc()

def test_incremental_classification():
    """Test incremental classification using saved model."""
    print("\n" + "="*60)
    print("TESTING INCREMENTAL CLASSIFICATION")
    print("="*60)
    
    model_dir = "test_output_enhanced"
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} not found. Run batch categorization first.")
        return
    
    # Check if model files exist
    required_files = ["centroids.npy", "classification_metadata.json"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    
    if missing_files:
        print(f"Missing model files: {missing_files}")
        return
    
    try:
        # Load the model
        print("Loading saved classification model...")
        centroids, metadata, tfidf_vectorizer, category_names = load_classification_model(model_dir)
        
        print(f"Loaded model with {len(centroids)} centroids")
        print(f"TF-IDF vectorizer: {'loaded' if tfidf_vectorizer else 'not available'}")
        
        # Load sentence transformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Test classification on some sample content
        test_samples = [
            "This is a research paper about artificial intelligence and machine learning algorithms.",
            "Invoice #123 - Payment due: $500.00 - Customer: ABC Corp",
            "Meeting agenda: Discuss project timeline and budget allocation for Q2.",
            "Personal note: Remember to buy groceries and call mom this weekend."
        ]
        
        print("\nTesting incremental classification:")
        for i, content in enumerate(test_samples, 1):
            print(f"\n--- Sample {i} ---")
            print(f"Content: {content}")
            
            # Create a temporary file for testing
            temp_file = f"temp_test_{i}.txt"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            try:
                result = classify_new_file(
                    temp_file, model, centroids, 
                    vec_for_weights=tfidf_vectorizer,
                    similarity_threshold=0.6
                )
                
                print(f"Classification: {result['label']}")
                print(f"Confidence: {result['score']:.3f}")
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
    except Exception as e:
        print(f"Error in incremental classification: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("ENHANCED CATEGORIZATION SYSTEM TEST SUITE")
    print("="*60)
    
    try:
        # Test 1: Text extraction and chunking
        test_text_extraction()
        
        # Test 2: Enhanced categorization
        test_enhanced_categorization()
        
        # Test 3: Batch categorization
        test_batch_categorization()
        
        # Test 4: Incremental classification
        test_incremental_classification()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        print("\nNext steps:")
        print("1. Install additional dependencies: pip install -r requirements_enhanced_categorization.txt")
        print("2. Run batch categorization on your Google Drive files")
        print("3. Use incremental classification for new files")
        print("4. Integrate with the enhanced embed service")
        
    except Exception as e:
        print(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
