#!/usr/bin/env python3
"""
demo_with_your_files.py
Practical demonstration of the enhanced categorization system using your actual test files.
"""

import os
import sys
import json
from pathlib import Path

def demo_text_extraction_and_chunking():
    """Demonstrate text extraction and chunking on your actual files."""
    print("="*70)
    print("DEMO: TEXT EXTRACTION & CHUNKING WITH YOUR FILES")
    print("="*70)
    
    # Mock the functions since we don't have dependencies installed yet
    def mock_extract_text(file_path):
        """Mock text extraction - read the actual file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading {file_path}: {e}"
    
    def mock_chunk_text(text, chunk_size=500, overlap=50):
        """Mock chunking function."""
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            if end == len(text):
                break
            start = max(0, end - overlap)
        return chunks
    
    test_files = [
        "test_content_files/invoice_sample.txt",
        "test_content_files/personal_journal.txt"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n📄 Processing: {file_path}")
            print("-" * 50)
            
            # Extract text
            text = mock_extract_text(file_path)
            print(f"📊 Extracted {len(text)} characters")
            print(f"📝 Preview: {text[:100]}...")
            
            # Create chunks
            chunks = mock_chunk_text(text, chunk_size=300, overlap=30)
            print(f"🧩 Created {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
                print(f"\n   Chunk {i+1}: {chunk[:80]}...")
        else:
            print(f"❌ File not found: {file_path}")

def demo_enhanced_categorization():
    """Demonstrate enhanced categorization logic on your files."""
    print("\n" + "="*70)
    print("DEMO: ENHANCED CATEGORIZATION LOGIC")
    print("="*70)
    
    # Read your actual file contents
    files_content = {}
    test_files = [
        "test_content_files/invoice_sample.txt",
        "test_content_files/personal_journal.txt"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                files_content[file_path] = f.read()
    
    # Mock enhanced categorization logic
    def mock_categorize_content(content, file_name):
        """Mock the enhanced categorization logic."""
        content_lower = content.lower()
        
        # Financial Document Detection
        financial_terms = ['invoice', 'bill', 'payment', 'amount', 'total', 'due', 'subtotal', 'tax']
        financial_score = sum(1 for term in financial_terms if term in content_lower)
        
        if financial_score >= 3:
            return "Finance: Invoice", financial_score, financial_terms
        
        # Personal Document Detection  
        personal_terms = ['personal', 'journal', 'thoughts', 'goals', 'mood', 'diary', 'note to self']
        personal_score = sum(1 for term in personal_terms if term in content_lower)
        
        if personal_score >= 2:
            return "Personal: Journal", personal_score, personal_terms
        
        # Business/Work Document Detection
        business_terms = ['meeting', 'project', 'team', 'presentation', 'business', 'development']
        business_score = sum(1 for term in business_terms if term in content_lower)
        
        if business_score >= 2:
            return "Work: Project", business_score, business_terms
        
        return "General: Document", 0, []
    
    print("🤖 Categorizing your files:")
    
    for file_path, content in files_content.items():
        print(f"\n📄 File: {os.path.basename(file_path)}")
        print("-" * 40)
        
        category, score, matching_terms = mock_categorize_content(content, file_path)
        
        print(f"🏷️  Category: {category}")
        print(f"📊 Confidence Score: {score}")
        print(f"🔍 Matching Terms: {matching_terms[:5]}")  # Show first 5 matching terms
        
        # Show why it was categorized this way
        content_preview = content[:200].replace('\n', ' ')
        print(f"📝 Content Preview: {content_preview}...")

def demo_api_integration():
    """Show how this integrates with your existing Clarifile services."""
    print("\n" + "="*70)
    print("DEMO: API INTEGRATION WITH CLARIFILE SERVICES")
    print("="*70)
    
    # Mock API calls that would work with your enhanced service
    api_examples = [
        {
            "endpoint": "POST /categorize_content",
            "description": "Categorize text content",
            "request": {
                "content": "INVOICE #123 - Amount Due: $1,250.00",
                "use_enhanced": True
            },
            "response": {
                "category": "Finance: Invoice",
                "method": "enhanced",
                "content_length": 35
            }
        },
        {
            "endpoint": "POST /process_file", 
            "description": "Process a file completely",
            "request": {
                "file_path": "test_content_files/personal_journal.txt",
                "extract_chunks": True,
                "categorize": True
            },
            "response": {
                "file_path": "test_content_files/personal_journal.txt",
                "text_length": 1234,
                "chunks": {"count": 3},
                "category": "Personal: Journal"
            }
        },
        {
            "endpoint": "POST /batch_categorize",
            "description": "Batch process multiple files",
            "request": {
                "file_paths": ["test_content_files/invoice_sample.txt", "test_content_files/personal_journal.txt"],
                "output_dir": "./categorized_output",
                "k": None
            },
            "response": {
                "success": True,
                "processed_files": 2,
                "categories": 2,
                "cluster_terms": {
                    "0": ["invoice", "payment", "business"],
                    "1": ["personal", "journal", "thoughts"]
                }
            }
        }
    ]
    
    for example in api_examples:
        print(f"\n🌐 {example['endpoint']}")
        print(f"📋 {example['description']}")
        print("📤 Request:")
        print(json.dumps(example['request'], indent=2))
        print("📥 Response:")
        print(json.dumps(example['response'], indent=2))

def demo_integration_with_existing_services():
    """Show how to integrate with your existing Clarifile architecture."""
    print("\n" + "="*70)
    print("DEMO: INTEGRATION WITH EXISTING CLARIFILE SERVICES")
    print("="*70)
    
    print("🏗️  Integration Points:")
    print()
    
    # Check existing services
    services = {
        "Parser Service (port 8000)": "services/parser",
        "Embed Service (port 8002)": "services/embed", 
        "Indexer Service (port 8003)": "services/indexer",
        "Dedup Service (port 8004)": "services/dedup",
        "Gateway (port 4000)": "gateway"
    }
    
    for service_name, path in services.items():
        status = "✅ Found" if os.path.exists(path) else "❌ Not Found"
        print(f"{status} {service_name}")
        
        if "Parser" in service_name and os.path.exists(path):
            print("   🔧 Can integrate enhanced categorization here")
        elif "Embed" in service_name and os.path.exists(path):
            print("   🔧 Enhanced service ready at services/embed/enhanced_app.py")
        elif "Gateway" in service_name and os.path.exists(path):
            print("   🔧 Can route to enhanced categorization endpoints")
    
    print("\n📋 Integration Steps:")
    print("1. Update Parser Service to use EnhancedCategorizer")
    print("2. Start Enhanced Embed Service on port 8002")
    print("3. Update Gateway to route categorization requests")
    print("4. Update UI to show enhanced categories")
    print("5. Update Browser Extension for real-time categorization")

def demo_google_drive_workflow():
    """Demonstrate the Google Drive integration workflow."""
    print("\n" + "="*70)
    print("DEMO: GOOGLE DRIVE INTEGRATION WORKFLOW")
    print("="*70)
    
    workflow_steps = [
        {
            "step": "1. Initial Batch Processing",
            "description": "Process existing Google Drive files",
            "command": "python smart_categorize_v2.py --source ./drive_files --dest ./categorized_drive",
            "result": "Creates categories and saves centroids for future use"
        },
        {
            "step": "2. Real-time Classification", 
            "description": "Classify new files as they're added to Drive",
            "command": "API call to /categorize_content endpoint",
            "result": "Instant categorization using saved model"
        },
        {
            "step": "3. Incremental Updates",
            "description": "Process new batches of files periodically", 
            "command": "python incremental_classifier.py --new_files ./new_drive_files --model_dir ./categorized_drive",
            "result": "Files organized into existing categories or sent to review"
        },
        {
            "step": "4. Human Review",
            "description": "Review uncertain classifications",
            "command": "Manual review of files in 'review' folder",
            "result": "Improved accuracy and model refinement"
        }
    ]
    
    for workflow in workflow_steps:
        print(f"\n{workflow['step']}")
        print(f"📋 {workflow['description']}")
        print(f"💻 Command: {workflow['command']}")
        print(f"✅ Result: {workflow['result']}")

def main():
    """Run the complete demonstration."""
    print("🚀 ENHANCED CATEGORIZATION SYSTEM DEMONSTRATION")
    print("Using your actual Clarifile test files")
    print("="*70)
    
    # Run all demonstrations
    demo_text_extraction_and_chunking()
    demo_enhanced_categorization()
    demo_api_integration()
    demo_integration_with_existing_services()
    demo_google_drive_workflow()
    
    print("\n" + "="*70)
    print("🎉 DEMONSTRATION COMPLETE!")
    print("="*70)
    
    print("\n📋 Next Steps to Get Started:")
    print("1. Install dependencies:")
    print("   pip install sentence-transformers scikit-learn pdfplumber python-docx")
    print()
    print("2. Test with your files:")
    print("   python smart_categorize_v2.py --source test_content_files --dest test_categorized")
    print()
    print("3. Start enhanced service:")
    print("   python services/embed/enhanced_app.py")
    print()
    print("4. Update your existing services to use the enhanced categorization")
    print()
    print("5. Integrate with your Google Drive browser extension")
    
    print("\n🎯 Expected Results:")
    print("• invoice_sample.txt → Finance: Invoice category")
    print("• personal_journal.txt → Personal: Journal category")
    print("• Real-time categorization via API")
    print("• Incremental classification for new files")
    print("• Seamless integration with existing Clarifile architecture")

if __name__ == "__main__":
    main()
