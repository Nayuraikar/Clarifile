#!/usr/bin/env python3
"""
enhanced_drive_integration.py
Integration script showing how to use the enhanced categorization system with Google Drive workflow.
"""

import os
import json
import requests
import time
from pathlib import Path

class EnhancedDriveIntegration:
    """Integration class for enhanced categorization with Google Drive."""
    
    def __init__(self, gateway_url="http://127.0.0.1:4000"):
        self.gateway_url = gateway_url
        self.embed_service_url = "http://127.0.0.1:8002"
        
    def test_services_health(self):
        """Test if all services are running."""
        print("🔍 Testing service health...")
        
        services = {
            "Gateway": self.gateway_url,
            "Enhanced Embed Service": self.embed_service_url
        }
        
        for name, url in services.items():
            try:
                if name == "Enhanced Embed Service":
                    response = requests.get(f"{url}/health", timeout=5)
                else:
                    response = requests.get(f"{url}/model_info", timeout=5)
                
                if response.status_code == 200:
                    print(f"✅ {name}: Running")
                    if name == "Enhanced Embed Service":
                        data = response.json()
                        print(f"   Model: {data.get('embedding_model', {}).get('name', 'Unknown')}")
                        print(f"   Enhanced: {'Yes' if data.get('categorization_model', {}).get('loaded') else 'No'}")
                else:
                    print(f"❌ {name}: Not responding (status {response.status_code})")
            except Exception as e:
                print(f"❌ {name}: Not reachable ({e})")
    
    def categorize_content_example(self):
        """Example of categorizing content using the enhanced system."""
        print("\n📝 Testing content categorization...")
        
        # Test with your actual file contents
        test_cases = [
            {
                "name": "Invoice Sample",
                "content": """INVOICE
Invoice Number: INV-2024-001
Date: January 15, 2024
Bill To: John Doe Company
Description: Software development services
Total Amount Due: $1,250.00""",
                "expected": "Finance"
            },
            {
                "name": "Personal Journal",
                "content": """Personal Journal Entry - January 15, 2024
Today was quite productive. I spent the morning working on my side project.
Goals for this week:
1. Finish the expense tracker app
2. Read two chapters of "Deep Learning"
Mood: Optimistic and focused""",
                "expected": "Personal"
            }
        ]
        
        for test_case in test_cases:
            print(f"\n🧪 Testing: {test_case['name']}")
            
            try:
                response = requests.post(
                    f"{self.gateway_url}/categorize_content",
                    json={
                        "content": test_case["content"],
                        "use_enhanced": True
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    category = result.get("category", "Unknown")
                    method = result.get("method", "unknown")
                    
                    print(f"   📊 Result: {category}")
                    print(f"   🔧 Method: {method}")
                    print(f"   ✅ Expected: {test_case['expected']} - {'✓' if test_case['expected'].lower() in category.lower() else '✗'}")
                else:
                    print(f"   ❌ Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
    
    def process_file_example(self):
        """Example of processing files with enhanced categorization."""
        print("\n📄 Testing file processing...")
        
        test_files = [
            "test_content_files/invoice_sample.txt",
            "test_content_files/personal_journal.txt"
        ]
        
        for file_path in test_files:
            if not os.path.exists(file_path):
                print(f"   ⚠️  File not found: {file_path}")
                continue
                
            print(f"\n📁 Processing: {os.path.basename(file_path)}")
            
            try:
                response = requests.post(
                    f"{self.gateway_url}/process_file",
                    json={
                        "file_path": os.path.abspath(file_path),
                        "extract_chunks": True,
                        "categorize": True
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    print(f"   📊 Text Length: {result.get('text_length', 0)} chars")
                    print(f"   🧩 Chunks: {result.get('chunks', {}).get('count', 0)}")
                    print(f"   🏷️  Category: {result.get('category', 'Unknown')}")
                    
                    if 'embeddings' in result:
                        emb_info = result['embeddings']
                        print(f"   🔢 Embeddings: {emb_info.get('shape', 'Unknown shape')}")
                else:
                    print(f"   ❌ Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
    
    def batch_categorize_example(self):
        """Example of batch categorization."""
        print("\n📦 Testing batch categorization...")
        
        # Check if test files exist
        test_dir = "test_content_files"
        if not os.path.exists(test_dir):
            print(f"   ⚠️  Test directory not found: {test_dir}")
            return
        
        # Get all files in test directory
        test_files = []
        for file in os.listdir(test_dir):
            file_path = os.path.join(test_dir, file)
            if os.path.isfile(file_path):
                test_files.append(os.path.abspath(file_path))
        
        if not test_files:
            print(f"   ⚠️  No test files found in {test_dir}")
            return
        
        print(f"   📁 Found {len(test_files)} files to categorize")
        
        try:
            response = requests.post(
                f"{self.gateway_url}/batch_categorize",
                json={
                    "file_paths": test_files,
                    "output_dir": os.path.abspath("test_batch_output"),
                    "k": None,  # Auto-determine
                    "chunk_size": 2000,
                    "overlap": 200,
                    "use_hdbscan": False
                },
                timeout=120  # Longer timeout for batch processing
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("success"):
                    print(f"   ✅ Success!")
                    print(f"   📊 Processed: {result.get('processed_files', 0)} files")
                    print(f"   🏷️  Categories: {result.get('categories', 0)}")
                    print(f"   📁 Output: {result.get('output_dir', 'Unknown')}")
                    
                    # Show cluster terms
                    cluster_terms = result.get('cluster_terms', {})
                    if cluster_terms:
                        print(f"   🔤 Category Terms:")
                        for cluster_id, terms in cluster_terms.items():
                            print(f"      Category {cluster_id}: {', '.join(terms)}")
                else:
                    print(f"   ❌ Batch processing failed: {result.get('error', 'Unknown error')}")
            else:
                print(f"   ❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    def google_drive_workflow_simulation(self):
        """Simulate the Google Drive integration workflow."""
        print("\n🔄 Simulating Google Drive workflow...")
        
        # Step 1: Check if we have a saved model
        print("\n1️⃣ Checking for saved categorization model...")
        
        model_dir = "test_batch_output"
        if os.path.exists(os.path.join(model_dir, "centroids.npy")):
            print(f"   ✅ Found saved model at {model_dir}")
            
            # Step 2: Load the model
            print("\n2️⃣ Loading saved model...")
            try:
                response = requests.post(
                    f"{self.gateway_url}/load_categorization_model",
                    json={"model_dir": os.path.abspath(model_dir)},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        print(f"   ✅ Model loaded successfully")
                        print(f"   🎯 Centroids: {'Yes' if result.get('centroids_loaded') else 'No'}")
                        print(f"   📊 TF-IDF: {'Yes' if result.get('tfidf_loaded') else 'No'}")
                    else:
                        print(f"   ❌ Failed to load model")
                else:
                    print(f"   ❌ Error loading model: {response.status_code}")
            except Exception as e:
                print(f"   ❌ Exception: {e}")
            
            # Step 3: Simulate real-time categorization of new Drive files
            print("\n3️⃣ Simulating real-time categorization...")
            
            new_content_examples = [
                "Meeting agenda for Q1 planning session with team leads and stakeholders",
                "Receipt for office supplies purchase - $45.99 total amount due",
                "Personal reminder to call dentist and schedule annual checkup appointment"
            ]
            
            for i, content in enumerate(new_content_examples, 1):
                print(f"\n   📄 New file {i}: {content[:50]}...")
                
                try:
                    response = requests.post(
                        f"{self.gateway_url}/categorize_content",
                        json={"content": content, "use_enhanced": True},
                        timeout=15
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        category = result.get("category", "Unknown")
                        print(f"      🏷️  → {category}")
                    else:
                        print(f"      ❌ Error: {response.status_code}")
                except Exception as e:
                    print(f"      ❌ Exception: {e}")
        else:
            print(f"   ⚠️  No saved model found. Run batch categorization first.")
    
    def integration_recommendations(self):
        """Provide integration recommendations."""
        print("\n💡 INTEGRATION RECOMMENDATIONS")
        print("="*50)
        
        recommendations = [
            {
                "component": "Browser Extension",
                "action": "Update to call /categorize_content endpoint",
                "benefit": "Real-time categorization as files are added to Drive"
            },
            {
                "component": "Parser Service", 
                "action": "Replace existing categorizer with EnhancedCategorizer",
                "benefit": "Better content understanding with chunking and TF-IDF"
            },
            {
                "component": "UI Components",
                "action": "Add category confidence scores and review interface",
                "benefit": "Human-in-the-loop improvement and quality control"
            },
            {
                "component": "Database Schema",
                "action": "Add tables for chunk embeddings and category metadata",
                "benefit": "Better search and incremental learning capabilities"
            },
            {
                "component": "Batch Processing",
                "action": "Set up periodic batch categorization jobs",
                "benefit": "Keep categorization model updated with new content"
            }
        ]
        
        for rec in recommendations:
            print(f"\n🔧 {rec['component']}")
            print(f"   Action: {rec['action']}")
            print(f"   Benefit: {rec['benefit']}")

def main():
    """Run the enhanced Drive integration demonstration."""
    print("🚀 ENHANCED CATEGORIZATION - GOOGLE DRIVE INTEGRATION")
    print("="*60)
    
    integration = EnhancedDriveIntegration()
    
    # Run all tests and demonstrations
    integration.test_services_health()
    integration.categorize_content_example()
    integration.process_file_example()
    integration.batch_categorize_example()
    integration.google_drive_workflow_simulation()
    integration.integration_recommendations()
    
    print("\n" + "="*60)
    print("🎉 INTEGRATION DEMONSTRATION COMPLETE!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. Start the enhanced embed service:")
    print("   python services/embed/enhanced_app.py")
    print()
    print("2. Update your browser extension to use new endpoints:")
    print("   POST /categorize_content - for real-time categorization")
    print("   POST /process_file - for comprehensive file processing")
    print()
    print("3. Run initial batch categorization on your Drive files:")
    print("   python smart_categorize_v2.py --source ./your_drive_files --dest ./categorized_model")
    print()
    print("4. Load the model in your service:")
    print("   POST /load_categorization_model with model directory")
    print()
    print("5. Use incremental classification for new files:")
    print("   python incremental_classifier.py --new_files ./new_drive_files --model_dir ./categorized_model")

if __name__ == "__main__":
    main()
