#!/usr/bin/env python3
"""
start_enhanced_system.py
Startup script to initialize and test the enhanced categorization system with Clarifile.
"""

import os
import sys
import subprocess
import time
import requests
import json
from pathlib import Path

class EnhancedSystemStarter:
    """Helper class to start and configure the enhanced categorization system."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.services = {
            "gateway": {"port": 4000, "path": "gateway/index.js", "cmd": ["node", "index.js"]},
            "enhanced_embed": {"port": 8002, "path": "services/embed/enhanced_app.py", "cmd": ["python", "enhanced_app.py"]}
        }
        
    def check_dependencies(self):
        """Check if required dependencies are installed."""
        print("🔍 Checking dependencies...")
        
        required_packages = [
            "sentence-transformers",
            "scikit-learn", 
            "numpy",
            "fastapi",
            "uvicorn"
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package} - MISSING")
                missing.append(package)
        
        if missing:
            print(f"\n⚠️  Missing packages: {', '.join(missing)}")
            print("Install with: pip install " + " ".join(missing))
            return False
        
        return True
    
    def check_file_structure(self):
        """Verify all required files are present."""
        print("\n📁 Checking file structure...")
        
        required_files = [
            "smart_categorize_v2.py",
            "incremental_classifier.py",
            "services/embed/enhanced_app.py",
            "gateway/index.js",
            "requirements_enhanced_categorization.txt"
        ]
        
        missing = []
        for file_path in required_files:
            full_path = self.base_dir / file_path
            if full_path.exists():
                print(f"   ✅ {file_path}")
            else:
                print(f"   ❌ {file_path} - MISSING")
                missing.append(file_path)
        
        if missing:
            print(f"\n⚠️  Missing files: {', '.join(missing)}")
            return False
        
        return True
    
    def start_service(self, service_name):
        """Start a specific service."""
        service = self.services[service_name]
        service_path = self.base_dir / service["path"]
        
        if not service_path.exists():
            print(f"   ❌ Service file not found: {service_path}")
            return None
        
        try:
            # Change to service directory
            service_dir = service_path.parent
            
            print(f"   🚀 Starting {service_name} on port {service['port']}...")
            
            # Start the service
            process = subprocess.Popen(
                service["cmd"],
                cwd=service_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give it a moment to start
            time.sleep(3)
            
            # Check if it's running
            if process.poll() is None:
                print(f"   ✅ {service_name} started (PID: {process.pid})")
                return process
            else:
                stdout, stderr = process.communicate()
                print(f"   ❌ {service_name} failed to start")
                print(f"      stdout: {stdout[:200]}")
                print(f"      stderr: {stderr[:200]}")
                return None
                
        except Exception as e:
            print(f"   ❌ Error starting {service_name}: {e}")
            return None
    
    def test_service_health(self, service_name, port):
        """Test if a service is responding."""
        try:
            if service_name == "enhanced_embed":
                url = f"http://127.0.0.1:{port}/health"
            else:
                url = f"http://127.0.0.1:{port}/model_info"
            
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                print(f"   ✅ {service_name} is healthy")
                return True
            else:
                print(f"   ⚠️  {service_name} responded with status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ {service_name} health check failed: {e}")
            return False
    
    def run_initial_categorization(self):
        """Run initial categorization on test files."""
        print("\n📊 Running initial categorization test...")
        
        test_dir = self.base_dir / "test_content_files"
        output_dir = self.base_dir / "initial_categorization_output"
        
        if not test_dir.exists():
            print(f"   ⚠️  Test directory not found: {test_dir}")
            return False
        
        # Check if we have test files
        test_files = list(test_dir.glob("*.txt"))
        if not test_files:
            print(f"   ⚠️  No test files found in {test_dir}")
            return False
        
        print(f"   📁 Found {len(test_files)} test files")
        
        try:
            # Run the categorization script
            cmd = [
                sys.executable, "smart_categorize_v2.py",
                "--source", str(test_dir),
                "--dest", str(output_dir),
                "--k", "0"  # Auto-determine
            ]
            
            print(f"   🔄 Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                print(f"   ✅ Categorization completed successfully")
                
                # Check output
                if output_dir.exists():
                    categories = [d for d in output_dir.iterdir() if d.is_dir()]
                    print(f"   📂 Created {len(categories)} categories:")
                    for cat in categories[:5]:  # Show first 5
                        files_in_cat = list(cat.glob("*"))
                        print(f"      - {cat.name}: {len(files_in_cat)} files")
                
                return True
            else:
                print(f"   ❌ Categorization failed:")
                print(f"      stdout: {result.stdout[:300]}")
                print(f"      stderr: {result.stderr[:300]}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Categorization timed out")
            return False
        except Exception as e:
            print(f"   ❌ Error running categorization: {e}")
            return False
    
    def test_api_endpoints(self):
        """Test the enhanced API endpoints."""
        print("\n🌐 Testing API endpoints...")
        
        gateway_url = "http://127.0.0.1:4000"
        
        # Test categorize_content endpoint
        print("   📝 Testing /categorize_content...")
        try:
            response = requests.post(
                f"{gateway_url}/categorize_content",
                json={
                    "content": "INVOICE #123 - Amount Due: $500.00 - Payment Terms: Net 30",
                    "use_enhanced": True
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                category = result.get("category", "Unknown")
                print(f"      ✅ Response: {category}")
            else:
                print(f"      ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"      ❌ Exception: {e}")
        
        # Test model_info endpoint
        print("   ℹ️  Testing /model_info...")
        try:
            response = requests.get(f"{gateway_url}/model_info", timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                embed_model = result.get("embedding_model", {})
                cat_model = result.get("categorization_model", {})
                
                print(f"      ✅ Embedding model: {embed_model.get('name', 'Unknown')}")
                print(f"      ✅ Categorization loaded: {cat_model.get('loaded', False)}")
            else:
                print(f"      ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"      ❌ Exception: {e}")
    
    def create_startup_summary(self):
        """Create a summary of the startup process."""
        print("\n" + "="*60)
        print("🎉 ENHANCED CATEGORIZATION SYSTEM STARTUP COMPLETE!")
        print("="*60)
        
        print("\n📋 System Status:")
        print("✅ Enhanced categorization system implemented")
        print("✅ API endpoints available through gateway")
        print("✅ Initial test categorization completed")
        print("✅ Services health checked")
        
        print("\n🌐 Available Endpoints:")
        endpoints = [
            "POST /categorize_content - Categorize text content",
            "POST /process_file - Process and categorize a file", 
            "POST /batch_categorize - Batch process multiple files",
            "POST /load_categorization_model - Load saved model",
            "GET /model_info - Get model information",
            "GET /list_models - List available models"
        ]
        
        for endpoint in endpoints:
            print(f"   • {endpoint}")
        
        print("\n🚀 Next Steps:")
        print("1. Update your browser extension to use the new endpoints")
        print("2. Run batch categorization on your Google Drive files:")
        print("   python smart_categorize_v2.py --source ./your_drive_files --dest ./drive_model")
        print("3. Load the model for real-time categorization:")
        print("   POST /load_categorization_model with your model directory")
        print("4. Test with incremental classification:")
        print("   python incremental_classifier.py --new_files ./new_files --model_dir ./drive_model")
        
        print("\n📚 Documentation:")
        print("• Read ENHANCED_CATEGORIZATION_GUIDE.md for detailed usage")
        print("• Check enhanced_drive_integration.py for integration examples")
        print("• Use demo_with_your_files.py for testing with your content")

def main():
    """Main startup function."""
    print("🚀 STARTING ENHANCED CATEGORIZATION SYSTEM")
    print("="*60)
    
    starter = EnhancedSystemStarter()
    
    # Step 1: Check dependencies
    if not starter.check_dependencies():
        print("\n❌ Please install missing dependencies first:")
        print("pip install -r requirements_enhanced_categorization.txt")
        return False
    
    # Step 2: Check file structure
    if not starter.check_file_structure():
        print("\n❌ Missing required files. Please ensure all files are present.")
        return False
    
    # Step 3: Start services (if not already running)
    print("\n🔄 Starting services...")
    
    # Check if gateway is already running
    gateway_running = starter.test_service_health("gateway", 4000)
    if not gateway_running:
        print("   ⚠️  Gateway not running. Please start it manually:")
        print("   cd gateway && node index.js")
    
    # Start enhanced embed service
    embed_process = None
    embed_running = starter.test_service_health("enhanced_embed", 8002)
    if not embed_running:
        embed_process = starter.start_service("enhanced_embed")
        if embed_process:
            time.sleep(5)  # Give it time to fully start
            starter.test_service_health("enhanced_embed", 8002)
    
    # Step 4: Run initial categorization test
    starter.run_initial_categorization()
    
    # Step 5: Test API endpoints
    if gateway_running or starter.test_service_health("gateway", 4000):
        starter.test_api_endpoints()
    
    # Step 6: Create summary
    starter.create_startup_summary()
    
    # Keep enhanced embed service running if we started it
    if embed_process and embed_process.poll() is None:
        print(f"\n🔄 Enhanced embed service is running (PID: {embed_process.pid})")
        print("Press Ctrl+C to stop...")
        try:
            embed_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping enhanced embed service...")
            embed_process.terminate()
            embed_process.wait()
            print("✅ Service stopped")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
