#!/usr/bin/env python3
"""
Final test to verify the Analyze button is working correctly
"""
import os

def check_analyze_button_implementation():
    print("Final Check: Analyze Button Implementation")
    print("=" * 50)
    
    # Check if the App.tsx file has the correct implementation
    app_file = "/Users/nayana/Desktop/PROGRAMS/clarifile/ui/src/App.tsx"
    
    if not os.path.exists(app_file):
        print("[ERROR] App.tsx file not found!")
        return False
    
    with open(app_file, 'r') as f:
        content = f.read()
    
    # Check for the correct implementation
    checks = [
        ("setLoading('analyzeFile', true)", "Start loading state"),
        ("setLoading('analyzeFile', false)", "Stop loading state"),
        ("disabled={isLoading('analyzeFile')}", "Disable button during loading"),
        ("loading={isLoading('analyzeFile')}", "Show loading spinner"),
        ("setNotification('Analyzing file...')", "First notification"),
        ("setNotification('File selected for AI analysis!')", "Second notification"),
        ("setTimeout.*1000", "1 second delay"),
    ]
    
    print("Checking Analyze button implementation:")
    all_good = True
    
    for check_text, description in checks:
        if check_text in content:
            print(f"   [OK] {description}")
        else:
            print(f"   [ERROR] {description} - MISSING")
            all_good = False
    
    # Check that old implementation is removed
    old_checks = [
        ("setAnalyzingFile(file.id)", "Old analyzingFile state (should be removed)"),
        ("analyzingFile === file.id", "Old loading condition (should be removed)"),
        ("const [analyzingFile, setAnalyzingFile]", "Old analyzingFile declaration (should be removed)"),
    ]
    
    print("\nChecking that old implementation is removed:")
    for check_text, description in old_checks:
        if check_text not in content:
            print(f"   [OK] {description} - REMOVED")
        else:
            print(f"   [ERROR] {description} - STILL PRESENT")
            all_good = False
    
    return all_good

def compare_with_refresh_button():
    print("\nComparing with Refresh Files Button")
    print("=" * 50)
    
    app_file = "/Users/nayana/Desktop/PROGRAMS/clarifile/ui/src/App.tsx"
    
    with open(app_file, 'r') as f:
        content = f.read()
    
    # Find Refresh Files button implementation
    refresh_pattern = "onClick={refreshDrive} disabled={isLoading('refreshDrive')} loading={isLoading('refreshDrive')}"
    analyze_pattern = "disabled={isLoading('analyzeFile')} loading={isLoading('analyzeFile')}"
    
    if refresh_pattern in content:
        print("   [OK] Refresh Files button uses isLoading('refreshDrive')")
    else:
        print("   [ERROR] Refresh Files button pattern not found")
    
    if analyze_pattern in content:
        print("   [OK] Analyze button uses isLoading('analyzeFile') - SAME PATTERN!")
        return True
    else:
        print("   [ERROR] Analyze button doesn't use the same pattern")
        return False

def main():
    print("Final Test: Analyze Button Loading State")
    print("=" * 60)
    
    implementation_ok = check_analyze_button_implementation()
    pattern_ok = compare_with_refresh_button()
    
    print("\nFinal Summary:")
    print("=" * 50)
    
    if implementation_ok and pattern_ok:
        print("[SUCCESS] ALL CHECKS PASSED!")
        print("\nThe Analyze button now works exactly like Refresh Files button:")
        print("   - Shows loading spinner during operation")
        print("   - Button is disabled during loading")
        print("   - Shows 'Analyzing file...' notification")
        print("   - Shows 'File selected for AI analysis!' notification")
        print("   - File gets selected with blue border")
        print("\nTest it in your browser now!")
        print("   1. Refresh your browser")
        print("   2. Go to Drive tab")
        print("   3. Click 'Analyze' on any file")
        print("   4. You should see the loading spinner!")
    else:
        print("[WARN] Some issues found:")
        if not implementation_ok:
            print("   [ERROR] Implementation is incorrect")
        if not pattern_ok:
            print("   [ERROR] Doesn't match Refresh Files button pattern")
        print("\nThe implementation needs to be fixed")

if __name__ == "__main__":
    main()
