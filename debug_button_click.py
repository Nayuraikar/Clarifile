#!/usr/bin/env python3
"""
Debug script to check for common issues with the Analyze button
"""
import os

def check_app_tsx():
    print("🔍 Debugging Analyze Button Issues")
    print("=" * 50)
    
    app_file = "/Users/nayana/Desktop/PROGRAMS/clarifile/ui/src/App.tsx"
    
    if not os.path.exists(app_file):
        print("❌ App.tsx file not found!")
        return False
    
    with open(app_file, 'r') as f:
        content = f.read()
    
    # Check for common issues
    checks = [
        ("console.log.*Analyze button clicked", "Debug console.log added"),
        ("setLoading('analyzeFile', true)", "Loading state set to true"),
        ("setLoading('analyzeFile', false)", "Loading state set to false"),
        ("disabled={isLoading('analyzeFile')}", "Button disabled during loading"),
        ("loading={isLoading('analyzeFile')}", "Loading spinner enabled"),
        ("const [notification, setNotification]", "Notification state declared"),
        ("Toast Notification", "Toast notification component"),
        ("const [driveAnalyzedId, setDriveAnalyzedId]", "Drive analyzed state declared"),
    ]
    
    print("📋 Checking implementation:")
    all_good = True
    
    for check_text, description in checks:
        if check_text in content:
            print(f"   ✅ {description}")
        else:
            print(f"   ❌ {description} - MISSING")
            all_good = False
    
    # Check for issues
    issues = [
        ("analyzingFile", "Old analyzingFile state (should be removed)"),
        ("syntax error", "Potential syntax errors"),
        ("undefined", "Potential undefined variables"),
    ]
    
    print("\n🐛 Checking for potential issues:")
    for issue, description in issues:
        if issue in content.lower():
            print(f"   ⚠️  {description} - FOUND")
        else:
            print(f"   ✅ {description} - NOT FOUND")
    
    return all_good

def check_browser_console():
    print("\n🌐 Browser Console Debug Instructions")
    print("=" * 50)
    print("To debug the issue, follow these steps:")
    print("1. Open your browser")
    print("2. Go to your Clarifile app")
    print("3. Press F12 to open Developer Tools")
    print("4. Click on 'Console' tab")
    print("5. Click the 'Analyze' button on a file")
    print("6. Look for console.log messages:")
    print("   - 'Analyze button clicked for file: ...'")
    print("   - 'Set loading to true'")
    print("   - 'Set notification to Analyzing file...'")
    print("   - 'Timeout completed, setting file as analyzed'")
    print("   - 'Set loading to false, file analyzed: ...'")
    print("\n🔍 If you don't see these messages, the button isn't being clicked")
    print("🔍 If you see errors, there's a JavaScript error")
    print("🔍 If you see messages but no UI changes, the state updates aren't working")

def main():
    print("🚀 Debug: Analyze Button Not Working")
    print("=" * 60)
    
    implementation_ok = check_app_tsx()
    check_browser_console()
    
    print("\n📋 Debug Summary:")
    print("=" * 50)
    
    if implementation_ok:
        print("✅ Implementation looks correct")
        print("🔧 The issue might be:")
        print("   1. JavaScript error in the browser")
        print("   2. Button not being clickable (CSS issue)")
        print("   3. State updates not triggering re-renders")
        print("   4. Browser caching old version")
        print("\n🎯 Next steps:")
        print("   1. Check browser console for errors")
        print("   2. Try hard refresh (Ctrl+F5 or Cmd+Shift+R)")
        print("   3. Check if button is clickable (hover effect)")
    else:
        print("❌ Implementation has issues")
        print("🔧 Fix the missing components above")

if __name__ == "__main__":
    main()
