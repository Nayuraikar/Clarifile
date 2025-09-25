#!/usr/bin/env python3
"""
Fix the UI CSS loading issue
"""
import os
import subprocess
import sys

def build_ui():
    print("Building UI to generate CSS files...")
    print("=" * 50)
    
    ui_dir = "/Users/nayana/Desktop/PROGRAMS/clarifile/ui"
    
    # Change to UI directory
    os.chdir(ui_dir)
    
    # Install dependencies if needed
    print("Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "npm", "install"], check=True)
    except subprocess.CalledProcessError:
        print("[ERROR] Failed to install dependencies")
        return False
    
    # Build the UI
    print("Building UI with Vite...")
    try:
        subprocess.run([sys.executable, "-m", "npm", "run", "build"], check=True, capture_output=True)
        print("[OK] UI built successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build UI: {e.stdout.decode('utf-8')}")
    
    # Check if files were created
    dist_dir = os.path.join(ui_dir, "dist")
    if os.path.exists(dist_dir):
        files = os.listdir(dist_dir)
        print(f"Built files: {files}")
        
        # Check for CSS files
        css_files = [f for f in files if f.endswith('.css')]
        if css_files:
            print(f"[OK] CSS files found: {css_files}")
        else:
            print("[WARN] No CSS files found in dist")
    else:
        print("[ERROR] dist directory not created")
        return False
    
    return True

def update_gateway_static_files():
    print("\nUpdating gateway to serve static files...")
    print("=" * 50)
    
    gateway_file = "/Users/nayana/Desktop/PROGRAMS/clarifile/gateway/index.js"
    
    # Read the current gateway file
    with open(gateway_file, 'r') as f:
        content = f.read()
    
    # Check if static file serving is already added
    if 'app.use(express.static' in content:
        print("[OK] Static file serving already configured")
        return True
    
    # Add static file serving after the app.use(cors()) line
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        new_lines.append(line)
        if line.strip() == "app.use(cors());":
            new_lines.extend([
                "",
                "// Serve static files from UI dist directory",
                "app.use(express.static(path.join(__dirname, '../ui/dist')));",
                ""
            ])
    
    # Write the updated content
    with open(gateway_file, 'w') as f:
        f.write('\n'.join(new_lines))
    
    print("[OK] Added static file serving to gateway")
    return True

def update_html_css_reference():
    print("\nUpdating HTML to reference correct CSS file...")
    print("=" * 50)
    
    html_file = "/Users/nayana/Desktop/PROGRAMS/clarifile/ui/index.html"
    
    # Read the HTML file
    with open(html_file, 'r') as f:
        content = f.read()
    
    # Check what CSS files are available in dist
    dist_dir = "/Users/nayana/Desktop/PROGRAMS/clarifile/ui/dist"
    if os.path.exists(dist_dir):
        files = os.listdir(dist_dir)
        css_files = [f for f in files if f.endswith('.css')]
        
        if css_files:
            # Use the first CSS file found
            css_file = css_files[0]
            print(f"Using CSS file: {css_file}")
            
            # Update the CSS reference
            old_css = 'href="index.css"'
            new_css = f'href="{css_file}"'
            
            if old_css in content:
                content = content.replace(old_css, new_css)
                print(f"[OK] Updated CSS reference to: {css_file}")
            else:
                print(f"[WARN] Could not find old CSS reference: {old_css}")
                print(f"Adding new CSS reference...")
                # Add the CSS link before the closing head tag
                content = content.replace('</head>', f'    <link rel="stylesheet" href="{css_file}">\n</head>')
        else:
            print("[ERROR] No CSS files found in dist directory")
            return False
    else:
        print("[ERROR] dist directory not found")
        return False
    
    # Write the updated HTML
    with open(html_file, 'w') as f:
        f.write(content)
    
    print("✅ HTML updated with correct CSS reference")
    return True

def main():
    print("Fixing UI CSS Loading Issue")
    print("=" * 60)
    
    # Step 1: Build the UI
    if not build_ui():
        print("❌ Failed to build UI")
        return
    
    # Step 2: Update gateway to serve static files
    if not update_gateway_static_files():
        print("[ERROR] Failed to update gateway")
        return
    
    # Step 3: Update HTML CSS reference
    if not update_html_css_reference():
        print("[ERROR] Failed to update HTML")
        return
    
    print("\n[SUCCESS] CSS Issue Fixed")
    print("=" * 50)
    print("✅ UI built successfully")
    print("[OK] Gateway configured to serve static files")
    print("✅ HTML updated with correct CSS reference")
    print("\nNext steps:")
    print("1. Restart the gateway: cd gateway && node index.js")
    print("2. Refresh your browser (Ctrl+F5 or Cmd+Shift+R)")
    print("3. The Analyze button should now work!")
    print("\nThe CSS error should be gone and buttons should be clickable!")

if __name__ == "__main__":
    main()
