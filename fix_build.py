#!/usr/bin/env python3
"""
Script to fix build errors in App.tsx
"""

import os

def fix_app_tsx():
    file_path = '/Users/nayana/Desktop/PROGRAMS/clarifile/ui/src/App.tsx'
    
    # Read the current file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix 1: Add the missing import
    if 'import DuplicateResolution from' not in content:
        # Find the line with React import and add after it
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('import React') and 'DuplicateResolution' not in line:
                lines.insert(i + 1, "import DuplicateResolution from './DuplicateResolution'")
                break
        content = '\n'.join(lines)
    
    # Fix 2: Replace </Section> with </div> on line 788 (around categories section)
    content = content.replace(
        "              </Section>\n          )}",
        "              </div>\n          )}"
    )
    
    # Fix 3: Replace </Section> with </div> on line 900 (around AI section)
    content = content.replace(
        "            </Section>\n          )}",
        "            </div>\n          )}"
    )
    
    # Fix 4: Remove the extra </div> tag
    content = content.replace(
        "          </div>\n      </div>",
        "      </div>"
    )
    
    # Write the fixed content back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Fixed App.tsx build errors")
    print("1. Added DuplicateResolution import")
    print("2. Fixed </Section> tags to </div>")
    print("3. Removed extra closing </div> tag")

if __name__ == "__main__":
    fix_app_tsx()
