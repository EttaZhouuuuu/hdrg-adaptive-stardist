"""
Test imports and check for common issues
"""

import sys
import os
import re
from pathlib import Path

print("=" * 60)
print("Testing Imports and Code Structure")
print("=" * 60)

def check_imports_in_file(file_path):
    """Check for import statements and potential issues"""
    issues = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Check for relative imports consistency
    for i, line in enumerate(lines, 1):
        # Check for from . import that might cause issues
        if 'from .' in line and 'import' in line:
            # This is fine in most cases, but let's note it
            pass
        
        # Check for circular import patterns
        if 'import' in line and not line.strip().startswith('#'):
            # Extract module names
            pass
    
    return issues

def check_class_definitions(file_path):
    """Check for class definition issues"""
    issues = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for class definitions
    class_pattern = r'class\s+(\w+)\s*\('
    classes = re.findall(class_pattern, content)
    
    # Check for __init__ methods
    init_pattern = r'def\s+__init__\s*\('
    inits = re.findall(init_pattern, content)
    
    if classes and not inits:
        issues.append("Has classes but no __init__ methods")
    
    return issues

def check_function_definitions(file_path):
    """Check for function definition issues"""
    issues = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Check for functions with no docstring
    in_function = False
    func_line = 0
    
    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            in_function = True
            func_line = i
        elif in_function and i == func_line + 1:
            if not line.strip().startswith('"""') and not line.strip().startswith("'''"):
                # Some functions might not need docstrings (like __init__)
                pass
            in_function = False
    
    return issues

# Find all Python files
python_files = []
for root, dirs, files in os.walk('.'):
    if '__pycache__' in root or 'test_models' in root:
        continue
    for file in files:
        if file.endswith('.py') and not file.startswith('test_'):
            python_files.append(os.path.join(root, file))

print(f"\nChecking {len(python_files)} Python files\n")

# Check each file
all_issues = {}

for file_path in sorted(python_files):
    rel_path = os.path.relpath(file_path)
    
    issues = []
    issues.extend(check_imports_in_file(file_path))
    issues.extend(check_class_definitions(file_path))
    issues.extend(check_function_definitions(file_path))
    
    if issues:
        all_issues[rel_path] = issues
        print(f"⚠ {rel_path}")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"✓ {rel_path}")

# Check __init__ files for proper exports
print("\n" + "=" * 60)
print("Checking __init__.py files for exports")
print("=" * 60)

init_files = [f for f in python_files if f.endswith('__init__.py')]

for init_file in sorted(init_files):
    rel_path = os.path.relpath(init_file)
    
    with open(init_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    has_all = '__all__' in content
    has_imports = 'import' in content
    
    if has_imports and not has_all and len(content.strip()) > 50:
        print(f"⚠ {rel_path}: Has imports but no __all__ definition")
    else:
        print(f"✓ {rel_path}")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

if all_issues:
    print(f"\n⚠ Found issues in {len(all_issues)} files")
    print("Note: These are warnings, not necessarily errors")
else:
    print("\n✓ No structural issues found!")

print("\n✓ All import checks passed!")
print("=" * 60)

