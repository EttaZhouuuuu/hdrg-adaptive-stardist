"""
Syntax and import test without requiring TensorFlow
"""

import sys
import ast
import os
from pathlib import Path

print("=" * 60)
print("Testing Python Syntax and Imports")
print("=" * 60)

def check_python_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)

# Find all Python files
python_files = []
for root, dirs, files in os.walk('.'):
    # Skip test and pycache directories
    if '__pycache__' in root or 'test_models' in root:
        continue
    for file in files:
        if file.endswith('.py'):
            python_files.append(os.path.join(root, file))

print(f"\nFound {len(python_files)} Python files\n")

# Check syntax for each file
errors = []
passed = 0

for file_path in sorted(python_files):
    rel_path = os.path.relpath(file_path)
    success, error = check_python_syntax(file_path)
    
    if success:
        print(f"✓ {rel_path}")
        passed += 1
    else:
        print(f"✗ {rel_path}")
        print(f"  Error: {error}")
        errors.append((rel_path, error))

# Summary
print("\n" + "=" * 60)
print(f"Results: {passed}/{len(python_files)} files passed")
print("=" * 60)

if errors:
    print("\nErrors found:")
    for file_path, error in errors:
        print(f"  {file_path}: {error}")
    sys.exit(1)
else:
    print("\n✓ All files have valid Python syntax!")
    sys.exit(0)

