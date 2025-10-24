"""
Fast environment check without running TensorFlow
"""

import sys

print("=" * 60)
print("Environment Check")
print("=" * 60)

# Test 1: Python
print("\n1. Python:", sys.version.split()[0])

# Test 2: Check if packages are installed
print("\n2. Checking packages...")
packages = {
    'tensorflow': None,
    'numpy': None,
    'scipy': None,
    'skimage': 'scikit-image',
    'matplotlib': None,
    'tqdm': None,
}

all_ok = True
for pkg, import_name in packages.items():
    try:
        if import_name:
            __import__(import_name.replace('-', '_'))
        else:
            __import__(pkg)
        print(f"   ✓ {pkg}")
    except ImportError:
        print(f"   ✗ {pkg} - NOT FOUND")
        all_ok = False

# Test 3: Check project structure
print("\n3. Checking project structure...")
import os
dirs = ['core', 'models', 'training', 'inference', 'configs', 'utils', 'examples']
for d in dirs:
    if os.path.isdir(d):
        print(f"   ✓ {d}/")
    else:
        print(f"   ✗ {d}/ - NOT FOUND")
        all_ok = False

print("\n" + "=" * 60)
if all_ok:
    print("✓ Environment check passed!")
    print("\nNext steps:")
    print("  1. Activate environment: conda activate adaptive_stardist")
    print("  2. Run examples: cd examples && python train_example.py")
else:
    print("✗ Some checks failed")
print("=" * 60)

