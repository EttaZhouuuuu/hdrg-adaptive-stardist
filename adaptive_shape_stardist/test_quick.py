"""
Quick test to verify the environment and basic imports
"""

import sys

print("=" * 60)
print("Quick Environment Test")
print("=" * 60)

# Test 1: Check Python version
print("\n1. Python version:")
print(f"   {sys.version}")

# Test 2: Import TensorFlow
print("\n2. Testing TensorFlow import...")
try:
    import tensorflow as tf
    print(f"   ✓ TensorFlow {tf.__version__} imported successfully")
except Exception as e:
    print(f"   ✗ TensorFlow import failed: {e}")
    sys.exit(1)

# Test 3: Import other dependencies
print("\n3. Testing other dependencies...")
try:
    import numpy as np
    print(f"   ✓ NumPy {np.__version__}")
    
    import scipy
    print(f"   ✓ SciPy {scipy.__version__}")
    
    import skimage
    print(f"   ✓ scikit-image {skimage.__version__}")
    
    import matplotlib
    print(f"   ✓ Matplotlib {matplotlib.__version__}")
    
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 4: Import project modules
print("\n4. Testing project imports...")
try:
    from adaptive_shape_stardist import AdaptiveShapeStarDist
    print("   ✓ AdaptiveShapeStarDist imported")
    
    from adaptive_shape_stardist.configs import get_default_config
    print("   ✓ Config module imported")
    
except Exception as e:
    print(f"   ✗ Project import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Create simple config
print("\n5. Testing configuration...")
try:
    config = get_default_config()
    print(f"   ✓ Config created")
    print(f"   - Backbone: {config.backbone}")
    print(f"   - Sampling points: {config.min_sampling_points}-{config.max_sampling_points}")
    
except Exception as e:
    print(f"   ✗ Config error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All quick tests passed!")
print("=" * 60)
print("\nEnvironment is ready to use!")

