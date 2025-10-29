#!/usr/bin/env python3
"""
Quick readiness check before starting the workflow
"""

import os
import sys
from pathlib import Path

def check_mark(condition):
    return "✓" if condition else "✗"

def main():
    print("=" * 60)
    print("Shape-aware StarDist - Readiness Check")
    print("=" * 60)
    print()
    
    # Check 1: Kaggle API
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
    kaggle_ok = kaggle_config.exists()
    print(f"{check_mark(kaggle_ok)} Kaggle API configured: {kaggle_config}")
    
    # Check 2: Scripts executable
    setup_script = Path("setup_dsb2018.sh")
    upload_script = Path("upload_to_drive.sh")
    scripts_ok = setup_script.exists() and upload_script.exists()
    print(f"{check_mark(scripts_ok)} Setup scripts available")
    
    # Check 3: Git status
    git_dir = Path("../.git")
    git_ok = git_dir.exists()
    print(f"{check_mark(git_ok)} Git repository initialized")
    
    # Check 4: Python environment
    try:
        import torch
        import numpy
        import skimage
        python_ok = True
        print(f"✓ Python environment ready")
        print(f"  - PyTorch: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        python_ok = False
        print(f"✗ Python environment incomplete: {e}")
    
    # Check 5: Project structure
    required_dirs = [
        "models",
        "training",
        "data",
        "configs",
        "colab"
    ]
    structure_ok = all((Path(d).exists() for d in required_dirs))
    print(f"{check_mark(structure_ok)} Project structure complete")
    
    print()
    print("=" * 60)
    
    all_ok = kaggle_ok and scripts_ok and git_ok and python_ok and structure_ok
    
    if all_ok:
        print("🎉 All checks passed! You're ready to start.")
        print()
        print("Next steps:")
        print("1. Push code to GitHub (see NEXT_STEPS.md)")
        print("2. Run: ./setup_dsb2018.sh")
        print("3. Run: ./upload_to_drive.sh")
        print("4. Open Colab notebook and start training")
        return 0
    else:
        print("⚠️  Some checks failed. Please review above.")
        print()
        if not kaggle_ok:
            print("→ Configure Kaggle API:")
            print("  1. Download kaggle.json from https://www.kaggle.com/settings")
            print("  2. mv ~/Downloads/kaggle.json ~/.kaggle/")
            print("  3. chmod 600 ~/.kaggle/kaggle.json")
        if not python_ok:
            print("→ Install dependencies:")
            print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())

