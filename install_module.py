#!/usr/bin/env python
"""
Installation and Module Loading Guide for SFR Analyzer

This script helps you install and load the SFR Analyzer module.
"""

import sys
import os
import subprocess

def install_dependencies():
    """Install all required dependencies"""
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Dependencies installed successfully")

def install_module():
    """Install the module in development mode"""
    print("\nInstalling SFR Analyzer module...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
    print("✅ Module installed successfully")

def load_module():
    """Test loading the module"""
    print("\nTesting module import...")
    try:
        import SFR_app_v2
        print("✅ SFR_app_v2 module loaded successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to load module: {e}")
        return False

def main():
    print("=" * 60)
    print("SFR Analyzer Module Installation and Loading Guide")
    print("=" * 60)

    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    # Step 1: Install dependencies
    try:
        install_dependencies()
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

    # Step 2: Install module
    try:
        install_module()
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install module: {e}")
        return False

    # Step 3: Test loading
    if not load_module():
        return False

    print("\n" + "=" * 60)
    print("✅ Module installation and loading complete!")
    print("=" * 60)
    print("\nUsage:")
    print("  python SFR_app_v2.py          # Run the GUI application")
    print("  python test_edge_detection.py # Run tests")
    print("\nOr import in Python:")
    print("  from SFR_app_v2 import SFRCalculator")
    print("\n" + "=" * 60)

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

