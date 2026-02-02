#!/usr/bin/env python
"""
Digit Recognition - Main Entry Point

This is the main entry point for the GUI application.
Launch the interactive digit recognition interface.

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))


def check_dependencies():
    """Check if all required dependencies are installed"""
    missing = []
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import PIL
    except ImportError:
        missing.append("Pillow")
    
    try:
        import customtkinter
    except ImportError:
        missing.append("customtkinter")
    
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    
    if missing:
        print("=" * 50)
        print("ERROR: Missing required dependencies!")
        print("=" * 50)
        print("\nPlease install the following packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nRun: pip install -r requirements.txt")
        print("=" * 50)
        return False
    
    return True


def main():
    """Main entry point for the application"""
    print("=" * 50)
    print("🔢 Digit Recognition - AI from Scratch")
    print("=" * 50)
    print()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("✓ All dependencies are installed")
    print()
    
    # TODO: Launch GUI when implemented
    # from gui import MainWindow
    # app = MainWindow()
    # app.run()
    
    print("GUI module is not yet implemented.")
    print("Please implement the GUI module first.")
    print()
    print("For now, you can use:")
    print("  - python train.py   : Train a model")
    print("  - python predict.py : Make predictions")
    print("=" * 50)


if __name__ == "__main__":
    main()
