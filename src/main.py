#!/usr/bin/env python3
"""
AI Digit Recognition

Main entry point for the application.
Run this file to launch the GUI application.

Usage:
    python -m src.main          # Launch GUI
    python -m src.main --train  # Train model first, then launch GUI
    python -m src.main --help   # Show help

Author: AI Digit Team
License: MIT
"""

import sys
import argparse
from pathlib import Path

# Ensure src is in path
src_path = str(Path(__file__).parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AI Digit Recognition - Handwritten Digit Recognition using Neural Networks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.main          Launch the GUI application
    python -m src.main --train  Train model before launching GUI
    python -m src.main --cli    Run in command-line mode
        """
    )
    
    parser.add_argument(
        '--train', 
        action='store_true',
        help='Train the model before launching GUI'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs (default: 30)'
    )
    parser.add_argument(
        '--cli',
        action='store_true',
        help='Run in command-line mode (no GUI)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run tests instead of launching app'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='AI Digit Recognition v1.0.0'
    )
    
    args = parser.parse_args()
    
    # Run tests
    if args.test:
        import pytest
        sys.exit(pytest.main([str(Path(__file__).parent.parent / 'tests'), '-v']))
    
    # Train model
    if args.train:
        print("Training model...")
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from train_mnist import train_mnist
        train_mnist(epochs=args.epochs)
        print("\nTraining complete!")
        
        if args.cli:
            return
    
    # CLI mode
    if args.cli:
        from core.serialization import load_model
        from preprocessing.image_preprocessor import ImagePreprocessor
        import numpy as np
        
        # Load model
        models_dir = Path(__file__).parent.parent / 'models'
        model_path = models_dir / 'digit_recognition_model.npz'
        
        if not model_path.exists():
            print("Error: No trained model found!")
            print(f"Expected: {model_path}")
            print("Run with --train flag to train a model first.")
            sys.exit(1)
        
        network, _, metadata = load_model(str(model_path))
        print(f"Model loaded: {model_path.name}")
        print(f"Metadata: {metadata}")
        
        print("\nCLI mode ready. Type 'quit' to exit.")
        print("Enter image path to predict digit:")
        
        while True:
            try:
                user_input = input("> ").strip()
                if user_input.lower() in ('quit', 'exit', 'q'):
                    break
                
                if not user_input:
                    continue
                
                # Load and preprocess image
                from PIL import Image
                img = Image.open(user_input).convert('L')
                img = img.resize((28, 28))
                arr = np.array(img).astype(np.float32) / 255.0
                
                # Invert if needed (white on black)
                if arr.mean() > 0.5:
                    arr = 1.0 - arr
                
                # Predict
                output = network.forward(arr.flatten().reshape(1, -1))
                predicted = int(np.argmax(output[0]))
                confidence = float(output[0][predicted] * 100)
                
                print(f"Predicted: {predicted} (confidence: {confidence:.1f}%)")
                
            except FileNotFoundError:
                print(f"File not found: {user_input}")
            except Exception as e:
                print(f"Error: {e}")
        
        return
    
    # Launch GUI
    print("Launching AI Digit Recognition...")
    from gui.app import DigitRecognitionApp
    
    app = DigitRecognitionApp()
    
    # Auto-load model if available
    models_dir = Path(__file__).parent.parent / 'models'
    default_model = models_dir / 'digit_recognition_model.npz'
    
    if default_model.exists():
        try:
            from core.serialization import load_model
            app.network, _, _ = load_model(str(default_model))
            app.set_model_status(True, "digit_recognition_model")
            print(f"Model loaded: {default_model.name}")
        except Exception as e:
            print(f"Warning: Could not load default model: {e}")
    else:
        print("No trained model found. Use 'Load Model' button or run with --train flag.")
    
    app.mainloop()


if __name__ == "__main__":
    main()
