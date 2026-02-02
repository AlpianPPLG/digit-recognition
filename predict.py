#!/usr/bin/env python
"""
Digit Recognition - Prediction Entry Point

This script makes predictions on images using a trained model.

Usage:
    python predict.py <image_path> [options]
    python predict.py --batch <folder_path> [options]

Options:
    --model-path PATH   Path to trained model
    --verbose           Show detailed output
    --batch             Process multiple images from a folder
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Make predictions with trained digit recognition model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input
    parser.add_argument(
        'input',
        type=str,
        nargs='?',
        help='Path to input image or folder (with --batch)'
    )
    
    # Model
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/default.npz',
        help='Path to trained model'
    )
    
    # Mode
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process multiple images from a folder'
    )
    
    # Output
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for predictions (CSV for batch mode)'
    )
    parser.add_argument(
        '--show-probabilities',
        action='store_true',
        help='Show probability distribution for all classes'
    )
    
    # Options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def predict_single(image_path: str, model_path: str, verbose: bool = False, show_probs: bool = False):
    """Make prediction on a single image"""
    print(f"\nProcessing: {image_path}")
    
    # TODO: Implement when modules are ready
    # from core import NeuralNetwork
    # from preprocessing import PreprocessingPipeline
    # 
    # model = NeuralNetwork.load(model_path)
    # pipeline = PreprocessingPipeline()
    # 
    # image = pipeline.process(image_path)
    # prediction, probabilities = model.predict(image)
    
    print("Prediction functionality not yet implemented.")
    print("Please implement core and preprocessing modules first.")


def predict_batch(folder_path: str, model_path: str, output_path: str = None, verbose: bool = False):
    """Make predictions on multiple images"""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder not found: {folder_path}")
        return
    
    # Find all images
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    images = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]
    
    print(f"\nFound {len(images)} images in {folder_path}")
    
    # TODO: Implement batch prediction
    print("Batch prediction functionality not yet implemented.")


def main():
    """Main prediction function"""
    args = parse_args()
    
    print("=" * 60)
    print("🔢 Digit Recognition - Prediction")
    print("=" * 60)
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"\n⚠ Model not found: {args.model_path}")
        print("\nPlease train a model first using:")
        print("  python train.py")
        print()
        print("Or specify a different model path:")
        print("  python predict.py <image> --model-path <path>")
        print("=" * 60)
        return
    
    # Check input
    if not args.input:
        print("\nUsage:")
        print("  python predict.py <image_path>")
        print("  python predict.py --batch <folder_path>")
        print()
        print("Examples:")
        print("  python predict.py data/custom/digit.png")
        print("  python predict.py --batch data/custom/")
        print("=" * 60)
        return
    
    # Make predictions
    if args.batch:
        predict_batch(
            args.input, 
            args.model_path, 
            args.output, 
            args.verbose
        )
    else:
        predict_single(
            args.input, 
            args.model_path, 
            args.verbose,
            args.show_probabilities
        )
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
