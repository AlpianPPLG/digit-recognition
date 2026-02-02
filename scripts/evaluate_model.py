#!/usr/bin/env python
"""
Evaluate Model Script

This script evaluates a trained model on the MNIST test set.

Usage:
    python scripts/evaluate_model.py [--model-path PATH]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate digit recognition model')
    parser.add_argument('--model-path', type=str, default='models/default.npz', help='Path to trained model')
    parser.add_argument('--verbose', action='store_true', help='Show detailed metrics')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 50)
    print("Digit Recognition - Model Evaluation")
    print("=" * 50)
    print(f"Model Path: {args.model_path}")
    print("=" * 50)
    
    # TODO: Import and run evaluation
    # from core import NeuralNetwork
    # from training import MNISTLoader
    # model = NeuralNetwork.load(args.model_path)
    # ...
    
    print("\nEvaluation module not yet implemented.")
    print("Please implement the core modules first.")


if __name__ == "__main__":
    main()
