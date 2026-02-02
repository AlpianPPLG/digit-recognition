#!/usr/bin/env python
"""
Train Model Script

This script trains the neural network on MNIST dataset.

Usage:
    python scripts/train_model.py [--epochs N] [--batch-size N] [--lr RATE]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def parse_args():
    parser = argparse.ArgumentParser(description='Train digit recognition model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model-path', type=str, default='models/default.npz', help='Path to save model')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 50)
    print("Digit Recognition - Model Training")
    print("=" * 50)
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Model Path: {args.model_path}")
    print("=" * 50)
    
    # TODO: Import and run training
    # from training import Trainer
    # trainer = Trainer(...)
    # trainer.train()
    
    print("\nTraining module not yet implemented.")
    print("Please implement the training module first.")


if __name__ == "__main__":
    main()
