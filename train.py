#!/usr/bin/env python
"""
Digit Recognition - Training Entry Point

This script trains the neural network on the MNIST dataset.

Usage:
    python train.py [options]

Options:
    --epochs N          Number of training epochs (default: 20)
    --batch-size N      Batch size (default: 32)
    --lr RATE           Learning rate (default: 0.001)
    --model-path PATH   Path to save the trained model
    --no-gui            Disable training progress GUI
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
        description='Train digit recognition neural network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=20,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=32,
        help='Mini-batch size'
    )
    parser.add_argument(
        '--lr', '--learning-rate',
        type=float, 
        default=0.001,
        dest='learning_rate',
        help='Learning rate'
    )
    
    # Model configuration
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/default.npz',
        help='Path to save the trained model'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='models/checkpoints',
        help='Directory to save checkpoints'
    )
    
    # Training options
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['sgd', 'adam', 'rmsprop'],
        default='adam',
        help='Optimizer to use'
    )
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Disable validation during training'
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.1,
        help='Fraction of training data to use for validation'
    )
    
    # UI options
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Disable GUI progress display'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    print("=" * 60)
    print("🧠 Digit Recognition - Neural Network Training")
    print("=" * 60)
    print()
    print("Training Configuration:")
    print(f"  • Epochs:        {args.epochs}")
    print(f"  • Batch Size:    {args.batch_size}")
    print(f"  • Learning Rate: {args.learning_rate}")
    print(f"  • Optimizer:     {args.optimizer}")
    print(f"  • Model Path:    {args.model_path}")
    print(f"  • Validation:    {'Disabled' if args.no_validation else f'{args.validation_split*100:.0f}%'}")
    print()
    print("=" * 60)
    
    # TODO: Implement training when modules are ready
    # Steps:
    # 1. Load MNIST dataset
    # 2. Create neural network
    # 3. Setup optimizer
    # 4. Training loop
    # 5. Save model
    
    try:
        # Check if core modules are implemented
        from core import NeuralNetwork
        print("✓ Core modules loaded successfully")
        
        # TODO: Continue with training implementation
        print("\nTraining implementation coming soon...")
        
    except ImportError as e:
        print(f"\n⚠ Core modules not yet fully implemented: {e}")
        print("\nPlease implement the following modules first:")
        print("  - src/core/network.py")
        print("  - src/core/layers.py")
        print("  - src/core/activations.py")
        print("  - src/core/losses.py")
        print("  - src/core/optimizers.py")
        print("  - src/training/mnist_loader.py")
        print("  - src/training/trainer.py")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
