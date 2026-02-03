"""
MNIST Training Script

Train the digit recognition neural network on the MNIST dataset.
Target: ≥97% accuracy
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent.parent / 'src')
sys.path.insert(0, src_path)

from core.network import NetworkBuilder
from core.optimizers import Adam
from core.losses import CrossEntropyLoss
from core.serialization import save_model
from preprocessing.mnist_loader import MNISTLoader, split_data
from training.trainer import Trainer, EarlyStopping, LearningRateScheduler


def create_network():
    """Create the digit recognition network architecture."""
    print("Creating network architecture...")
    print("  Architecture: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Softmax)")
    
    network = (NetworkBuilder()
               .input(784)
               .dense(128, activation='relu')
               .dense(64, activation='relu')
               .dense(10, activation='softmax')
               .build())
    
    # Count parameters
    total_params = 0
    for layer in network.layers:
        if hasattr(layer, 'weights') and layer.weights is not None:
            total_params += layer.weights.size
        if hasattr(layer, 'bias') and layer.bias is not None:
            total_params += layer.bias.size
    
    print(f"  Total parameters: {total_params:,}")
    return network


def train_mnist(epochs=50, batch_size=64, learning_rate=0.001, 
                data_dir=None, save_path=None, verbose=True):
    """
    Train model on MNIST dataset.
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        data_dir: Directory for MNIST data
        save_path: Path to save trained model
        verbose: Print training progress
        
    Returns:
        Tuple of (trained_network, training_history, test_accuracy)
    """
    # Setup paths
    project_root = Path(__file__).parent.parent
    if data_dir is None:
        data_dir = project_root / "data" / "mnist"
    if save_path is None:
        save_path = project_root / "models" / "digit_recognition_model.npz"
    
    # Ensure directories exist
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MNIST Digit Recognition Training")
    print("=" * 60)
    
    # Load MNIST data
    print("\n1. Loading MNIST dataset...")
    loader = MNISTLoader(data_dir=str(data_dir))
    
    try:
        (X_train, y_train), (X_test, y_test) = loader.load(
            normalize=True, flatten=True, one_hot=False
        )
    except Exception as e:
        print(f"   Error loading MNIST: {e}")
        print(f"   Downloading MNIST dataset...")
        loader.download()
        (X_train, y_train), (X_test, y_test) = loader.load(
            normalize=True, flatten=True, one_hot=False
        )
    
    print(f"   Training samples: {X_train.shape[0]:,}")
    print(f"   Test samples: {X_test.shape[0]:,}")
    
    # Data is already normalized by loader
    print("\n2. Preprocessing data...")
    
    # Convert labels to one-hot
    def to_one_hot(labels, num_classes=10):
        one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
        one_hot[np.arange(len(labels)), labels.astype(int)] = 1
        return one_hot
    
    y_train_oh = to_one_hot(y_train)
    y_test_oh = to_one_hot(y_test)
    
    # Split training into train/validation
    (X_train, y_train_oh), (X_val, y_val_oh) = split_data(X_train, y_train_oh, val_split=0.1)
    
    print(f"   Training: {X_train.shape[0]:,} samples")
    print(f"   Validation: {X_val.shape[0]:,} samples")
    print(f"   Test: {X_test.shape[0]:,} samples")
    
    # Create network
    print("\n3. Building neural network...")
    network = create_network()
    
    # Setup training
    print("\n4. Configuring training...")
    print(f"   Optimizer: Adam (lr={learning_rate})")
    print(f"   Loss: Cross-Entropy")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {epochs}")
    
    optimizer = Adam(learning_rate=learning_rate)
    loss_fn = CrossEntropyLoss()
    
    trainer = Trainer(
        network=network,
        optimizer=optimizer,
        loss_fn=loss_fn
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        patience=10,
        min_delta=0.001,
        mode='min',
        restore_best=True
    )
    
    lr_scheduler = LearningRateScheduler(
        initial_lr=learning_rate,
        schedule_type='plateau',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    
    # Train
    print("\n5. Training...")
    print("-" * 60)
    
    history = trainer.fit(
        X_train, y_train_oh,
        X_val, y_val_oh,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping=early_stopping,
        lr_scheduler=lr_scheduler,
        verbose=2 if verbose else 0
    )
    
    print("-" * 60)
    
    # Evaluate on test set
    print("\n6. Evaluating on test set...")
    test_loss, test_acc = trainer.evaluate(X_test, y_test_oh)
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc*100:.2f}%")
    
    # Save model
    print(f"\n7. Saving model to {save_path}...")
    
    # Get training summary
    summary = history.summary()
    
    save_model(
        network,
        str(save_path),
        metadata={
            'epochs_trained': summary['epochs'],
            'final_train_loss': summary.get('best_loss', 0),
            'final_val_loss': summary.get('best_val_loss', 0),
            'final_train_acc': summary.get('best_accuracy', 0),
            'final_val_acc': summary.get('best_val_accuracy', 0),
            'test_accuracy': test_acc,
            'architecture': '784-128-64-10',
            'optimizer': 'Adam',
            'batch_size': batch_size
        }
    )
    print("   Model saved successfully!")
    
    # Final report
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")
    
    if test_acc >= 0.97:
        print("✓ Target accuracy (97%) ACHIEVED!")
    else:
        print(f"✗ Target accuracy (97%) not reached. Achieved: {test_acc*100:.2f}%")
    
    print(f"\nModel saved to: {save_path}")
    print("=" * 60)
    
    return network, history, test_acc


def quick_train(epochs=10):
    """Quick training for testing purposes."""
    return train_mnist(epochs=epochs, batch_size=128)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MNIST digit recognition model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--quick', action='store_true', help='Quick training (10 epochs)')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_train()
    else:
        train_mnist(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
