"""
Training Module

Training loop implementation with early stopping, learning rate scheduling,
and checkpoint management.
"""

import numpy as np
import time
from typing import Tuple, Optional, Callable, Dict, List, Any
from pathlib import Path

# Import from other modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.network import NeuralNetwork
from core.losses import Loss, CrossEntropyLoss
from core.optimizers import Optimizer, Adam
from core.metrics import accuracy
from utils.math_utils import batch_iterator


class EarlyStopping:
    """
    Early stopping callback to prevent overfitting.
    
    Monitors a metric and stops training when it stops improving.
    
    Attributes:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        restore_best: Whether to restore best weights on stop
        
    Example:
        >>> early_stop = EarlyStopping(patience=5, min_delta=0.001)
        >>> for epoch in range(100):
        ...     val_loss = train_epoch()
        ...     if early_stop(val_loss):
        ...         print("Early stopping!")
        ...         break
    """
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.0001,
                 mode: str = 'min',
                 restore_best: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Epochs to wait before stopping
            min_delta: Minimum improvement threshold
            mode: 'min' or 'max' for monitoring direction
            restore_best: Restore best weights when stopped
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.best_weights = None
        self.counter = 0
        self.stopped_epoch = 0
    
    def __call__(self, current_value: float, 
                 weights: Optional[List] = None) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_value: Current metric value
            weights: Current model weights (for restoration)
            
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.counter = 0
            if self.restore_best and weights is not None:
                self.best_weights = [w.copy() if w is not None else None 
                                    for w in weights]
        else:
            self.counter += 1
        
        return self.counter >= self.patience
    
    def reset(self):
        """Reset early stopping state."""
        self.best_value = np.inf if self.mode == 'min' else -np.inf
        self.best_weights = None
        self.counter = 0


class LearningRateScheduler:
    """
    Learning rate scheduler for adaptive learning rates.
    
    Supports various scheduling strategies:
    - Step decay
    - Exponential decay
    - Cosine annealing
    - Reduce on plateau
    """
    
    def __init__(self, 
                 initial_lr: float = 0.001,
                 schedule_type: str = 'constant',
                 **kwargs):
        """
        Initialize scheduler.
        
        Args:
            initial_lr: Initial learning rate
            schedule_type: Type of schedule ('constant', 'step', 'exponential', 
                          'cosine', 'plateau')
            **kwargs: Schedule-specific parameters
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.schedule_type = schedule_type
        self.kwargs = kwargs
        self.epoch = 0
        
        # For plateau scheduling
        self.best_value = np.inf
        self.plateau_counter = 0
    
    def step(self, epoch: int = None, metric: float = None) -> float:
        """
        Update learning rate based on epoch or metric.
        
        Args:
            epoch: Current epoch (optional, auto-increments)
            metric: Current metric value (for plateau scheduling)
            
        Returns:
            New learning rate
        """
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch += 1
        
        if self.schedule_type == 'constant':
            pass
        
        elif self.schedule_type == 'step':
            step_size = self.kwargs.get('step_size', 10)
            gamma = self.kwargs.get('gamma', 0.1)
            self.current_lr = self.initial_lr * (gamma ** (self.epoch // step_size))
        
        elif self.schedule_type == 'exponential':
            gamma = self.kwargs.get('gamma', 0.95)
            self.current_lr = self.initial_lr * (gamma ** self.epoch)
        
        elif self.schedule_type == 'cosine':
            T_max = self.kwargs.get('T_max', 100)
            eta_min = self.kwargs.get('eta_min', 0)
            self.current_lr = eta_min + (self.initial_lr - eta_min) * \
                             (1 + np.cos(np.pi * self.epoch / T_max)) / 2
        
        elif self.schedule_type == 'plateau':
            if metric is not None:
                factor = self.kwargs.get('factor', 0.1)
                patience = self.kwargs.get('patience', 5)
                min_lr = self.kwargs.get('min_lr', 1e-6)
                
                if metric < self.best_value:
                    self.best_value = metric
                    self.plateau_counter = 0
                else:
                    self.plateau_counter += 1
                
                if self.plateau_counter >= patience:
                    self.current_lr = max(self.current_lr * factor, min_lr)
                    self.plateau_counter = 0
        
        return self.current_lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr


class TrainingHistory:
    """
    Records training history for visualization and analysis.
    """
    
    def __init__(self):
        """Initialize history."""
        self.history: Dict[str, List[float]] = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': [],
            'lr': [],
            'epoch_time': []
        }
    
    def record(self, **metrics):
        """Record metrics for current epoch."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def get(self, key: str) -> List[float]:
        """Get history for a metric."""
        return self.history.get(key, [])
    
    def summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            'epochs': len(self.history['loss']),
            'best_loss': min(self.history['loss']) if self.history['loss'] else None,
            'best_val_loss': min(self.history['val_loss']) if self.history['val_loss'] else None,
            'best_accuracy': max(self.history['accuracy']) if self.history['accuracy'] else None,
            'best_val_accuracy': max(self.history['val_accuracy']) if self.history['val_accuracy'] else None,
            'final_lr': self.history['lr'][-1] if self.history['lr'] else None,
        }


class Trainer:
    """
    Neural network trainer with comprehensive training features.
    
    Features:
    - Mini-batch training
    - Validation during training
    - Early stopping
    - Learning rate scheduling
    - Checkpoint saving
    - Progress reporting
    
    Example:
        >>> trainer = Trainer(network, optimizer, loss_fn)
        >>> history = trainer.fit(X_train, y_train, X_val, y_val, epochs=50)
    """
    
    def __init__(self,
                 network: NeuralNetwork,
                 optimizer: Optimizer = None,
                 loss_fn: Loss = None,
                 checkpoint_dir: str = None):
        """
        Initialize trainer.
        
        Args:
            network: Neural network to train
            optimizer: Optimizer for weight updates
            loss_fn: Loss function
            checkpoint_dir: Directory for saving checkpoints
        """
        self.network = network
        self.optimizer = optimizer or Adam(learning_rate=0.001)
        self.loss_fn = loss_fn or CrossEntropyLoss()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = TrainingHistory()
        self.early_stopping = None
        self.lr_scheduler = None
    
    def _forward_backward(self, X_batch: np.ndarray, 
                          y_batch: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Perform forward and backward pass.
        
        Returns:
            (loss, predictions)
        """
        # Forward pass
        predictions = self.network.forward(X_batch)
        
        # Compute loss
        loss = self.loss_fn.forward(predictions, y_batch)
        
        # Backward pass
        gradient = self.loss_fn.backward(predictions, y_batch)
        self.network.backward(gradient)
        
        return loss, predictions
    
    def _update_weights(self):
        """Update network weights using optimizer."""
        params = self.network.get_parameters()
        grads = self.network.get_gradients()
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.optimizer.update(param, grad, layer_id=i)
    
    def train_epoch(self, X: np.ndarray, y: np.ndarray,
                    batch_size: int = 32,
                    shuffle: bool = True) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            X: Training features
            y: Training labels
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            (epoch_loss, epoch_accuracy)
        """
        self.network.set_training(True)
        
        total_loss = 0
        total_correct = 0
        n_batches = 0
        
        for X_batch, y_batch in batch_iterator(X, y, batch_size, shuffle):
            # Forward and backward
            loss, predictions = self._forward_backward(X_batch, y_batch)
            
            # Update weights
            self._update_weights()
            
            # Track metrics
            total_loss += loss
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y_batch, axis=1) if y_batch.ndim > 1 else y_batch
            total_correct += np.sum(pred_classes == true_classes)
            n_batches += 1
        
        epoch_loss = total_loss / n_batches
        epoch_accuracy = total_correct / len(X)
        
        return epoch_loss, epoch_accuracy
    
    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 batch_size: int = 32) -> Tuple[float, float]:
        """
        Evaluate model on data.
        
        Args:
            X: Features
            y: Labels
            batch_size: Batch size
            
        Returns:
            (loss, accuracy)
        """
        self.network.set_training(False)
        
        total_loss = 0
        total_correct = 0
        n_batches = 0
        
        for X_batch, y_batch in batch_iterator(X, y, batch_size, shuffle=False):
            predictions = self.network.forward(X_batch)
            loss = self.loss_fn.forward(predictions, y_batch)
            
            total_loss += loss
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y_batch, axis=1) if y_batch.ndim > 1 else y_batch
            total_correct += np.sum(pred_classes == true_classes)
            n_batches += 1
        
        return total_loss / n_batches, total_correct / len(X)
    
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None,
            epochs: int = 50,
            batch_size: int = 32,
            early_stopping: EarlyStopping = None,
            lr_scheduler: LearningRateScheduler = None,
            verbose: int = 1,
            checkpoint_freq: int = 0) -> TrainingHistory:
        """
        Train the network.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            early_stopping: Early stopping callback
            lr_scheduler: Learning rate scheduler
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
            checkpoint_freq: Save checkpoint every N epochs (0=disabled)
            
        Returns:
            Training history
        """
        self.early_stopping = early_stopping
        self.lr_scheduler = lr_scheduler
        
        has_validation = X_val is not None and y_val is not None
        
        if verbose:
            print(f"Training started - {epochs} epochs, batch size {batch_size}")
            print(f"Training samples: {len(X_train)}", end="")
            if has_validation:
                print(f", Validation samples: {len(X_val)}")
            else:
                print()
            print("-" * 60)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train epoch
            train_loss, train_acc = self.train_epoch(X_train, y_train, batch_size)
            
            # Validate
            val_loss, val_acc = None, None
            if has_validation:
                val_loss, val_acc = self.evaluate(X_val, y_val, batch_size)
            
            # Update learning rate
            current_lr = self.optimizer.lr
            if lr_scheduler:
                metric = val_loss if has_validation else train_loss
                current_lr = lr_scheduler.step(epoch, metric)
                self.optimizer.lr = current_lr
            
            epoch_time = time.time() - epoch_start
            
            # Record history
            self.history.record(
                loss=train_loss,
                accuracy=train_acc,
                val_loss=val_loss if val_loss is not None else train_loss,
                val_accuracy=val_acc if val_acc is not None else train_acc,
                lr=current_lr,
                epoch_time=epoch_time
            )
            
            # Print progress
            if verbose:
                msg = f"Epoch {epoch+1:3d}/{epochs} - "
                msg += f"loss: {train_loss:.4f} - acc: {train_acc:.4f}"
                if has_validation:
                    msg += f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
                msg += f" - lr: {current_lr:.6f} - {epoch_time:.2f}s"
                print(msg)
            
            # Checkpoint
            if checkpoint_freq > 0 and (epoch + 1) % checkpoint_freq == 0:
                self._save_checkpoint(epoch)
            
            # Early stopping
            if early_stopping:
                monitor_value = val_loss if has_validation else train_loss
                weights = self._get_weights()
                
                if early_stopping(monitor_value, weights):
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    
                    if early_stopping.restore_best and early_stopping.best_weights:
                        self._set_weights(early_stopping.best_weights)
                        if verbose:
                            print(f"Restored best weights (val_loss: {early_stopping.best_value:.4f})")
                    break
        
        if verbose:
            print("-" * 60)
            summary = self.history.summary()
            print(f"Training completed - Best val_acc: {summary['best_val_accuracy']:.4f}")
        
        return self.history
    
    def _get_weights(self) -> List[np.ndarray]:
        """Get all network weights."""
        weights = []
        for layer in self.network.layers:
            if hasattr(layer, 'weights'):
                weights.append(layer.weights.copy())
            if hasattr(layer, 'bias') and layer.bias is not None:
                weights.append(layer.bias.copy())
        return weights
    
    def _set_weights(self, weights: List[np.ndarray]):
        """Set network weights."""
        idx = 0
        for layer in self.network.layers:
            if hasattr(layer, 'weights'):
                layer.weights = weights[idx].copy()
                idx += 1
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias = weights[idx].copy()
                idx += 1
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        filepath = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.npz"
        self.network.save(str(filepath))


def train_model(network: NeuralNetwork,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray = None,
                y_val: np.ndarray = None,
                epochs: int = 50,
                batch_size: int = 32,
                learning_rate: float = 0.001,
                early_stopping_patience: int = 10,
                verbose: int = 1) -> Tuple[NeuralNetwork, TrainingHistory]:
    """
    Convenience function to train a network with sensible defaults.
    
    Args:
        network: Neural network to train
        X_train: Training features
        y_train: Training labels (one-hot encoded)
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        early_stopping_patience: Patience for early stopping
        verbose: Verbosity level
        
    Returns:
        (trained_network, training_history)
    """
    optimizer = Adam(learning_rate=learning_rate)
    loss_fn = CrossEntropyLoss()
    
    trainer = Trainer(network, optimizer, loss_fn)
    
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=0.0001,
        mode='min',
        restore_best=True
    )
    
    lr_scheduler = LearningRateScheduler(
        initial_lr=learning_rate,
        schedule_type='plateau',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    history = trainer.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping=early_stopping,
        lr_scheduler=lr_scheduler,
        verbose=verbose
    )
    
    return network, history


# Convenience exports
__all__ = [
    'EarlyStopping',
    'LearningRateScheduler',
    'TrainingHistory',
    'Trainer',
    'train_model',
]
