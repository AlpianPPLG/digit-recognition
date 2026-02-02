"""
Loss Functions
==============

This module contains loss function implementations for training neural networks.
Each loss function computes the error between predictions and targets,
along with the gradient for backpropagation.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class Loss(ABC):
    """
    Abstract base class for loss functions.
    
    All loss implementations should inherit from this class.
    """
    
    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute loss value.
        
        Args:
            y_pred: Predicted values
            y_true: True/target values
            
        Returns:
            Scalar loss value
        """
        pass
    
    @abstractmethod
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. predictions.
        
        Args:
            y_pred: Predicted values
            y_true: True/target values
            
        Returns:
            Gradient array with same shape as y_pred
        """
        pass
    
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Allow calling loss as function."""
        return self.forward(y_pred, y_true)


class CrossEntropyLoss(Loss):
    """
    Cross-Entropy Loss for classification.
    
    Formula: L = -Σ y_true * log(y_pred)
    
    Used with softmax activation for multi-class classification.
    Handles numerical stability with small epsilon value.
    
    Example:
        >>> loss = CrossEntropyLoss()
        >>> y_pred = np.array([[0.7, 0.2, 0.1]])  # softmax output
        >>> y_true = np.array([[1, 0, 0]])  # one-hot encoded
        >>> loss.forward(y_pred, y_true)
        0.357
    """
    
    def __init__(self, epsilon: float = 1e-15):
        """
        Initialize cross-entropy loss.
        
        Args:
            epsilon: Small value for numerical stability
        """
        self.epsilon = epsilon
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            y_pred: Predicted probabilities (batch_size, num_classes)
            y_true: True labels, one-hot encoded (batch_size, num_classes)
            
        Returns:
            Mean cross-entropy loss
        """
        # Clip predictions for numerical stability
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Compute cross-entropy: -sum(y_true * log(y_pred))
        batch_size = y_pred.shape[0]
        loss = -np.sum(y_true * np.log(y_pred)) / batch_size
        
        return loss
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute gradient of cross-entropy loss.
        
        When combined with softmax, the gradient simplifies to:
        dL/dz = y_pred - y_true
        
        Args:
            y_pred: Predicted probabilities
            y_true: True labels (one-hot encoded)
            
        Returns:
            Gradient w.r.t. softmax input
        """
        # For softmax + cross-entropy, gradient is simply (y_pred - y_true)
        batch_size = y_pred.shape[0]
        return (y_pred - y_true) / batch_size


class MSELoss(Loss):
    """
    Mean Squared Error Loss.
    
    Formula: L = (1/n) * Σ (y_pred - y_true)²
    
    Used for regression tasks.
    
    Example:
        >>> loss = MSELoss()
        >>> y_pred = np.array([[1.0, 2.0]])
        >>> y_true = np.array([[1.5, 2.5]])
        >>> loss.forward(y_pred, y_true)
        0.25
    """
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute mean squared error.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            Mean squared error
        """
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute gradient of MSE.
        
        Formula: dL/dy_pred = 2 * (y_pred - y_true) / n
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            Gradient w.r.t. predictions
        """
        batch_size = y_pred.shape[0]
        return 2 * (y_pred - y_true) / (batch_size * y_pred.shape[1])


class BinaryCrossEntropyLoss(Loss):
    """
    Binary Cross-Entropy Loss.
    
    Formula: L = -[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
    
    Used for binary classification with sigmoid activation.
    
    Example:
        >>> loss = BinaryCrossEntropyLoss()
        >>> y_pred = np.array([[0.8]])  # sigmoid output
        >>> y_true = np.array([[1.0]])  # binary label
        >>> loss.forward(y_pred, y_true)
        0.223
    """
    
    def __init__(self, epsilon: float = 1e-15):
        """
        Initialize binary cross-entropy loss.
        
        Args:
            epsilon: Small value for numerical stability
        """
        self.epsilon = epsilon
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss.
        
        Args:
            y_pred: Predicted probabilities (0 to 1)
            y_true: True binary labels (0 or 1)
            
        Returns:
            Mean binary cross-entropy loss
        """
        # Clip for numerical stability
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # BCE formula
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        return np.mean(loss)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute gradient of binary cross-entropy.
        
        Formula: dL/dy_pred = -(y_true/y_pred - (1-y_true)/(1-y_pred))
        
        Args:
            y_pred: Predicted probabilities
            y_true: True binary labels
            
        Returns:
            Gradient w.r.t. predictions
        """
        # Clip for numerical stability
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        batch_size = y_pred.shape[0]
        grad = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / batch_size
        
        return grad


class HuberLoss(Loss):
    """
    Huber Loss (Smooth L1 Loss).
    
    Combines MSE and MAE, less sensitive to outliers than MSE.
    
    Formula:
        L = 0.5 * (y_pred - y_true)² if |y_pred - y_true| < delta
        L = delta * |y_pred - y_true| - 0.5 * delta² otherwise
    """
    
    def __init__(self, delta: float = 1.0):
        """
        Initialize Huber loss.
        
        Args:
            delta: Threshold for switching between MSE and MAE
        """
        self.delta = delta
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute Huber loss.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            Mean Huber loss
        """
        error = y_pred - y_true
        abs_error = np.abs(error)
        
        # Quadratic for small errors, linear for large
        quadratic = 0.5 * error ** 2
        linear = self.delta * abs_error - 0.5 * self.delta ** 2
        
        loss = np.where(abs_error <= self.delta, quadratic, linear)
        
        return np.mean(loss)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute gradient of Huber loss.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            Gradient w.r.t. predictions
        """
        error = y_pred - y_true
        abs_error = np.abs(error)
        
        # Gradient: error for small errors, delta * sign(error) for large
        grad = np.where(
            abs_error <= self.delta,
            error,
            self.delta * np.sign(error)
        )
        
        batch_size = y_pred.shape[0]
        return grad / batch_size


# Loss function factory
LOSSES = {
    'cross_entropy': CrossEntropyLoss,
    'crossentropy': CrossEntropyLoss,
    'categorical_crossentropy': CrossEntropyLoss,
    'mse': MSELoss,
    'mean_squared_error': MSELoss,
    'bce': BinaryCrossEntropyLoss,
    'binary_crossentropy': BinaryCrossEntropyLoss,
    'huber': HuberLoss,
}


def get_loss(name: str, **kwargs) -> Loss:
    """
    Get loss function by name.
    
    Args:
        name: Loss function name
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function instance
        
    Raises:
        ValueError: If loss name is unknown
    """
    name = name.lower().replace(' ', '_')
    if name not in LOSSES:
        raise ValueError(f"Unknown loss: {name}. "
                        f"Available: {list(LOSSES.keys())}")
    return LOSSES[name](**kwargs)
