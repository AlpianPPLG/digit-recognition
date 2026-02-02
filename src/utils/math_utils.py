"""
Mathematical Utilities for Neural Network Operations

This module provides low-level mathematical functions with numerical stability
for neural network computations. All functions are implemented using NumPy.

Functions:
    - matmul: Matrix multiplication with shape validation
    - clip: Clip values for numerical stability
    - stable_softmax: Numerically stable softmax
    - stable_log: Logarithm with numerical stability
    - stable_cross_entropy: Numerically stable cross-entropy
"""

import numpy as np
from typing import Union, Tuple

# Type alias for array-like inputs
ArrayLike = Union[np.ndarray, list, float]

# Constants for numerical stability
EPSILON = 1e-15
CLIP_MIN = 1e-15
CLIP_MAX = 1 - 1e-15


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication with shape validation.
    
    Performs matrix multiplication A @ B and validates that the shapes
    are compatible for multiplication.
    
    Args:
        a: First matrix of shape (m, n)
        b: Second matrix of shape (n, p)
    
    Returns:
        Result matrix of shape (m, p)
    
    Raises:
        ValueError: If shapes are incompatible for matrix multiplication
    
    Example:
        >>> a = np.array([[1, 2], [3, 4]])
        >>> b = np.array([[5, 6], [7, 8]])
        >>> matmul(a, b)
        array([[19, 22],
               [43, 50]])
    """
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Handle 1D arrays
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    
    # Validate shapes
    if a.shape[-1] != b.shape[0]:
        raise ValueError(
            f"Incompatible shapes for matrix multiplication: "
            f"{a.shape} and {b.shape}. "
            f"Last dimension of first array ({a.shape[-1]}) must match "
            f"first dimension of second array ({b.shape[0]})"
        )
    
    return np.dot(a, b)


def clip(x: np.ndarray, min_val: float = CLIP_MIN, max_val: float = CLIP_MAX) -> np.ndarray:
    """
    Clip values to a specified range for numerical stability.
    
    This is commonly used before taking logarithms to avoid log(0).
    
    Args:
        x: Input array
        min_val: Minimum value (default: 1e-15)
        max_val: Maximum value (default: 1 - 1e-15)
    
    Returns:
        Clipped array with values in [min_val, max_val]
    
    Example:
        >>> x = np.array([0, 0.5, 1])
        >>> clip(x)
        array([1.e-15, 5.e-01, 1.e+00 - 1e-15])
    """
    return np.clip(x, min_val, max_val)


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax function.
    
    Computes softmax while avoiding numerical overflow by subtracting
    the maximum value before exponentiation.
    
    Formula:
        softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    
    Args:
        x: Input array of any shape
        axis: Axis along which to compute softmax (default: -1, last axis)
    
    Returns:
        Softmax probabilities with same shape as input.
        Values sum to 1 along the specified axis.
    
    Example:
        >>> x = np.array([[1, 2, 3], [1, 2, 3]])
        >>> stable_softmax(x, axis=1)
        array([[0.09003057, 0.24472847, 0.66524096],
               [0.09003057, 0.24472847, 0.66524096]])
    """
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    
    # Normalize
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def stable_log(x: np.ndarray, eps: float = EPSILON) -> np.ndarray:
    """
    Numerically stable logarithm.
    
    Adds a small epsilon to avoid log(0) = -inf.
    
    Args:
        x: Input array (should be positive)
        eps: Small value to add for stability (default: 1e-15)
    
    Returns:
        Natural logarithm of (x + eps)
    
    Example:
        >>> stable_log(np.array([0, 0.5, 1]))
        array([-34.53877639,  -0.69314718,   0.        ])
    """
    return np.log(x + eps)


def stable_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, 
                         eps: float = EPSILON) -> np.ndarray:
    """
    Numerically stable cross-entropy loss.
    
    Computes: -sum(y_true * log(y_pred + eps))
    
    Args:
        y_true: True labels (one-hot encoded), shape (batch_size, num_classes)
        y_pred: Predicted probabilities, shape (batch_size, num_classes)
        eps: Small value for numerical stability
    
    Returns:
        Cross-entropy loss for each sample, shape (batch_size,)
    
    Example:
        >>> y_true = np.array([[1, 0, 0], [0, 1, 0]])
        >>> y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
        >>> stable_cross_entropy(y_true, y_pred)
        array([0.10536052, 0.22314355])
    """
    # Clip predictions to avoid log(0)
    y_pred_clipped = clip(y_pred, eps, 1 - eps)
    
    # Compute cross-entropy
    return -np.sum(y_true * np.log(y_pred_clipped), axis=-1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function.
    
    Uses different formulas for positive and negative values to avoid overflow.
    
    Formula:
        sigmoid(x) = 1 / (1 + exp(-x))  for x >= 0
        sigmoid(x) = exp(x) / (1 + exp(x))  for x < 0
    
    Args:
        x: Input array
    
    Returns:
        Sigmoid values in range (0, 1)
    
    Example:
        >>> sigmoid(np.array([-1, 0, 1]))
        array([0.26894142, 0.5       , 0.73105858])
    """
    # Create output array
    result = np.zeros_like(x, dtype=np.float64)
    
    # For positive values: 1 / (1 + exp(-x))
    pos_mask = x >= 0
    result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    
    # For negative values: exp(x) / (1 + exp(x))
    neg_mask = ~pos_mask
    exp_x = np.exp(x[neg_mask])
    result[neg_mask] = exp_x / (1 + exp_x)
    
    return result


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid function.
    
    Formula: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    
    Args:
        x: Input array (raw values, not sigmoid output)
    
    Returns:
        Derivative values
    """
    s = sigmoid(x)
    return s * (1 - s)


def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU) activation.
    
    Formula: relu(x) = max(0, x)
    
    Args:
        x: Input array
    
    Returns:
        ReLU activated values (all negative values become 0)
    
    Example:
        >>> relu(np.array([-2, -1, 0, 1, 2]))
        array([0, 0, 0, 1, 2])
    """
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU function.
    
    Formula: relu'(x) = 1 if x > 0, else 0
    
    Args:
        x: Input array (raw values, not ReLU output)
    
    Returns:
        Derivative values (1 for positive, 0 for non-positive)
    """
    return (x > 0).astype(np.float64)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Leaky ReLU activation.
    
    Formula: leaky_relu(x) = x if x > 0, else alpha * x
    
    Args:
        x: Input array
        alpha: Slope for negative values (default: 0.01)
    
    Returns:
        Leaky ReLU activated values
    
    Example:
        >>> leaky_relu(np.array([-2, -1, 0, 1, 2]), alpha=0.1)
        array([-0.2, -0.1,  0. ,  1. ,  2. ])
    """
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Derivative of Leaky ReLU function.
    
    Formula: leaky_relu'(x) = 1 if x > 0, else alpha
    
    Args:
        x: Input array
        alpha: Slope for negative values (default: 0.01)
    
    Returns:
        Derivative values
    """
    return np.where(x > 0, 1.0, alpha)


def softmax_derivative(softmax_output: np.ndarray) -> np.ndarray:
    """
    Derivative (Jacobian) of softmax function.
    
    For a single sample, the Jacobian is:
        J[i,j] = s[i] * (delta[i,j] - s[j])
    where s is the softmax output and delta is Kronecker delta.
    
    Note: This returns the diagonal of the Jacobian for element-wise derivative.
    For full Jacobian, use softmax_jacobian().
    
    Args:
        softmax_output: Output from softmax function, shape (batch_size, num_classes)
    
    Returns:
        Element-wise derivative: s * (1 - s)
    """
    return softmax_output * (1 - softmax_output)


def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded vectors.
    
    Args:
        labels: Integer labels, shape (num_samples,)
        num_classes: Total number of classes
    
    Returns:
        One-hot encoded array, shape (num_samples, num_classes)
    
    Example:
        >>> one_hot_encode(np.array([0, 2, 1]), num_classes=3)
        array([[1., 0., 0.],
               [0., 0., 1.],
               [0., 1., 0.]])
    """
    num_samples = len(labels)
    one_hot = np.zeros((num_samples, num_classes), dtype=np.float64)
    one_hot[np.arange(num_samples), labels] = 1.0
    return one_hot


def one_hot_decode(one_hot: np.ndarray) -> np.ndarray:
    """
    Convert one-hot encoded vectors to integer labels.
    
    Args:
        one_hot: One-hot encoded array, shape (num_samples, num_classes)
    
    Returns:
        Integer labels, shape (num_samples,)
    
    Example:
        >>> one_hot_decode(np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))
        array([0, 2, 1])
    """
    return np.argmax(one_hot, axis=-1)


def normalize(x: np.ndarray, axis: int = None) -> np.ndarray:
    """
    Normalize array to range [0, 1].
    
    Args:
        x: Input array
        axis: Axis along which to normalize (None for global)
    
    Returns:
        Normalized array
    """
    x_min = np.min(x, axis=axis, keepdims=True)
    x_max = np.max(x, axis=axis, keepdims=True)
    
    # Avoid division by zero
    range_val = x_max - x_min
    range_val = np.where(range_val == 0, 1, range_val)
    
    return (x - x_min) / range_val


def standardize(x: np.ndarray, axis: int = None, eps: float = EPSILON) -> np.ndarray:
    """
    Standardize array to zero mean and unit variance.
    
    Args:
        x: Input array
        axis: Axis along which to standardize (None for global)
        eps: Small value to avoid division by zero
    
    Returns:
        Standardized array with mean ≈ 0 and std ≈ 1
    """
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    
    return (x - mean) / (std + eps)


def batch_iterator(X: np.ndarray, y: np.ndarray, batch_size: int, 
                   shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]: # pyright: ignore[reportInvalidTypeForm]
    """
    Generator that yields mini-batches from data.
    
    Args:
        X: Input features, shape (num_samples, ...)
        y: Labels, shape (num_samples, ...)
        batch_size: Size of each mini-batch
        shuffle: Whether to shuffle data before batching
    
    Yields:
        Tuple of (X_batch, y_batch)
    
    Example:
        >>> X = np.arange(10).reshape(10, 1)
        >>> y = np.arange(10)
        >>> for X_batch, y_batch in batch_iterator(X, y, batch_size=3):
        ...     print(X_batch.shape, y_batch.shape)
        (3, 1) (3,)
        (3, 1) (3,)
        (3, 1) (3,)
        (1, 1) (1,)
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy score.
    
    Args:
        y_true: True labels (integers or one-hot)
        y_pred: Predicted labels or probabilities
    
    Returns:
        Accuracy as a float in range [0, 1]
    """
    # Convert one-hot to integers if needed
    if y_true.ndim > 1:
        y_true = one_hot_decode(y_true)
    if y_pred.ndim > 1:
        y_pred = one_hot_decode(y_pred)
    
    return np.mean(y_true == y_pred)
