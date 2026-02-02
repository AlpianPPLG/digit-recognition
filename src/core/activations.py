"""
Activation Functions
====================

This module contains standalone activation functions and their derivatives
for use in neural network computations.

All functions support numpy arrays and handle numerical stability.
"""

import numpy as np
from typing import Tuple


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.
    
    Formula: σ(x) = 1 / (1 + e^(-x))
    
    Args:
        x: Input array of any shape
        
    Returns:
        Output array with same shape, values in (0, 1)
        
    Example:
        >>> sigmoid(np.array([0, 1, -1]))
        array([0.5, 0.731, 0.269])
    """
    # Clip for numerical stability
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid function.
    
    Formula: σ'(x) = σ(x) * (1 - σ(x))
    
    Args:
        x: Input array
        
    Returns:
        Derivative at each point
    """
    s = sigmoid(x)
    return s * (1.0 - s)


def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU) activation.
    
    Formula: ReLU(x) = max(0, x)
    
    Args:
        x: Input array
        
    Returns:
        Output array with negative values zeroed
        
    Example:
        >>> relu(np.array([-2, -1, 0, 1, 2]))
        array([0, 0, 0, 1, 2])
    """
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU function.
    
    Formula: ReLU'(x) = 1 if x > 0, else 0
    
    Args:
        x: Input array
        
    Returns:
        Derivative at each point (0 or 1)
    """
    return (x > 0).astype(np.float64)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Leaky ReLU activation.
    
    Formula: LeakyReLU(x) = x if x > 0, else α * x
    
    Args:
        x: Input array
        alpha: Slope for negative values (default: 0.01)
        
    Returns:
        Output array
        
    Example:
        >>> leaky_relu(np.array([-2, -1, 0, 1, 2]), alpha=0.1)
        array([-0.2, -0.1, 0, 1, 2])
    """
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Derivative of Leaky ReLU.
    
    Formula: LeakyReLU'(x) = 1 if x > 0, else α
    
    Args:
        x: Input array
        alpha: Slope for negative values
        
    Returns:
        Derivative at each point
    """
    return np.where(x > 0, 1.0, alpha)


def tanh(x: np.ndarray) -> np.ndarray:
    """
    Hyperbolic tangent activation.
    
    Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    Args:
        x: Input array
        
    Returns:
        Output array with values in (-1, 1)
    """
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of tanh function.
    
    Formula: tanh'(x) = 1 - tanh(x)^2
    
    Args:
        x: Input array
        
    Returns:
        Derivative at each point
    """
    return 1.0 - np.tanh(x) ** 2


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax activation function.
    
    Formula: softmax(x_i) = e^(x_i) / Σ e^(x_j)
    
    Converts logits to probability distribution that sums to 1.
    Uses numerical stability trick of subtracting max value.
    
    Args:
        x: Input array of shape (batch_size, num_classes)
        
    Returns:
        Probability distribution array of same shape
        
    Example:
        >>> softmax(np.array([[1, 2, 3]]))
        array([[0.09, 0.24, 0.67]])
    """
    # Subtract max for numerical stability
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def softmax_derivative(softmax_output: np.ndarray) -> np.ndarray:
    """
    Derivative of softmax function (Jacobian).
    
    Note: Usually combined with cross-entropy loss for simplification.
    The combined derivative is simply (softmax_output - target).
    
    Args:
        softmax_output: Output from softmax function
        
    Returns:
        Jacobian matrix
    """
    # This returns the diagonal of the Jacobian
    # Full Jacobian: J_ij = s_i * (δ_ij - s_j)
    return softmax_output * (1.0 - softmax_output)


def linear(x: np.ndarray) -> np.ndarray:
    """
    Linear (identity) activation.
    
    Formula: f(x) = x
    
    Args:
        x: Input array
        
    Returns:
        Same array unchanged
    """
    return x


def linear_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of linear activation.
    
    Formula: f'(x) = 1
    
    Args:
        x: Input array
        
    Returns:
        Array of ones with same shape
    """
    return np.ones_like(x)


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Exponential Linear Unit (ELU) activation.
    
    Formula: ELU(x) = x if x > 0, else α * (e^x - 1)
    
    Args:
        x: Input array
        alpha: Scale for negative values
        
    Returns:
        Output array
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def elu_derivative(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Derivative of ELU.
    
    Formula: ELU'(x) = 1 if x > 0, else ELU(x) + α
    
    Args:
        x: Input array
        alpha: Scale for negative values
        
    Returns:
        Derivative at each point
    """
    return np.where(x > 0, 1.0, elu(x, alpha) + alpha)


def swish(x: np.ndarray) -> np.ndarray:
    """
    Swish activation (self-gated activation).
    
    Formula: swish(x) = x * sigmoid(x)
    
    Args:
        x: Input array
        
    Returns:
        Output array
    """
    return x * sigmoid(x)


def swish_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of swish function.
    
    Formula: swish'(x) = swish(x) + sigmoid(x) * (1 - swish(x))
    
    Args:
        x: Input array
        
    Returns:
        Derivative at each point
    """
    s = sigmoid(x)
    sw = x * s
    return sw + s * (1.0 - sw)


# Dictionary mapping names to functions
ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'leaky_relu': (leaky_relu, leaky_relu_derivative),
    'tanh': (tanh, tanh_derivative),
    'softmax': (softmax, softmax_derivative),
    'linear': (linear, linear_derivative),
    'elu': (elu, elu_derivative),
    'swish': (swish, swish_derivative),
}


def get_activation(name: str) -> Tuple[callable, callable]:
    """
    Get activation function and its derivative by name.
    
    Args:
        name: Activation function name
        
    Returns:
        Tuple of (activation_fn, derivative_fn)
        
    Raises:
        ValueError: If activation name is unknown
    """
    name = name.lower()
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. "
                        f"Available: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]
