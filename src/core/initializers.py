"""
Weight Initializers
===================

This module contains weight initialization strategies for neural network layers.
Proper initialization is crucial for training stability and convergence.

Implemented initializers:
- Xavier/Glorot: Good for tanh/sigmoid activations
- He/Kaiming: Good for ReLU activations  
- Random: Simple random initialization
- Zeros: Initialize to zeros (for biases)
- Ones: Initialize to ones
"""

import numpy as np
from typing import Tuple, Union

Shape = Union[Tuple[int, ...], int]


def xavier_init(shape: Shape, gain: float = 1.0) -> np.ndarray:
    """
    Xavier/Glorot initialization.
    
    Designed to keep the scale of gradients roughly the same in all layers.
    Good for tanh and sigmoid activations.
    
    Formula: W ~ U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out))
    Or:      W ~ N(0, σ²) where σ² = gain² * 2 / (fan_in + fan_out)
    
    Reference:
        Glorot & Bengio, 2010: "Understanding the difficulty of training 
        deep feedforward neural networks"
    
    Args:
        shape: Shape of weight matrix (fan_in, fan_out) or tuple
        gain: Scaling factor (default: 1.0)
    
    Returns:
        Initialized weight matrix
        
    Example:
        >>> weights = xavier_init((784, 128))
        >>> weights.shape
        (784, 128)
    """
    if isinstance(shape, int):
        fan_in = fan_out = shape
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    else:
        fan_in = shape[0]
        fan_out = shape[1] if len(shape) > 1 else shape[0]
    
    # Standard deviation for Xavier initialization
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    
    return np.random.randn(*_to_tuple(shape)) * std


def xavier_uniform_init(shape: Shape, gain: float = 1.0) -> np.ndarray:
    """
    Xavier/Glorot uniform initialization.
    
    Same as xavier_init but uses uniform distribution.
    
    Formula: W ~ U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out))
    
    Args:
        shape: Shape of weight matrix
        gain: Scaling factor (default: 1.0)
    
    Returns:
        Initialized weight matrix
    """
    if isinstance(shape, int):
        fan_in = fan_out = shape
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    else:
        fan_in = shape[0]
        fan_out = shape[1] if len(shape) > 1 else shape[0]
    
    # Limit for uniform distribution
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    
    return np.random.uniform(-limit, limit, _to_tuple(shape))


def he_init(shape: Shape, mode: str = 'fan_in') -> np.ndarray:
    """
    He/Kaiming initialization.
    
    Designed for ReLU activations. Takes into account that ReLU
    zeros out half of the activations.
    
    Formula: W ~ N(0, σ²) where σ² = 2 / fan_in (or fan_out)
    
    Reference:
        He et al., 2015: "Delving Deep into Rectifiers"
    
    Args:
        shape: Shape of weight matrix (fan_in, fan_out)
        mode: 'fan_in' (default) or 'fan_out'
    
    Returns:
        Initialized weight matrix
        
    Example:
        >>> weights = he_init((784, 128))
        >>> weights.shape
        (784, 128)
    """
    if isinstance(shape, int):
        fan = shape
    elif len(shape) == 1:
        fan = shape[0]
    else:
        if mode == 'fan_in':
            fan = shape[0]
        else:
            fan = shape[1] if len(shape) > 1 else shape[0]
    
    # Standard deviation for He initialization
    std = np.sqrt(2.0 / fan)
    
    return np.random.randn(*_to_tuple(shape)) * std


def he_uniform_init(shape: Shape, mode: str = 'fan_in') -> np.ndarray:
    """
    He/Kaiming uniform initialization.
    
    Same as he_init but uses uniform distribution.
    
    Formula: W ~ U(-a, a) where a = sqrt(6 / fan_in)
    
    Args:
        shape: Shape of weight matrix
        mode: 'fan_in' (default) or 'fan_out'
    
    Returns:
        Initialized weight matrix
    """
    if isinstance(shape, int):
        fan = shape
    elif len(shape) == 1:
        fan = shape[0]
    else:
        if mode == 'fan_in':
            fan = shape[0]
        else:
            fan = shape[1] if len(shape) > 1 else shape[0]
    
    # Limit for uniform distribution
    limit = np.sqrt(6.0 / fan)
    
    return np.random.uniform(-limit, limit, _to_tuple(shape))


def random_init(shape: Shape, scale: float = 0.01) -> np.ndarray:
    """
    Simple random initialization.
    
    Initialize weights with small random values from normal distribution.
    
    Formula: W ~ N(0, scale²)
    
    Args:
        shape: Shape of weight matrix
        scale: Standard deviation (default: 0.01)
    
    Returns:
        Initialized weight matrix
    """
    return np.random.randn(*_to_tuple(shape)) * scale


def uniform_init(shape: Shape, low: float = -0.1, high: float = 0.1) -> np.ndarray:
    """
    Uniform random initialization.
    
    Formula: W ~ U(low, high)
    
    Args:
        shape: Shape of weight matrix
        low: Lower bound (default: -0.1)
        high: Upper bound (default: 0.1)
    
    Returns:
        Initialized weight matrix
    """
    return np.random.uniform(low, high, _to_tuple(shape))


def zeros_init(shape: Shape) -> np.ndarray:
    """
    Initialize to zeros.
    
    Commonly used for bias initialization.
    
    Args:
        shape: Shape of array
    
    Returns:
        Array of zeros
    """
    return np.zeros(_to_tuple(shape))


def ones_init(shape: Shape) -> np.ndarray:
    """
    Initialize to ones.
    
    Args:
        shape: Shape of array
    
    Returns:
        Array of ones
    """
    return np.ones(_to_tuple(shape))


def constant_init(shape: Shape, value: float) -> np.ndarray:
    """
    Initialize to constant value.
    
    Args:
        shape: Shape of array
        value: Constant value to fill
    
    Returns:
        Array filled with constant value
    """
    return np.full(_to_tuple(shape), value)


def orthogonal_init(shape: Shape, gain: float = 1.0) -> np.ndarray:
    """
    Orthogonal initialization.
    
    Creates a (semi-)orthogonal matrix. Good for RNNs.
    
    Reference:
        Saxe et al., 2013: "Exact solutions to the nonlinear dynamics 
        of learning in deep linear neural networks"
    
    Args:
        shape: Shape of weight matrix (should be 2D)
        gain: Scaling factor
    
    Returns:
        Orthogonal weight matrix
    """
    shape = _to_tuple(shape)
    
    if len(shape) < 2:
        raise ValueError("Orthogonal initialization requires at least 2D shape")
    
    rows = shape[0]
    cols = np.prod(shape[1:])
    
    # Generate random matrix
    flat_shape = (rows, cols) if rows >= cols else (cols, rows)
    a = np.random.randn(*flat_shape)
    
    # QR decomposition
    q, r = np.linalg.qr(a)
    
    # Make Q uniform
    d = np.diag(r)
    ph = np.sign(d)
    q *= ph
    
    if rows < cols:
        q = q.T
    
    return gain * q.reshape(shape)


def lecun_init(shape: Shape) -> np.ndarray:
    """
    LeCun initialization.
    
    Similar to He initialization but with different scaling.
    Good for SELU activation.
    
    Formula: W ~ N(0, σ²) where σ² = 1 / fan_in
    
    Args:
        shape: Shape of weight matrix
    
    Returns:
        Initialized weight matrix
    """
    if isinstance(shape, int):
        fan_in = shape
    elif len(shape) == 1:
        fan_in = shape[0]
    else:
        fan_in = shape[0]
    
    std = np.sqrt(1.0 / fan_in)
    
    return np.random.randn(*_to_tuple(shape)) * std


def _to_tuple(shape: Shape) -> Tuple[int, ...]:
    """Convert shape to tuple."""
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


def get_initializer(name: str):
    """
    Get initializer function by name.
    
    Args:
        name: Name of initializer ('xavier', 'he', 'random', 'zeros', etc.)
    
    Returns:
        Initializer function
        
    Raises:
        ValueError: If unknown initializer name
    """
    initializers = {
        'xavier': xavier_init,
        'glorot': xavier_init,
        'xavier_uniform': xavier_uniform_init,
        'glorot_uniform': xavier_uniform_init,
        'he': he_init,
        'kaiming': he_init,
        'he_uniform': he_uniform_init,
        'kaiming_uniform': he_uniform_init,
        'random': random_init,
        'normal': random_init,
        'uniform': uniform_init,
        'zeros': zeros_init,
        'ones': ones_init,
        'orthogonal': orthogonal_init,
        'lecun': lecun_init,
    }
    
    name = name.lower()
    if name not in initializers:
        raise ValueError(
            f"Unknown initializer: {name}. "
            f"Available: {list(initializers.keys())}"
        )
    
    return initializers[name]
