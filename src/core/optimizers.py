"""
Optimizers Module for AiDigit Neural Network

This module provides various optimization algorithms for training neural networks:
- SGD: Stochastic Gradient Descent
- SGDMomentum: SGD with momentum
- Adam: Adaptive Moment Estimation
- RMSprop: Root Mean Square Propagation

Each optimizer updates network parameters based on computed gradients.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional


class Optimizer(ABC):
    """
    Abstract base class for all optimizers.
    
    Optimizers update network parameters based on gradients
    computed during backpropagation.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize optimizer.
        
        Args:
            learning_rate: Step size for parameter updates
        """
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        self.lr = learning_rate
    
    @abstractmethod
    def update(self, params: Dict[str, np.ndarray], 
               grads: Dict[str, np.ndarray], 
               layer_id: int = 0) -> None:
        """
        Update parameters using gradients.
        
        Args:
            params: Dictionary of parameter arrays {'weights': ..., 'bias': ...}
            grads: Dictionary of gradient arrays {'weights': ..., 'bias': ...}
            layer_id: Identifier for the layer (used for state tracking)
        """
        pass
    
    def reset(self) -> None:
        """Reset optimizer state (momentum, etc.)"""
        pass


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    The simplest optimizer that updates parameters in the opposite
    direction of the gradient, scaled by learning rate.
    
    Update rule: w = w - lr * gradient
    
    Attributes:
        lr: Learning rate (step size)
    
    Example:
        >>> optimizer = SGD(learning_rate=0.01)
        >>> optimizer.update({'weights': W, 'bias': b}, 
        ...                  {'weights': dW, 'bias': db})
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Step size for parameter updates (default: 0.01)
        """
        super().__init__(learning_rate)
    
    def update(self, params: Dict[str, np.ndarray], 
               grads: Dict[str, np.ndarray], 
               layer_id: int = 0) -> None:
        """
        Update parameters using vanilla gradient descent.
        
        Args:
            params: Parameter dictionary
            grads: Gradient dictionary
            layer_id: Not used in vanilla SGD
        """
        for key in params:
            if key in grads and grads[key] is not None:
                params[key] -= self.lr * grads[key]
    
    def __repr__(self) -> str:
        return f"SGD(lr={self.lr})"


class SGDMomentum(Optimizer):
    """
    SGD with Momentum optimizer.
    
    Adds a velocity term that accumulates gradients over time,
    helping to accelerate learning and smooth out updates.
    
    Update rules:
        v = momentum * v - lr * gradient
        w = w + v
    
    Attributes:
        lr: Learning rate
        momentum: Momentum coefficient (typically 0.9)
        velocities: Dictionary storing velocity for each layer
    
    Example:
        >>> optimizer = SGDMomentum(learning_rate=0.01, momentum=0.9)
        >>> optimizer.update({'weights': W}, {'weights': dW}, layer_id=0)
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        """
        Initialize SGD with Momentum.
        
        Args:
            learning_rate: Step size for parameter updates (default: 0.01)
            momentum: Momentum coefficient (default: 0.9)
        """
        super().__init__(learning_rate)
        if not 0 <= momentum < 1:
            raise ValueError("Momentum must be in [0, 1)")
        self.momentum = momentum
        self.velocities: Dict[int, Dict[str, np.ndarray]] = {}
    
    def update(self, params: Dict[str, np.ndarray], 
               grads: Dict[str, np.ndarray], 
               layer_id: int = 0) -> None:
        """
        Update parameters using momentum.
        
        Args:
            params: Parameter dictionary
            grads: Gradient dictionary
            layer_id: Unique identifier for the layer
        """
        # Initialize velocities for this layer if needed
        if layer_id not in self.velocities:
            self.velocities[layer_id] = {}
            for key in params:
                self.velocities[layer_id][key] = np.zeros_like(params[key])
        
        for key in params:
            if key in grads and grads[key] is not None:
                # Update velocity
                v = self.velocities[layer_id][key]
                v = self.momentum * v - self.lr * grads[key]
                self.velocities[layer_id][key] = v
                
                # Update parameters
                params[key] += v
    
    def reset(self) -> None:
        """Reset velocity states."""
        self.velocities.clear()
    
    def __repr__(self) -> str:
        return f"SGDMomentum(lr={self.lr}, momentum={self.momentum})"


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.
    
    Combines ideas from momentum and RMSprop, maintaining both
    first moment (mean) and second moment (variance) of gradients.
    
    Update rules:
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
    
    Attributes:
        lr: Learning rate
        beta1: Exponential decay rate for first moment (default: 0.9)
        beta2: Exponential decay rate for second moment (default: 0.999)
        epsilon: Small constant for numerical stability (default: 1e-8)
    
    Example:
        >>> optimizer = Adam(learning_rate=0.001)
        >>> optimizer.update({'weights': W}, {'weights': dW}, layer_id=0)
    """
    
    def __init__(self, learning_rate: float = 0.001, 
                 beta1: float = 0.9, 
                 beta2: float = 0.999,
                 epsilon: float = 1e-8):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Step size (default: 0.001)
            beta1: Exponential decay for first moment (default: 0.9)
            beta2: Exponential decay for second moment (default: 0.999)
            epsilon: Numerical stability constant (default: 1e-8)
        """
        super().__init__(learning_rate)
        
        if not 0 <= beta1 < 1:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0 <= beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # First moment (mean of gradients)
        self.m: Dict[int, Dict[str, np.ndarray]] = {}
        # Second moment (variance of gradients)
        self.v: Dict[int, Dict[str, np.ndarray]] = {}
        # Time step
        self.t = 0
    
    def update(self, params: Dict[str, np.ndarray], 
               grads: Dict[str, np.ndarray], 
               layer_id: int = 0) -> None:
        """
        Update parameters using Adam algorithm.
        
        Args:
            params: Parameter dictionary
            grads: Gradient dictionary
            layer_id: Unique identifier for the layer
        """
        self.t += 1
        
        # Initialize moment estimates for this layer if needed
        if layer_id not in self.m:
            self.m[layer_id] = {}
            self.v[layer_id] = {}
            for key in params:
                self.m[layer_id][key] = np.zeros_like(params[key])
                self.v[layer_id][key] = np.zeros_like(params[key])
        
        for key in params:
            if key not in grads or grads[key] is None:
                continue
            
            g = grads[key]
            
            # Update biased first moment estimate
            self.m[layer_id][key] = (
                self.beta1 * self.m[layer_id][key] + 
                (1 - self.beta1) * g
            )
            
            # Update biased second moment estimate
            self.v[layer_id][key] = (
                self.beta2 * self.v[layer_id][key] + 
                (1 - self.beta2) * np.square(g)
            )
            
            # Bias correction
            m_hat = self.m[layer_id][key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[layer_id][key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def reset(self) -> None:
        """Reset moment estimates and time step."""
        self.m.clear()
        self.v.clear()
        self.t = 0
    
    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2})"


class RMSprop(Optimizer):
    """
    RMSprop (Root Mean Square Propagation) optimizer.
    
    Maintains a moving average of squared gradients to normalize
    the gradient, adapting learning rate per parameter.
    
    Update rules:
        v = decay * v + (1 - decay) * gradient^2
        w = w - lr * gradient / (sqrt(v) + epsilon)
    
    Attributes:
        lr: Learning rate
        decay: Decay rate for moving average (default: 0.9)
        epsilon: Small constant for numerical stability
    
    Example:
        >>> optimizer = RMSprop(learning_rate=0.001, decay=0.9)
        >>> optimizer.update({'weights': W}, {'weights': dW}, layer_id=0)
    """
    
    def __init__(self, learning_rate: float = 0.001, 
                 decay: float = 0.9,
                 epsilon: float = 1e-8):
        """
        Initialize RMSprop optimizer.
        
        Args:
            learning_rate: Step size (default: 0.001)
            decay: Decay rate for moving average (default: 0.9)
            epsilon: Numerical stability constant (default: 1e-8)
        """
        super().__init__(learning_rate)
        
        if not 0 <= decay < 1:
            raise ValueError("decay must be in [0, 1)")
        
        self.decay = decay
        self.epsilon = epsilon
        
        # Moving average of squared gradients
        self.v: Dict[int, Dict[str, np.ndarray]] = {}
    
    def update(self, params: Dict[str, np.ndarray], 
               grads: Dict[str, np.ndarray], 
               layer_id: int = 0) -> None:
        """
        Update parameters using RMSprop algorithm.
        
        Args:
            params: Parameter dictionary
            grads: Gradient dictionary
            layer_id: Unique identifier for the layer
        """
        # Initialize squared gradient cache for this layer if needed
        if layer_id not in self.v:
            self.v[layer_id] = {}
            for key in params:
                self.v[layer_id][key] = np.zeros_like(params[key])
        
        for key in params:
            if key not in grads or grads[key] is None:
                continue
            
            g = grads[key]
            
            # Update squared gradient moving average
            self.v[layer_id][key] = (
                self.decay * self.v[layer_id][key] + 
                (1 - self.decay) * np.square(g)
            )
            
            # Update parameters
            params[key] -= (
                self.lr * g / 
                (np.sqrt(self.v[layer_id][key]) + self.epsilon)
            )
    
    def reset(self) -> None:
        """Reset squared gradient cache."""
        self.v.clear()
    
    def __repr__(self) -> str:
        return f"RMSprop(lr={self.lr}, decay={self.decay})"


def get_optimizer(name: str, **kwargs) -> Optimizer:
    """
    Factory function to get optimizer by name.
    
    Args:
        name: Name of optimizer ('sgd', 'sgd_momentum', 'adam', 'rmsprop')
        **kwargs: Additional arguments for the optimizer
    
    Returns:
        Initialized optimizer instance
    
    Raises:
        ValueError: If optimizer name is unknown
    
    Example:
        >>> optimizer = get_optimizer('adam', learning_rate=0.001)
        >>> optimizer = get_optimizer('sgd', learning_rate=0.01)
    """
    optimizers = {
        'sgd': SGD,
        'sgd_momentum': SGDMomentum,
        'momentum': SGDMomentum,
        'adam': Adam,
        'rmsprop': RMSprop,
        'rms_prop': RMSprop,
    }
    
    name_lower = name.lower()
    if name_lower not in optimizers:
        valid = ', '.join(optimizers.keys())
        raise ValueError(f"Unknown optimizer '{name}'. Valid options: {valid}")
    
    return optimizers[name_lower](**kwargs)


# Convenience exports
__all__ = [
    'Optimizer',
    'SGD',
    'SGDMomentum',
    'Adam',
    'RMSprop',
    'get_optimizer',
]
