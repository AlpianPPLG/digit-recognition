"""
Neural Network - Main Network Container
=======================================

This module contains the main NeuralNetwork class that serves as a container
for layers and provides methods for forward pass, backward pass, and training.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from .layers import Layer


class NeuralNetwork:
    """
    Neural Network container class.
    
    A modular neural network implementation that allows adding layers
    and provides forward/backward propagation methods.
    
    Attributes:
        layers: List of Layer objects
        loss_function: Loss function for training
        optimizer: Optimizer for weight updates
        
    Example:
        >>> network = NeuralNetwork()
        >>> network.add(DenseLayer(784, 128))
        >>> network.add(ActivationLayer('relu'))
        >>> network.add(DenseLayer(128, 10))
        >>> network.add(ActivationLayer('softmax'))
    """
    
    def __init__(self):
        """Initialize an empty neural network."""
        self.layers: List[Layer] = []
        self.loss_function = None
        self.optimizer = None
        self._is_training = True
    
    def add(self, layer: Layer) -> 'NeuralNetwork':
        """
        Add a layer to the network.
        
        Args:
            layer: Layer object to add
            
        Returns:
            self for method chaining
        """
        self.layers.append(layer)
        return self
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through all layers.
        
        Args:
            x: Input array of shape (batch_size, input_size)
            
        Returns:
            Output array after passing through all layers
        """
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass through all layers (in reverse order).
        
        Args:
            gradient: Gradient from loss function
            
        Returns:
            Gradient with respect to input
        """
        grad = gradient
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make prediction (forward pass in evaluation mode).
        
        Args:
            x: Input array
            
        Returns:
            Predicted output
        """
        self._is_training = False
        output = self.forward(x)
        self._is_training = True
        return output
    
    def predict_classes(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            x: Input array
            
        Returns:
            Array of predicted class indices
        """
        probabilities = self.predict(x)
        return np.argmax(probabilities, axis=1)
    
    def get_parameters(self) -> List[Dict[str, np.ndarray]]:
        """
        Get all trainable parameters from layers.
        
        Returns:
            List of parameter dictionaries from each layer
        """
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters') and layer.parameters:
                params.append(layer.parameters)
        return params
    
    def get_gradients(self) -> List[Dict[str, np.ndarray]]:
        """
        Get all gradients from layers.
        
        Returns:
            List of gradient dictionaries from each layer
        """
        grads = []
        for layer in self.layers:
            if hasattr(layer, 'gradients') and layer.gradients:
                grads.append(layer.gradients)
        return grads
    
    def set_training(self, mode: bool = True) -> None:
        """
        Set training mode for all layers.
        
        Args:
            mode: True for training, False for evaluation
        """
        self._is_training = mode
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = mode
    
    def summary(self) -> str:
        """
        Get a summary of the network architecture.
        
        Returns:
            String representation of the network
        """
        lines = []
        lines.append("=" * 60)
        lines.append("Neural Network Summary")
        lines.append("=" * 60)
        
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            
            if hasattr(layer, 'input_size') and hasattr(layer, 'output_size'):
                shape = f"({layer.input_size}, {layer.output_size})"
                params = layer.input_size * layer.output_size + layer.output_size
            else:
                shape = "-"
                params = 0
            
            total_params += params
            lines.append(f"Layer {i + 1}: {layer_name:20} Shape: {shape:15} Params: {params}")
        
        lines.append("=" * 60)
        lines.append(f"Total Parameters: {total_params:,}")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save(self, filepath: str) -> None:
        """
        Save network weights to file.
        
        Args:
            filepath: Path to save file (.npz format)
        """
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                weights[f'layer_{i}_weights'] = layer.weights
            if hasattr(layer, 'bias'):
                weights[f'layer_{i}_bias'] = layer.bias
        
        np.savez(filepath, **weights)
    
    def load(self, filepath: str) -> None:
        """
        Load network weights from file.
        
        Args:
            filepath: Path to weights file (.npz format)
        """
        data = np.load(filepath)
        
        for i, layer in enumerate(self.layers):
            weight_key = f'layer_{i}_weights'
            bias_key = f'layer_{i}_bias'
            
            if weight_key in data and hasattr(layer, 'weights'):
                layer.weights = data[weight_key]
            if bias_key in data and hasattr(layer, 'bias'):
                layer.bias = data[bias_key]
    
    def __repr__(self) -> str:
        return f"NeuralNetwork(layers={len(self.layers)})"


class NetworkBuilder:
    """
    Builder pattern for creating neural networks.
    
    Example:
        >>> network = (NetworkBuilder()
        ...     .input(784)
        ...     .dense(128, activation='relu')
        ...     .dense(64, activation='relu')
        ...     .dense(10, activation='softmax')
        ...     .build())
    """
    
    def __init__(self):
        """Initialize builder."""
        self._layers_config = []
        self._input_size = None
    
    def input(self, size: int) -> 'NetworkBuilder':
        """
        Set input size.
        
        Args:
            size: Input dimension
        """
        self._input_size = size
        return self
    
    def dense(self, units: int, activation: Optional[str] = None,
              initializer: str = 'he') -> 'NetworkBuilder':
        """
        Add a dense layer.
        
        Args:
            units: Number of output units
            activation: Activation function name
            initializer: Weight initializer name
        """
        self._layers_config.append({
            'type': 'dense',
            'units': units,
            'activation': activation,
            'initializer': initializer
        })
        return self
    
    def dropout(self, rate: float = 0.5) -> 'NetworkBuilder':
        """
        Add a dropout layer.
        
        Args:
            rate: Dropout rate (0 to 1)
        """
        self._layers_config.append({
            'type': 'dropout',
            'rate': rate
        })
        return self
    
    def build(self) -> NeuralNetwork:
        """
        Build and return the neural network.
        
        Returns:
            Configured NeuralNetwork instance
        """
        from .layers import DenseLayer, ActivationLayer, DropoutLayer
        
        network = NeuralNetwork()
        current_size = self._input_size
        
        for config in self._layers_config:
            if config['type'] == 'dense':
                network.add(DenseLayer(
                    input_size=current_size,
                    output_size=config['units'],
                    initializer=config['initializer']
                ))
                current_size = config['units']
                
                if config['activation']:
                    network.add(ActivationLayer(config['activation']))
            
            elif config['type'] == 'dropout':
                network.add(DropoutLayer(config['rate']))
        
        return network
