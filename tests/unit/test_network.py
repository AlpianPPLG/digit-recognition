"""
Unit Tests for Neural Network

Tests for src/core/network.py
"""

import pytest
import numpy as np
import sys
import tempfile
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from core.network import NeuralNetwork, NetworkBuilder
from core.layers import DenseLayer, ActivationLayer, DropoutLayer


class TestNeuralNetwork:
    """Tests for NeuralNetwork class."""
    
    def test_initialization(self):
        """Test network initialization."""
        network = NeuralNetwork()
        assert len(network.layers) == 0
        assert network.loss_function is None
        assert network.optimizer is None
    
    def test_add_layer(self):
        """Test adding layers."""
        network = NeuralNetwork()
        layer = DenseLayer(10, 5)
        
        result = network.add(layer)
        
        assert len(network.layers) == 1
        assert result is network  # Method chaining
    
    def test_add_multiple_layers(self):
        """Test adding multiple layers with chaining."""
        network = (NeuralNetwork()
                   .add(DenseLayer(784, 128))
                   .add(ActivationLayer('relu'))
                   .add(DenseLayer(128, 10))
                   .add(ActivationLayer('softmax')))
        
        assert len(network.layers) == 4
    
    def test_forward_pass(self):
        """Test forward pass through network."""
        np.random.seed(42)
        network = NeuralNetwork()
        network.add(DenseLayer(784, 128))
        network.add(ActivationLayer('relu'))
        network.add(DenseLayer(128, 10))
        network.add(ActivationLayer('softmax'))
        
        x = np.random.randn(32, 784)
        output = network.forward(x)
        
        assert output.shape == (32, 10)
        # Softmax output should sum to 1
        np.testing.assert_array_almost_equal(
            np.sum(output, axis=1),
            np.ones(32)
        )
    
    def test_backward_pass(self):
        """Test backward pass through network."""
        np.random.seed(42)
        network = NeuralNetwork()
        network.add(DenseLayer(10, 5))
        network.add(ActivationLayer('relu'))
        network.add(DenseLayer(5, 3))
        
        x = np.random.randn(4, 10)
        _ = network.forward(x)
        
        grad = np.random.randn(4, 3)
        grad_input = network.backward(grad)
        
        assert grad_input.shape == (4, 10)
    
    def test_predict(self):
        """Test prediction (evaluation mode)."""
        np.random.seed(42)
        network = NeuralNetwork()
        network.add(DenseLayer(10, 5))
        network.add(ActivationLayer('softmax'))
        
        x = np.random.randn(4, 10)
        output = network.predict(x)
        
        assert output.shape == (4, 5)
    
    def test_predict_classes(self):
        """Test class prediction."""
        np.random.seed(42)
        network = NeuralNetwork()
        network.add(DenseLayer(10, 5))
        network.add(ActivationLayer('softmax'))
        
        x = np.random.randn(4, 10)
        classes = network.predict_classes(x)
        
        assert classes.shape == (4,)
        assert all(0 <= c < 5 for c in classes)
    
    def test_get_parameters(self):
        """Test getting trainable parameters."""
        np.random.seed(42)
        network = NeuralNetwork()
        network.add(DenseLayer(10, 5))
        network.add(ActivationLayer('relu'))
        network.add(DenseLayer(5, 3))
        
        params = network.get_parameters()
        
        assert len(params) == 2  # Two dense layers
        assert 'weights' in params[0]
        assert 'bias' in params[0]
    
    def test_get_gradients(self):
        """Test getting gradients after backward pass."""
        np.random.seed(42)
        network = NeuralNetwork()
        network.add(DenseLayer(10, 5))
        network.add(DenseLayer(5, 3))
        
        x = np.random.randn(4, 10)
        _ = network.forward(x)
        network.backward(np.ones((4, 3)))
        
        grads = network.get_gradients()
        
        assert len(grads) == 2
        assert 'weights' in grads[0]
    
    def test_set_training_mode(self):
        """Test setting training mode."""
        network = NeuralNetwork()
        network.add(DropoutLayer(0.5))
        
        network.set_training(False)
        assert network._is_training == False
        assert network.layers[0].training == False
        
        network.set_training(True)
        assert network._is_training == True
        assert network.layers[0].training == True
    
    def test_summary(self):
        """Test network summary."""
        network = NeuralNetwork()
        network.add(DenseLayer(784, 128))
        network.add(ActivationLayer('relu'))
        network.add(DenseLayer(128, 10))
        
        summary = network.summary()
        
        assert "Neural Network Summary" in summary
        assert "DenseLayer" in summary
        assert "Total Parameters" in summary
    
    def test_save_and_load(self):
        """Test saving and loading weights."""
        np.random.seed(42)
        network = NeuralNetwork()
        network.add(DenseLayer(10, 5))
        network.add(DenseLayer(5, 3))
        
        # Store original weights
        original_weights = [
            network.layers[0].weights.copy(),
            network.layers[2].weights.copy() if len(network.layers) > 2 else network.layers[1].weights.copy()
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filepath = f.name
        
        try:
            # Save
            network.save(filepath)
            
            # Modify weights
            network.layers[0].weights = np.zeros_like(network.layers[0].weights)
            
            # Load
            network.load(filepath)
            
            # Verify weights restored
            np.testing.assert_array_almost_equal(
                network.layers[0].weights,
                original_weights[0]
            )
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_repr(self):
        """Test string representation."""
        network = NeuralNetwork()
        network.add(DenseLayer(10, 5))
        network.add(DenseLayer(5, 3))
        
        assert "NeuralNetwork(layers=2)" == repr(network)


class TestNetworkBuilder:
    """Tests for NetworkBuilder class."""
    
    def test_simple_network(self):
        """Test building a simple network."""
        network = (NetworkBuilder()
                   .input(784)
                   .dense(128, activation='relu')
                   .dense(10, activation='softmax')
                   .build())
        
        assert len(network.layers) == 4  # 2 dense + 2 activation
    
    def test_network_shapes(self):
        """Test that built network has correct shapes."""
        np.random.seed(42)
        network = (NetworkBuilder()
                   .input(784)
                   .dense(128, activation='relu')
                   .dense(64, activation='relu')
                   .dense(10, activation='softmax')
                   .build())
        
        x = np.random.randn(16, 784)
        output = network.forward(x)
        
        assert output.shape == (16, 10)
    
    def test_with_dropout(self):
        """Test building network with dropout."""
        network = (NetworkBuilder()
                   .input(100)
                   .dense(50, activation='relu')
                   .dropout(0.5)
                   .dense(10, activation='softmax')
                   .build())
        
        # Should have: dense, relu, dropout, dense, softmax
        assert len(network.layers) == 5
    
    def test_custom_initializer(self):
        """Test building network with custom initializer."""
        np.random.seed(42)
        network = (NetworkBuilder()
                   .input(100)
                   .dense(50, activation='relu', initializer='xavier')
                   .dense(10, activation='softmax', initializer='he')
                   .build())
        
        # Verify layers were created with correct shapes
        # First dense layer at index 0
        assert network.layers[0].weights.shape == (100, 50)
        # Second dense layer at index 2 (after ReLU activation)
        assert network.layers[2].weights.shape == (50, 10)
    
    def test_no_activation(self):
        """Test dense layer without activation."""
        network = (NetworkBuilder()
                   .input(10)
                   .dense(5)  # No activation
                   .build())
        
        # Should only have 1 dense layer, no activation
        assert len(network.layers) == 1
        assert isinstance(network.layers[0], DenseLayer)
    
    def test_method_chaining(self):
        """Test that all methods support chaining."""
        builder = NetworkBuilder()
        
        result = builder.input(100)
        assert result is builder
        
        result = builder.dense(50)
        assert result is builder
        
        result = builder.dropout(0.5)
        assert result is builder


class TestNetworkIntegration:
    """Integration tests for neural network."""
    
    def test_digit_recognition_architecture(self):
        """Test the full digit recognition architecture."""
        np.random.seed(42)
        
        # Build the target architecture: 784 -> 128 -> 64 -> 10
        network = (NetworkBuilder()
                   .input(784)
                   .dense(128, activation='relu')
                   .dense(64, activation='relu')
                   .dense(10, activation='softmax')
                   .build())
        
        # Test forward pass
        batch = np.random.randn(32, 784)
        output = network.forward(batch)
        
        assert output.shape == (32, 10)
        
        # Test backward pass
        gradient = output.copy()  # Dummy gradient
        grad_input = network.backward(gradient)
        
        assert grad_input.shape == (32, 784)
    
    def test_training_step(self):
        """Test a single training step."""
        np.random.seed(42)
        
        network = (NetworkBuilder()
                   .input(10)
                   .dense(5, activation='relu')
                   .dense(3, activation='softmax')
                   .build())
        
        # Sample data
        x = np.random.randn(4, 10)
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
        
        # Forward pass
        y_pred = network.forward(x)
        
        # Simple cross-entropy gradient (softmax + cross-entropy)
        gradient = y_pred - y_true
        
        # Backward pass
        network.backward(gradient)
        
        # Get gradients
        grads = network.get_gradients()
        params = network.get_parameters()
        
        # Verify gradients were computed
        assert len(grads) == len(params)
        for g in grads:
            assert g['weights'] is not None
            assert g['bias'] is not None
    
    def test_parameter_count(self):
        """Test total parameter count for target architecture."""
        network = (NetworkBuilder()
                   .input(784)
                   .dense(128, activation='relu')
                   .dense(64, activation='relu')
                   .dense(10, activation='softmax')
                   .build())
        
        # Calculate expected parameters
        # Layer 1: 784 * 128 + 128 = 100,480
        # Layer 2: 128 * 64 + 64 = 8,256
        # Layer 3: 64 * 10 + 10 = 650
        # Total: 109,386
        
        total = 0
        for layer in network.layers:
            if hasattr(layer, 'weights'):
                total += layer.weights.size
            if hasattr(layer, 'bias') and layer.bias is not None:
                total += layer.bias.size
        
        assert total == 109386  # Target parameter count from PRD


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
