"""
Unit Tests for Neural Network Layers

Tests for src/core/layers.py
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from core.layers import Layer, DenseLayer, ActivationLayer, DropoutLayer


class TestDenseLayer:
    """Tests for Dense/Fully Connected Layer."""
    
    @pytest.fixture
    def layer(self):
        """Create a standard dense layer for testing."""
        np.random.seed(42)
        return DenseLayer(input_size=4, output_size=3)
    
    def test_initialization_shapes(self, layer):
        """Test that weights and bias have correct shapes."""
        assert layer.weights.shape == (4, 3)
        assert layer.bias.shape == (1, 3)
    
    def test_forward_pass_shape(self, layer):
        """Test forward pass output shape."""
        x = np.random.randn(10, 4)  # batch of 10, 4 features
        output = layer.forward(x)
        assert output.shape == (10, 3)
    
    def test_forward_pass_computation(self):
        """Test forward pass mathematical correctness."""
        layer = DenseLayer(2, 2)
        layer.weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        layer.bias = np.array([[0.5, 0.5]])
        
        x = np.array([[1.0, 1.0]])
        output = layer.forward(x)
        
        # Expected: x @ W + b = [1,1] @ [[1,2],[3,4]] + [0.5, 0.5]
        #         = [4, 6] + [0.5, 0.5] = [4.5, 6.5]
        expected = np.array([[4.5, 6.5]])
        np.testing.assert_array_almost_equal(output, expected)
    
    def test_backward_pass_shape(self, layer):
        """Test backward pass gradient shapes."""
        x = np.random.randn(10, 4)
        layer.forward(x)
        
        grad = np.random.randn(10, 3)
        grad_input = layer.backward(grad)
        
        assert grad_input.shape == (10, 4)
        assert layer.dW.shape == (4, 3)
        assert layer.db.shape == (1, 3)
    
    def test_backward_pass_computation(self):
        """Test backward pass gradients are computed correctly."""
        np.random.seed(42)
        layer = DenseLayer(3, 2)
        
        x = np.array([[1.0, 2.0, 3.0]])
        layer.forward(x)
        
        grad = np.array([[1.0, 1.0]])
        grad_input = layer.backward(grad)
        
        # grad_input should have shape (1, 3)
        assert grad_input.shape == (1, 3)
        
        # dW should be x.T @ grad / batch_size
        expected_dW = x.T @ grad / 1
        np.testing.assert_array_almost_equal(layer.dW, expected_dW)
    
    def test_no_bias(self):
        """Test layer without bias."""
        layer = DenseLayer(4, 3, use_bias=False)
        assert layer.bias is None
        
        x = np.random.randn(5, 4)
        output = layer.forward(x)
        assert output.shape == (5, 3)
    
    def test_he_initialization(self):
        """Test He initialization produces reasonable scale."""
        np.random.seed(42)
        layer = DenseLayer(784, 128, initializer='he')
        
        # He init std = sqrt(2/fan_in) = sqrt(2/784) ≈ 0.05
        expected_std = np.sqrt(2.0 / 784)
        actual_std = np.std(layer.weights)
        
        # Should be close (within 20%)
        assert abs(actual_std - expected_std) / expected_std < 0.2
    
    def test_xavier_initialization(self):
        """Test Xavier initialization produces reasonable scale."""
        np.random.seed(42)
        layer = DenseLayer(784, 128, initializer='xavier')
        
        # Xavier init std = sqrt(2/(fan_in + fan_out))
        expected_std = np.sqrt(2.0 / (784 + 128))
        actual_std = np.std(layer.weights)
        
        # Should be close (within 20%)
        assert abs(actual_std - expected_std) / expected_std < 0.2
    
    def test_parameters_property(self, layer):
        """Test parameters property returns weights and bias."""
        params = layer.parameters
        assert 'weights' in params
        assert 'bias' in params
        np.testing.assert_array_equal(params['weights'], layer.weights)
        np.testing.assert_array_equal(params['bias'], layer.bias)
    
    def test_gradients_property(self, layer):
        """Test gradients property after backward pass."""
        x = np.random.randn(5, 4)
        layer.forward(x)
        layer.backward(np.random.randn(5, 3))
        
        grads = layer.gradients
        assert 'weights' in grads
        assert 'bias' in grads
    
    def test_repr(self, layer):
        """Test string representation."""
        assert "DenseLayer(4, 3)" in repr(layer)


class TestActivationLayer:
    """Tests for Activation Layer."""
    
    def test_relu_forward(self):
        """Test ReLU activation forward pass."""
        layer = ActivationLayer('relu')
        x = np.array([[-2, -1, 0, 1, 2]])
        output = layer.forward(x)
        expected = np.array([[0, 0, 0, 1, 2]])
        np.testing.assert_array_equal(output, expected)
    
    def test_relu_backward(self):
        """Test ReLU activation backward pass."""
        layer = ActivationLayer('relu')
        x = np.array([[-2, -1, 0, 1, 2]], dtype=np.float64)
        layer.forward(x)
        
        grad = np.ones_like(x)
        grad_input = layer.backward(grad)
        expected = np.array([[0, 0, 0, 1, 1]], dtype=np.float64)
        np.testing.assert_array_equal(grad_input, expected)
    
    def test_leaky_relu_forward(self):
        """Test Leaky ReLU activation."""
        layer = ActivationLayer('leaky_relu', alpha=0.1)
        x = np.array([[-2, -1, 0, 1, 2]], dtype=np.float64)
        output = layer.forward(x)
        expected = np.array([[-0.2, -0.1, 0, 1, 2]])
        np.testing.assert_array_almost_equal(output, expected)
    
    def test_sigmoid_forward(self):
        """Test sigmoid activation."""
        layer = ActivationLayer('sigmoid')
        x = np.array([[0]])
        output = layer.forward(x)
        np.testing.assert_almost_equal(output[0, 0], 0.5)
    
    def test_sigmoid_backward(self):
        """Test sigmoid backward pass."""
        layer = ActivationLayer('sigmoid')
        x = np.array([[0]], dtype=np.float64)
        layer.forward(x)
        
        grad = np.ones((1, 1))
        grad_input = layer.backward(grad)
        
        # At x=0, sigmoid = 0.5, derivative = 0.5 * 0.5 = 0.25
        np.testing.assert_almost_equal(grad_input[0, 0], 0.25)
    
    def test_tanh_forward(self):
        """Test tanh activation."""
        layer = ActivationLayer('tanh')
        x = np.array([[0]])
        output = layer.forward(x)
        np.testing.assert_almost_equal(output[0, 0], 0)
    
    def test_softmax_forward(self):
        """Test softmax activation."""
        layer = ActivationLayer('softmax')
        x = np.array([[1, 2, 3]])
        output = layer.forward(x)
        
        # Should sum to 1
        np.testing.assert_almost_equal(np.sum(output), 1.0)
        
        # All values should be positive
        assert np.all(output > 0)
    
    def test_softmax_batch(self):
        """Test softmax with batch input."""
        layer = ActivationLayer('softmax')
        x = np.array([[1, 2, 3], [1, 2, 3]])
        output = layer.forward(x)
        
        # Each row should sum to 1
        np.testing.assert_array_almost_equal(
            np.sum(output, axis=1), 
            [1.0, 1.0]
        )
    
    def test_unknown_activation(self):
        """Test that unknown activation raises error."""
        layer = ActivationLayer('unknown')
        with pytest.raises(ValueError, match="Unknown activation"):
            layer.forward(np.array([[1, 2, 3]]))
    
    def test_repr(self):
        """Test string representation."""
        layer = ActivationLayer('relu')
        assert "ActivationLayer('relu')" in repr(layer)


class TestDropoutLayer:
    """Tests for Dropout Layer."""
    
    def test_initialization(self):
        """Test dropout initialization."""
        layer = DropoutLayer(rate=0.5)
        assert layer.rate == 0.5
        assert layer.training == True
    
    def test_invalid_rate(self):
        """Test that invalid rate raises error."""
        with pytest.raises(ValueError):
            DropoutLayer(rate=1.0)
        with pytest.raises(ValueError):
            DropoutLayer(rate=-0.1)
    
    def test_training_mode(self):
        """Test dropout in training mode."""
        np.random.seed(42)
        layer = DropoutLayer(rate=0.5)
        layer.training = True
        
        x = np.ones((100, 100))
        output = layer.forward(x)
        
        # Approximately 50% should be zero
        zero_fraction = np.mean(output == 0)
        assert 0.4 < zero_fraction < 0.6
    
    def test_inference_mode(self):
        """Test dropout in inference mode (no dropout applied)."""
        layer = DropoutLayer(rate=0.5)
        layer.training = False
        
        x = np.ones((10, 10))
        output = layer.forward(x)
        
        # All values should be unchanged
        np.testing.assert_array_equal(output, x)
    
    def test_scaling(self):
        """Test inverted dropout scaling."""
        np.random.seed(42)
        layer = DropoutLayer(rate=0.5)
        layer.training = True
        
        x = np.ones((1000, 100))
        output = layer.forward(x)
        
        # Mean should be approximately 1 due to scaling
        assert 0.9 < np.mean(output) < 1.1
    
    def test_backward_pass(self):
        """Test backward pass through dropout."""
        np.random.seed(42)
        layer = DropoutLayer(rate=0.5)
        layer.training = True
        
        x = np.ones((10, 10))
        layer.forward(x)
        
        grad = np.ones((10, 10))
        grad_output = layer.backward(grad)
        
        # Gradient should have same pattern as forward (masked & scaled)
        assert grad_output.shape == grad.shape
    
    def test_repr(self):
        """Test string representation."""
        layer = DropoutLayer(rate=0.3)
        assert "DropoutLayer(rate=0.3)" in repr(layer)


class TestLayerIntegration:
    """Integration tests combining multiple layers."""
    
    def test_dense_relu_chain(self):
        """Test chaining Dense -> ReLU layers."""
        np.random.seed(42)
        
        dense = DenseLayer(784, 128)
        relu = ActivationLayer('relu')
        
        x = np.random.randn(32, 784)
        
        # Forward pass
        z = dense.forward(x)
        a = relu.forward(z)
        
        assert a.shape == (32, 128)
        assert np.all(a >= 0)  # ReLU output is non-negative
    
    def test_backward_chain(self):
        """Test backward pass through layer chain."""
        np.random.seed(42)
        
        dense = DenseLayer(10, 5)
        relu = ActivationLayer('relu')
        
        x = np.random.randn(4, 10)
        
        # Forward
        z = dense.forward(x)
        a = relu.forward(z)
        
        # Backward
        grad = np.ones((4, 5))
        grad = relu.backward(grad)
        grad = dense.backward(grad)
        
        assert grad.shape == (4, 10)
        assert dense.dW is not None
        assert dense.db is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
