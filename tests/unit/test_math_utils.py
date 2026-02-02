"""
Unit Tests for Mathematical Utilities

Tests for src/utils/math_utils.py
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from utils.math_utils import (
    matmul, clip, stable_softmax, stable_log, stable_cross_entropy,
    sigmoid, sigmoid_derivative, relu, relu_derivative,
    leaky_relu, leaky_relu_derivative,
    one_hot_encode, one_hot_decode, normalize, standardize,
    batch_iterator, accuracy_score
)


class TestMatmul:
    """Tests for matrix multiplication function."""
    
    def test_basic_multiplication(self):
        """Test basic 2x2 matrix multiplication."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = matmul(a, b)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(result, expected)
    
    def test_different_shapes(self):
        """Test multiplication with different shapes."""
        a = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
        b = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2
        result = matmul(a, b)
        assert result.shape == (2, 2)
    
    def test_batch_multiplication(self):
        """Test with batch input."""
        a = np.random.randn(10, 784)  # batch of 10, 784 features
        b = np.random.randn(784, 128)  # weight matrix
        result = matmul(a, b)
        assert result.shape == (10, 128)
    
    def test_incompatible_shapes(self):
        """Test that incompatible shapes raise error."""
        a = np.array([[1, 2], [3, 4]])  # 2x2
        b = np.array([[1, 2, 3]])  # 1x3
        with pytest.raises(ValueError, match="Incompatible shapes"):
            matmul(a, b)


class TestClip:
    """Tests for clip function."""
    
    def test_clip_default(self):
        """Test clipping with default values."""
        x = np.array([0, 0.5, 1])
        result = clip(x)
        assert result[0] > 0  # Not exactly 0
        assert result[2] < 1  # Not exactly 1
        assert result[1] == 0.5  # Unchanged
    
    def test_clip_custom_range(self):
        """Test clipping with custom range."""
        x = np.array([-5, 0, 5, 10])
        result = clip(x, min_val=0, max_val=5)
        np.testing.assert_array_equal(result, [0, 0, 5, 5])


class TestStableSoftmax:
    """Tests for numerically stable softmax."""
    
    def test_basic_softmax(self):
        """Test softmax output properties."""
        x = np.array([[1, 2, 3]])
        result = stable_softmax(x)
        
        # Should sum to 1
        np.testing.assert_almost_equal(np.sum(result), 1.0)
        
        # All values should be positive
        assert np.all(result > 0)
        
        # Largest input should have largest probability
        assert np.argmax(result) == 2
    
    def test_softmax_batch(self):
        """Test softmax with batch input."""
        x = np.array([[1, 2, 3], [1, 2, 3]])
        result = stable_softmax(x, axis=1)
        
        # Each row should sum to 1
        np.testing.assert_array_almost_equal(np.sum(result, axis=1), [1.0, 1.0])
    
    def test_numerical_stability(self):
        """Test that large values don't cause overflow."""
        x = np.array([[1000, 1001, 1002]])
        result = stable_softmax(x)
        
        # Should not contain inf or nan
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        
        # Should still sum to 1
        np.testing.assert_almost_equal(np.sum(result), 1.0)


class TestStableLog:
    """Tests for numerically stable logarithm."""
    
    def test_basic_log(self):
        """Test basic logarithm."""
        x = np.array([1, np.e, np.e**2])
        result = stable_log(x)
        expected = np.array([0, 1, 2])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
    
    def test_zero_input(self):
        """Test that zero input doesn't cause -inf."""
        x = np.array([0])
        result = stable_log(x)
        assert not np.isinf(result[0])


class TestSigmoid:
    """Tests for sigmoid function."""
    
    def test_basic_sigmoid(self):
        """Test sigmoid at key points."""
        # sigmoid(0) = 0.5
        np.testing.assert_almost_equal(sigmoid(np.array([0]))[0], 0.5)
        
        # sigmoid should be symmetric: sigmoid(-x) = 1 - sigmoid(x)
        x = np.array([2])
        np.testing.assert_almost_equal(sigmoid(-x)[0], 1 - sigmoid(x)[0])
    
    def test_sigmoid_range(self):
        """Test sigmoid output is in (0, 1)."""
        x = np.linspace(-10, 10, 100)
        result = sigmoid(x)
        assert np.all(result > 0)
        assert np.all(result < 1)
    
    def test_sigmoid_numerical_stability(self):
        """Test sigmoid with large values."""
        x = np.array([-1000, 1000])
        result = sigmoid(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestSigmoidDerivative:
    """Tests for sigmoid derivative."""
    
    def test_derivative_at_zero(self):
        """Derivative at x=0 should be 0.25."""
        result = sigmoid_derivative(np.array([0]))
        np.testing.assert_almost_equal(result[0], 0.25)
    
    def test_derivative_is_positive(self):
        """Derivative should always be positive."""
        x = np.linspace(-10, 10, 100)
        result = sigmoid_derivative(x)
        assert np.all(result > 0)


class TestReLU:
    """Tests for ReLU function."""
    
    def test_basic_relu(self):
        """Test ReLU at various points."""
        x = np.array([-2, -1, 0, 1, 2])
        expected = np.array([0, 0, 0, 1, 2])
        np.testing.assert_array_equal(relu(x), expected)
    
    def test_relu_preserves_positive(self):
        """Test that positive values are unchanged."""
        x = np.array([1.5, 2.5, 3.5])
        np.testing.assert_array_equal(relu(x), x)


class TestReLUDerivative:
    """Tests for ReLU derivative."""
    
    def test_basic_derivative(self):
        """Test ReLU derivative."""
        x = np.array([-2, -1, 0, 1, 2])
        expected = np.array([0, 0, 0, 1, 1])
        np.testing.assert_array_equal(relu_derivative(x), expected)


class TestLeakyReLU:
    """Tests for Leaky ReLU function."""
    
    def test_basic_leaky_relu(self):
        """Test Leaky ReLU."""
        x = np.array([-2, -1, 0, 1, 2])
        result = leaky_relu(x, alpha=0.1)
        expected = np.array([-0.2, -0.1, 0, 1, 2])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_negative_slope(self):
        """Test that negative values have small slope."""
        x = np.array([-10])
        result = leaky_relu(x, alpha=0.01)
        assert result[0] == -0.1


class TestOneHotEncode:
    """Tests for one-hot encoding."""
    
    def test_basic_encoding(self):
        """Test basic one-hot encoding."""
        labels = np.array([0, 2, 1])
        result = one_hot_encode(labels, num_classes=3)
        expected = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])
        np.testing.assert_array_equal(result, expected)
    
    def test_encoding_shape(self):
        """Test output shape."""
        labels = np.array([0, 1, 2, 3, 4])
        result = one_hot_encode(labels, num_classes=10)
        assert result.shape == (5, 10)


class TestOneHotDecode:
    """Tests for one-hot decoding."""
    
    def test_basic_decoding(self):
        """Test basic one-hot decoding."""
        one_hot = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])
        result = one_hot_decode(one_hot)
        expected = np.array([0, 2, 1])
        np.testing.assert_array_equal(result, expected)
    
    def test_encode_decode_roundtrip(self):
        """Test that encoding then decoding gives original."""
        labels = np.array([0, 5, 3, 9, 1])
        encoded = one_hot_encode(labels, num_classes=10)
        decoded = one_hot_decode(encoded)
        np.testing.assert_array_equal(decoded, labels)


class TestNormalize:
    """Tests for normalization."""
    
    def test_basic_normalize(self):
        """Test basic normalization."""
        x = np.array([0, 50, 100])
        result = normalize(x)
        expected = np.array([0, 0.5, 1])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_normalize_range(self):
        """Test that output is in [0, 1]."""
        x = np.random.randn(100)
        result = normalize(x)
        assert np.min(result) >= 0
        assert np.max(result) <= 1


class TestStandardize:
    """Tests for standardization."""
    
    def test_basic_standardize(self):
        """Test basic standardization."""
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        result = standardize(x)
        
        # Mean should be ~0
        np.testing.assert_almost_equal(np.mean(result), 0, decimal=5)
        
        # Std should be ~1
        np.testing.assert_almost_equal(np.std(result), 1, decimal=5)


class TestBatchIterator:
    """Tests for batch iterator."""
    
    def test_basic_iteration(self):
        """Test basic batch iteration."""
        X = np.arange(10).reshape(10, 1)
        y = np.arange(10)
        
        batches = list(batch_iterator(X, y, batch_size=3, shuffle=False))
        
        assert len(batches) == 4  # 10 / 3 = 3.33, so 4 batches
        assert batches[0][0].shape == (3, 1)
        assert batches[-1][0].shape == (1, 1)  # Last batch smaller
    
    def test_batch_size_equals_data(self):
        """Test when batch size equals data size."""
        X = np.arange(5).reshape(5, 1)
        y = np.arange(5)
        
        batches = list(batch_iterator(X, y, batch_size=5, shuffle=False))
        
        assert len(batches) == 1
        assert batches[0][0].shape == (5, 1)


class TestAccuracyScore:
    """Tests for accuracy calculation."""
    
    def test_perfect_accuracy(self):
        """Test 100% accuracy."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        assert accuracy_score(y_true, y_pred) == 1.0
    
    def test_zero_accuracy(self):
        """Test 0% accuracy."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1])
        assert accuracy_score(y_true, y_pred) == 0.0
    
    def test_partial_accuracy(self):
        """Test partial accuracy."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 0, 0, 0])  # 2 correct out of 5
        assert accuracy_score(y_true, y_pred) == 0.4
    
    def test_with_one_hot(self):
        """Test accuracy with one-hot encoded inputs."""
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        assert accuracy_score(y_true, y_pred) == 1.0


class TestCrossEntropy:
    """Tests for cross-entropy loss."""
    
    def test_perfect_prediction(self):
        """Test cross-entropy with perfect prediction."""
        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0.99, 0.005, 0.005]])
        result = stable_cross_entropy(y_true, y_pred)
        assert result[0] < 0.1  # Should be very small
    
    def test_bad_prediction(self):
        """Test cross-entropy with bad prediction."""
        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0.01, 0.49, 0.5]])
        result = stable_cross_entropy(y_true, y_pred)
        assert result[0] > 2  # Should be large


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
