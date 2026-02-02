"""
Unit Tests for Loss Functions

Tests for src/core/losses.py
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from core.losses import CrossEntropyLoss, MSELoss


class TestCrossEntropyLoss:
    """Tests for Cross-Entropy Loss."""
    
    @pytest.fixture
    def loss_fn(self):
        """Create loss function instance."""
        return CrossEntropyLoss()
    
    def test_perfect_prediction(self, loss_fn):
        """Test loss with near-perfect predictions."""
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[0.999, 0.0005, 0.0005], 
                          [0.0005, 0.999, 0.0005], 
                          [0.0005, 0.0005, 0.999]])
        
        loss = loss_fn.forward(y_pred, y_true)
        
        # Loss should be very small for near-perfect predictions
        assert loss < 0.01
    
    def test_worst_prediction(self, loss_fn):
        """Test loss with completely wrong predictions."""
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_pred = np.array([[0.01, 0.495, 0.495], 
                          [0.495, 0.01, 0.495]])
        
        loss = loss_fn.forward(y_pred, y_true)
        
        # Loss should be large
        assert loss > 2.0
    
    def test_numerical_stability(self, loss_fn):
        """Test that loss handles edge cases without NaN."""
        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[1.0, 0.0, 0.0]])  # Edge case: exact 0 and 1
        
        loss = loss_fn.forward(y_pred, y_true)
        
        assert not np.isnan(loss)
        assert not np.isinf(loss)
    
    def test_backward_shape(self, loss_fn):
        """Test backward gradient shape."""
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        
        grad = loss_fn.backward(y_pred, y_true)
        
        assert grad.shape == y_true.shape
    
    def test_backward_direction(self, loss_fn):
        """Test gradient points in correct direction."""
        # Perfect prediction
        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0.9, 0.05, 0.05]])
        
        grad = loss_fn.backward(y_pred, y_true)
        
        # Gradient for softmax + cross-entropy is (y_pred - y_true) / batch_size
        # At true class: should be negative (0.9 - 1 = -0.1)
        assert grad[0, 0] < 0
        # At other classes: should be positive
        assert grad[0, 1] > 0
        assert grad[0, 2] > 0
    
    def test_callable(self, loss_fn):
        """Test loss can be called as function."""
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[0.9, 0.1], [0.1, 0.9]])
        
        loss = loss_fn(y_pred, y_true)
        
        # Should return a scalar
        assert np.isscalar(loss) or loss.ndim == 0


class TestMSELoss:
    """Tests for Mean Squared Error Loss."""
    
    @pytest.fixture
    def loss_fn(self):
        """Create loss function instance."""
        return MSELoss()
    
    def test_perfect_prediction(self, loss_fn):
        """Test MSE with perfect predictions."""
        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[1, 0, 0]])
        
        loss = loss_fn.forward(y_pred, y_true)
        
        assert loss == 0.0
    
    def test_known_loss(self, loss_fn):
        """Test MSE with known values."""
        y_true = np.array([[0, 0]])
        y_pred = np.array([[1, 1]])
        
        loss = loss_fn.forward(y_pred, y_true)
        
        # MSE = mean((1-0)^2 + (1-0)^2) = mean(1 + 1) = 1.0
        assert loss == 1.0
    
    def test_backward_shape(self, loss_fn):
        """Test backward gradient shape."""
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_pred = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2]])
        
        grad = loss_fn.backward(y_pred, y_true)
        
        assert grad.shape == y_true.shape
    
    def test_backward_values(self, loss_fn):
        """Test backward gradient computation."""
        y_true = np.array([[1.0]])
        y_pred = np.array([[0.0]])
        
        grad = loss_fn.backward(y_pred, y_true)
        
        # Gradient of MSE: 2 * (y_pred - y_true) / n
        # = 2 * (0 - 1) / 1 = -2
        np.testing.assert_almost_equal(grad[0, 0], -2.0)
    
    def test_callable(self, loss_fn):
        """Test MSE can be called as function."""
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[0.9, 0.1], [0.1, 0.9]])
        
        loss = loss_fn(y_pred, y_true)
        
        # MSE = mean((0.9-1)^2 + (0.1-0)^2 + (0.1-0)^2 + (0.9-1)^2)
        # = mean(0.01 + 0.01 + 0.01 + 0.01) = 0.01
        np.testing.assert_almost_equal(loss, 0.01)


class TestLossComparison:
    """Compare behavior of different loss functions."""
    
    def test_both_zero_on_perfect(self):
        """Both losses should be (near) zero on perfect predictions."""
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        ce_loss = CrossEntropyLoss()
        mse_loss = MSELoss()
        
        ce_value = ce_loss.forward(y_true, y_true)
        mse_value = mse_loss.forward(y_true, y_true)
        
        assert mse_value == 0.0
        # CE will use clipping, so check it's very small
        assert ce_value < 0.001
    
    def test_gradients_same_direction(self):
        """Both should push predictions toward true values."""
        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0.2, 0.4, 0.4]])  # Wrong prediction
        
        ce_loss = CrossEntropyLoss()
        mse_loss = MSELoss()
        
        ce_grad = ce_loss.backward(y_pred, y_true)
        mse_grad = mse_loss.backward(y_pred, y_true)
        
        # Both should have negative gradient at true class
        # (to increase probability there)
        assert ce_grad[0, 0] < 0
        assert mse_grad[0, 0] < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
