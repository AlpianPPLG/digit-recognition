"""
Unit Tests for Optimizers

Tests for src/core/optimizers.py
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from core.optimizers import (
    Optimizer, SGD, SGDMomentum, Adam, RMSprop, get_optimizer
)


class TestSGD:
    """Tests for SGD optimizer."""
    
    def test_initialization(self):
        """Test SGD initialization."""
        opt = SGD(learning_rate=0.01)
        assert opt.lr == 0.01
    
    def test_invalid_lr(self):
        """Test that negative learning rate raises error."""
        with pytest.raises(ValueError):
            SGD(learning_rate=-0.01)
        with pytest.raises(ValueError):
            SGD(learning_rate=0)
    
    def test_update(self):
        """Test parameter update."""
        opt = SGD(learning_rate=0.1)
        params = {'weights': np.array([1.0, 2.0, 3.0])}
        grads = {'weights': np.array([0.1, 0.2, 0.3])}
        
        opt.update(params, grads)
        
        # Expected: w = w - lr * grad = [1, 2, 3] - 0.1 * [0.1, 0.2, 0.3]
        expected = np.array([0.99, 1.98, 2.97])
        np.testing.assert_array_almost_equal(params['weights'], expected)
    
    def test_update_with_bias(self):
        """Test update with multiple parameters."""
        opt = SGD(learning_rate=0.1)
        params = {
            'weights': np.array([[1.0, 2.0], [3.0, 4.0]]),
            'bias': np.array([0.5, 0.5])
        }
        grads = {
            'weights': np.array([[0.1, 0.1], [0.1, 0.1]]),
            'bias': np.array([0.1, 0.1])
        }
        
        opt.update(params, grads)
        
        expected_weights = np.array([[0.99, 1.99], [2.99, 3.99]])
        expected_bias = np.array([0.49, 0.49])
        
        np.testing.assert_array_almost_equal(params['weights'], expected_weights)
        np.testing.assert_array_almost_equal(params['bias'], expected_bias)
    
    def test_skip_none_gradients(self):
        """Test that None gradients are skipped."""
        opt = SGD(learning_rate=0.1)
        params = {'weights': np.array([1.0, 2.0])}
        grads = {'weights': None}
        
        opt.update(params, grads)
        
        # Should be unchanged
        np.testing.assert_array_equal(params['weights'], np.array([1.0, 2.0]))
    
    def test_repr(self):
        """Test string representation."""
        opt = SGD(learning_rate=0.01)
        assert "SGD(lr=0.01)" == repr(opt)


class TestSGDMomentum:
    """Tests for SGD with Momentum optimizer."""
    
    def test_initialization(self):
        """Test initialization."""
        opt = SGDMomentum(learning_rate=0.01, momentum=0.9)
        assert opt.lr == 0.01
        assert opt.momentum == 0.9
    
    def test_invalid_momentum(self):
        """Test invalid momentum values."""
        with pytest.raises(ValueError):
            SGDMomentum(momentum=1.0)  # Too high
        with pytest.raises(ValueError):
            SGDMomentum(momentum=-0.1)  # Negative
    
    def test_momentum_accumulation(self):
        """Test that momentum accumulates velocity."""
        opt = SGDMomentum(learning_rate=0.1, momentum=0.9)
        params = {'weights': np.array([1.0])}
        grads = {'weights': np.array([1.0])}
        
        # First update: v = 0 * 0.9 - 0.1 * 1.0 = -0.1, w = 1.0 + (-0.1) = 0.9
        opt.update(params, grads, layer_id=0)
        np.testing.assert_almost_equal(params['weights'][0], 0.9)
        
        # Second update: v = -0.1 * 0.9 - 0.1 * 1.0 = -0.19, w = 0.9 + (-0.19) = 0.71
        params_fresh = {'weights': np.array([0.9])}
        opt.update(params_fresh, grads, layer_id=0)
        np.testing.assert_almost_equal(params_fresh['weights'][0], 0.71)
    
    def test_different_layers(self):
        """Test that each layer has separate velocity."""
        opt = SGDMomentum(learning_rate=0.1, momentum=0.9)
        
        params1 = {'weights': np.array([1.0])}
        params2 = {'weights': np.array([2.0])}
        grads1 = {'weights': np.array([1.0])}
        grads2 = {'weights': np.array([0.5])}
        
        opt.update(params1, grads1, layer_id=0)
        opt.update(params2, grads2, layer_id=1)
        
        # Layer 0: v = -0.1, w = 0.9
        np.testing.assert_almost_equal(params1['weights'][0], 0.9)
        # Layer 1: v = -0.05, w = 1.95
        np.testing.assert_almost_equal(params2['weights'][0], 1.95)
    
    def test_reset(self):
        """Test velocity reset."""
        opt = SGDMomentum()
        params = {'weights': np.array([1.0])}
        grads = {'weights': np.array([1.0])}
        
        opt.update(params, grads, layer_id=0)
        assert len(opt.velocities) > 0
        
        opt.reset()
        assert len(opt.velocities) == 0
    
    def test_repr(self):
        """Test string representation."""
        opt = SGDMomentum(learning_rate=0.01, momentum=0.9)
        assert "SGDMomentum(lr=0.01, momentum=0.9)" == repr(opt)


class TestAdam:
    """Tests for Adam optimizer."""
    
    def test_initialization(self):
        """Test Adam initialization."""
        opt = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
        assert opt.lr == 0.001
        assert opt.beta1 == 0.9
        assert opt.beta2 == 0.999
        assert opt.t == 0
    
    def test_invalid_betas(self):
        """Test invalid beta values."""
        with pytest.raises(ValueError):
            Adam(beta1=1.0)
        with pytest.raises(ValueError):
            Adam(beta2=1.0)
        with pytest.raises(ValueError):
            Adam(beta1=-0.1)
    
    def test_time_step_increment(self):
        """Test that time step increments on each update."""
        opt = Adam()
        params = {'weights': np.array([1.0])}
        grads = {'weights': np.array([0.1])}
        
        opt.update(params, grads, layer_id=0)
        assert opt.t == 1
        
        opt.update(params, grads, layer_id=0)
        assert opt.t == 2
    
    def test_convergence(self):
        """Test that Adam moves parameters toward optimum."""
        np.random.seed(42)
        opt = Adam(learning_rate=0.1)
        
        # Simple quadratic: min at x=0
        x = np.array([5.0])
        params = {'weights': x.copy()}
        
        for _ in range(100):
            # Gradient of x^2 is 2x
            grads = {'weights': 2 * params['weights']}
            opt.update(params, grads, layer_id=0)
        
        # Should be close to 0
        assert abs(params['weights'][0]) < 0.5
    
    def test_reset(self):
        """Test moment estimate reset."""
        opt = Adam()
        params = {'weights': np.array([1.0])}
        grads = {'weights': np.array([0.1])}
        
        opt.update(params, grads, layer_id=0)
        assert opt.t == 1
        assert len(opt.m) > 0
        assert len(opt.v) > 0
        
        opt.reset()
        assert opt.t == 0
        assert len(opt.m) == 0
        assert len(opt.v) == 0
    
    def test_repr(self):
        """Test string representation."""
        opt = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
        assert "Adam(lr=0.001, beta1=0.9, beta2=0.999)" == repr(opt)


class TestRMSprop:
    """Tests for RMSprop optimizer."""
    
    def test_initialization(self):
        """Test RMSprop initialization."""
        opt = RMSprop(learning_rate=0.001, decay=0.9)
        assert opt.lr == 0.001
        assert opt.decay == 0.9
    
    def test_invalid_decay(self):
        """Test invalid decay values."""
        with pytest.raises(ValueError):
            RMSprop(decay=1.0)
        with pytest.raises(ValueError):
            RMSprop(decay=-0.1)
    
    def test_adaptive_learning_rate(self):
        """Test that RMSprop adapts learning rate."""
        opt = RMSprop(learning_rate=0.1, decay=0.9)
        
        # Parameter with large gradients
        params1 = {'weights': np.array([1.0])}
        grads1 = {'weights': np.array([10.0])}
        
        # Parameter with small gradients
        params2 = {'weights': np.array([1.0])}
        grads2 = {'weights': np.array([0.1])}
        
        opt.update(params1, grads1, layer_id=0)
        opt.update(params2, grads2, layer_id=1)
        
        # Large gradient param should move less relative to gradient
        change1 = 1.0 - params1['weights'][0]
        change2 = 1.0 - params2['weights'][0]
        
        # Change should be scaled differently
        assert change1 != change2
    
    def test_reset(self):
        """Test squared gradient cache reset."""
        opt = RMSprop()
        params = {'weights': np.array([1.0])}
        grads = {'weights': np.array([0.1])}
        
        opt.update(params, grads, layer_id=0)
        assert len(opt.v) > 0
        
        opt.reset()
        assert len(opt.v) == 0
    
    def test_repr(self):
        """Test string representation."""
        opt = RMSprop(learning_rate=0.001, decay=0.9)
        assert "RMSprop(lr=0.001, decay=0.9)" == repr(opt)


class TestGetOptimizer:
    """Tests for optimizer factory function."""
    
    def test_get_sgd(self):
        """Test getting SGD optimizer."""
        opt = get_optimizer('sgd', learning_rate=0.01)
        assert isinstance(opt, SGD)
        assert opt.lr == 0.01
    
    def test_get_momentum(self):
        """Test getting SGD with momentum."""
        opt = get_optimizer('momentum', learning_rate=0.01, momentum=0.95)
        assert isinstance(opt, SGDMomentum)
        assert opt.momentum == 0.95
    
    def test_get_adam(self):
        """Test getting Adam optimizer."""
        opt = get_optimizer('adam', learning_rate=0.001)
        assert isinstance(opt, Adam)
    
    def test_get_rmsprop(self):
        """Test getting RMSprop optimizer."""
        opt = get_optimizer('rmsprop', learning_rate=0.001)
        assert isinstance(opt, RMSprop)
    
    def test_case_insensitive(self):
        """Test that names are case insensitive."""
        opt1 = get_optimizer('ADAM')
        opt2 = get_optimizer('Adam')
        opt3 = get_optimizer('adam')
        
        assert isinstance(opt1, Adam)
        assert isinstance(opt2, Adam)
        assert isinstance(opt3, Adam)
    
    def test_unknown_optimizer(self):
        """Test that unknown optimizer raises error."""
        with pytest.raises(ValueError, match="Unknown optimizer"):
            get_optimizer('unknown')


class TestOptimizerIntegration:
    """Integration tests for optimizers with realistic scenarios."""
    
    def test_simple_linear_regression(self):
        """Test optimizers on simple linear regression."""
        np.random.seed(42)
        
        # Generate data: y = 2x + 1
        X = np.random.randn(100, 1)
        y = 2 * X + 1
        
        # Test each optimizer
        optimizers = [
            SGD(learning_rate=0.1),
            SGDMomentum(learning_rate=0.1, momentum=0.9),
            Adam(learning_rate=0.1),
            RMSprop(learning_rate=0.1),
        ]
        
        for opt in optimizers:
            # Initialize parameters
            w = np.array([[0.0]])
            b = np.array([[0.0]])
            params = {'weights': w, 'bias': b}
            
            # Train for a few iterations
            for _ in range(100):
                # Forward: y_pred = X @ w + b
                y_pred = X @ params['weights'] + params['bias']
                
                # Loss gradient (MSE)
                error = y_pred - y
                dw = 2 * X.T @ error / len(X)
                db = 2 * np.mean(error, axis=0, keepdims=True)
                
                grads = {'weights': dw, 'bias': db}
                opt.update(params, grads)
            
            # Should be close to w=2, b=1
            assert abs(params['weights'][0, 0] - 2.0) < 1.0
            assert abs(params['bias'][0, 0] - 1.0) < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
