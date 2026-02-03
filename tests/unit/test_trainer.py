"""
Unit Tests for Training Module

Tests for trainer, early stopping, learning rate scheduler, etc.
"""

import pytest
import numpy as np
import sys
import tempfile
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from training.trainer import (
    EarlyStopping, 
    LearningRateScheduler, 
    TrainingHistory,
    Trainer,
    train_model
)
from core.network import NeuralNetwork, NetworkBuilder
from core.layers import DenseLayer, ActivationLayer
from core.optimizers import SGD, Adam
from core.losses import CrossEntropyLoss


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""
    
    def test_initialization(self):
        """Test early stopping initialization."""
        es = EarlyStopping(patience=5, min_delta=0.001)
        assert es.patience == 5
        assert es.min_delta == 0.001
        assert es.counter == 0
    
    def test_improvement_resets_counter(self):
        """Test that improvement resets counter."""
        es = EarlyStopping(patience=3, mode='min')
        
        # First value
        es(1.0)
        assert es.counter == 0
        
        # Improvement
        es(0.9)
        assert es.counter == 0
        
        # No improvement
        es(0.95)
        assert es.counter == 1
        
        # Improvement again
        es(0.8)
        assert es.counter == 0
    
    def test_stops_after_patience(self):
        """Test stopping after patience exhausted."""
        es = EarlyStopping(patience=3, mode='min')
        
        # Set baseline
        assert not es(1.0)
        
        # Three non-improvements
        assert not es(1.1)  # counter = 1
        assert not es(1.2)  # counter = 2
        assert es(1.3)      # counter = 3 -> stop
    
    def test_max_mode(self):
        """Test max mode (for accuracy)."""
        es = EarlyStopping(patience=2, mode='max')
        
        assert not es(0.8)  # Best = 0.8
        assert not es(0.9)  # Better, best = 0.9
        assert not es(0.85) # Worse, counter = 1
        assert es(0.85)     # Worse, counter = 2 -> stop
    
    def test_min_delta(self):
        """Test minimum delta threshold."""
        es = EarlyStopping(patience=2, min_delta=0.1, mode='min')
        
        assert not es(1.0)  # Best = 1.0
        assert not es(0.95) # Not enough improvement (delta < 0.1), counter = 1
        assert es(0.92)     # Still not enough, counter = 2 -> stop
    
    def test_weight_storage(self):
        """Test weight storage for restoration."""
        es = EarlyStopping(patience=2, restore_best=True)
        
        weights1 = [np.array([1, 2, 3])]
        weights2 = [np.array([4, 5, 6])]
        
        # First call stores weights
        es(1.0, weights1)
        np.testing.assert_array_equal(es.best_weights[0], weights1[0])
        
        # Better value stores new weights
        es(0.9, weights2)
        np.testing.assert_array_equal(es.best_weights[0], weights2[0])
    
    def test_reset(self):
        """Test reset functionality."""
        es = EarlyStopping(patience=2)
        es(1.0)
        es(1.1)
        es(1.2)
        
        assert es.counter > 0
        
        es.reset()
        assert es.counter == 0
        assert es.best_value == np.inf


class TestLearningRateScheduler:
    """Tests for LearningRateScheduler."""
    
    def test_constant_schedule(self):
        """Test constant learning rate."""
        scheduler = LearningRateScheduler(
            initial_lr=0.01, 
            schedule_type='constant'
        )
        
        for _ in range(10):
            lr = scheduler.step()
        
        assert scheduler.get_lr() == 0.01
    
    def test_step_schedule(self):
        """Test step decay schedule."""
        scheduler = LearningRateScheduler(
            initial_lr=0.1,
            schedule_type='step',
            step_size=5,
            gamma=0.1
        )
        
        # First 5 epochs: lr = 0.1
        for epoch in range(5):
            lr = scheduler.step(epoch)
            np.testing.assert_almost_equal(lr, 0.1)
        
        # Epochs 5-9: lr = 0.01
        for epoch in range(5, 10):
            lr = scheduler.step(epoch)
            np.testing.assert_almost_equal(lr, 0.01)
    
    def test_exponential_schedule(self):
        """Test exponential decay."""
        scheduler = LearningRateScheduler(
            initial_lr=0.1,
            schedule_type='exponential',
            gamma=0.9
        )
        
        lr0 = scheduler.step(0)
        lr1 = scheduler.step(1)
        
        np.testing.assert_almost_equal(lr0, 0.1)
        np.testing.assert_almost_equal(lr1, 0.09)  # 0.1 * 0.9
    
    def test_cosine_schedule(self):
        """Test cosine annealing."""
        scheduler = LearningRateScheduler(
            initial_lr=0.1,
            schedule_type='cosine',
            T_max=20,
            eta_min=0.001
        )
        
        # At epoch 0, lr should be initial
        lr = scheduler.step(0)
        np.testing.assert_almost_equal(lr, 0.1, decimal=3)
        
        # At T_max, lr should be eta_min
        lr = scheduler.step(20)
        np.testing.assert_almost_equal(lr, 0.001, decimal=3)
    
    def test_plateau_schedule(self):
        """Test reduce on plateau."""
        scheduler = LearningRateScheduler(
            initial_lr=0.1,
            schedule_type='plateau',
            factor=0.5,
            patience=2,
            min_lr=0.001
        )
        
        # Improving metrics - no reduction
        scheduler.step(0, metric=1.0)
        scheduler.step(1, metric=0.9)
        assert scheduler.get_lr() == 0.1
        
        # No improvement for patience epochs
        scheduler.step(2, metric=0.95)  # worse
        scheduler.step(3, metric=0.95)  # worse, patience reached
        
        # LR should be reduced
        assert scheduler.get_lr() == 0.05


class TestTrainingHistory:
    """Tests for TrainingHistory."""
    
    def test_record(self):
        """Test recording metrics."""
        history = TrainingHistory()
        
        history.record(loss=0.5, accuracy=0.8)
        history.record(loss=0.4, accuracy=0.85)
        
        assert history.get('loss') == [0.5, 0.4]
        assert history.get('accuracy') == [0.8, 0.85]
    
    def test_summary(self):
        """Test training summary."""
        history = TrainingHistory()
        
        history.record(loss=0.5, val_loss=0.6, accuracy=0.8, val_accuracy=0.75)
        history.record(loss=0.3, val_loss=0.4, accuracy=0.9, val_accuracy=0.88)
        
        summary = history.summary()
        
        assert summary['epochs'] == 2
        assert summary['best_loss'] == 0.3
        assert summary['best_val_loss'] == 0.4
        assert summary['best_accuracy'] == 0.9
        assert summary['best_val_accuracy'] == 0.88
    
    def test_custom_metrics(self):
        """Test recording custom metrics."""
        history = TrainingHistory()
        
        history.record(loss=0.5, custom_metric=42)
        
        assert history.get('custom_metric') == [42]


class TestTrainer:
    """Tests for Trainer class."""
    
    @pytest.fixture
    def simple_network(self):
        """Create simple network for testing."""
        np.random.seed(42)
        return (NetworkBuilder()
                .input(10)
                .dense(5, activation='relu')
                .dense(3, activation='softmax')
                .build())
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.zeros((100, 3), dtype=np.float32)
        y[np.arange(100), np.random.randint(0, 3, 100)] = 1
        return X, y
    
    def test_trainer_initialization(self, simple_network):
        """Test trainer initialization."""
        trainer = Trainer(simple_network)
        
        assert trainer.network is simple_network
        assert trainer.optimizer is not None
        assert trainer.loss_fn is not None
    
    def test_train_epoch(self, simple_network, sample_data):
        """Test single epoch training."""
        X, y = sample_data
        trainer = Trainer(simple_network)
        
        loss, acc = trainer.train_epoch(X, y, batch_size=32)
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1
    
    def test_evaluate(self, simple_network, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        trainer = Trainer(simple_network)
        
        loss, acc = trainer.evaluate(X, y)
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
    
    def test_fit_without_validation(self, simple_network, sample_data):
        """Test training without validation."""
        X, y = sample_data
        trainer = Trainer(simple_network)
        
        history = trainer.fit(X, y, epochs=3, batch_size=32, verbose=0)
        
        assert len(history.get('loss')) == 3
        assert len(history.get('accuracy')) == 3
    
    def test_fit_with_validation(self, simple_network, sample_data):
        """Test training with validation."""
        X, y = sample_data
        X_train, y_train = X[:80], y[:80]
        X_val, y_val = X[80:], y[80:]
        
        trainer = Trainer(simple_network)
        history = trainer.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=3,
            batch_size=16,
            verbose=0
        )
        
        assert len(history.get('val_loss')) == 3
        assert len(history.get('val_accuracy')) == 3
    
    def test_fit_with_early_stopping(self, simple_network, sample_data):
        """Test training with early stopping."""
        X, y = sample_data
        X_train, y_train = X[:80], y[:80]
        X_val, y_val = X[80:], y[80:]
        
        trainer = Trainer(simple_network)
        early_stop = EarlyStopping(patience=2, mode='min')
        
        history = trainer.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=100,  # High number, should stop early
            batch_size=16,
            early_stopping=early_stop,
            verbose=0
        )
        
        # Should stop before 100 epochs
        assert len(history.get('loss')) < 100
    
    def test_fit_with_lr_scheduler(self, simple_network, sample_data):
        """Test training with learning rate scheduler."""
        X, y = sample_data
        trainer = Trainer(simple_network, optimizer=SGD(learning_rate=0.1))
        
        scheduler = LearningRateScheduler(
            initial_lr=0.1,
            schedule_type='step',
            step_size=2,
            gamma=0.5
        )
        
        history = trainer.fit(
            X, y,
            epochs=5,
            batch_size=32,
            lr_scheduler=scheduler,
            verbose=0
        )
        
        lrs = history.get('lr')
        # LR should decrease
        assert lrs[-1] < lrs[0]
    
    def test_checkpoint_saving(self, simple_network, sample_data):
        """Test checkpoint saving."""
        X, y = sample_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(simple_network, checkpoint_dir=tmpdir)
            
            trainer.fit(
                X, y,
                epochs=4,
                batch_size=32,
                checkpoint_freq=2,
                verbose=0
            )
            
            # Should have checkpoints at epoch 2 and 4
            files = list(Path(tmpdir).glob("*.npz"))
            assert len(files) == 2


class TestTrainModelFunction:
    """Tests for train_model convenience function."""
    
    def test_train_model(self):
        """Test convenience training function."""
        np.random.seed(42)
        
        network = (NetworkBuilder()
                   .input(10)
                   .dense(5, activation='relu')
                   .dense(3, activation='softmax')
                   .build())
        
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.zeros((50, 3), dtype=np.float32)
        y[np.arange(50), np.random.randint(0, 3, 50)] = 1
        
        trained_net, history = train_model(
            network, X, y,
            epochs=3,
            batch_size=16,
            verbose=0
        )
        
        assert trained_net is network
        assert len(history.get('loss')) == 3


class TestTrainingIntegration:
    """Integration tests for training."""
    
    def test_training_improves_accuracy(self):
        """Test that training actually improves accuracy."""
        np.random.seed(42)
        
        # Create XOR-like problem
        X = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1],
            [0, 0], [0, 1], [1, 0], [1, 1],
        ] * 10, dtype=np.float32)
        y_raw = np.array([0, 1, 1, 0] * 20)
        y = np.zeros((80, 2), dtype=np.float32)
        y[np.arange(80), y_raw] = 1
        
        network = (NetworkBuilder()
                   .input(2)
                   .dense(8, activation='relu')
                   .dense(2, activation='softmax')
                   .build())
        
        trainer = Trainer(network, optimizer=Adam(learning_rate=0.1))
        
        # Get initial accuracy
        _, initial_acc = trainer.evaluate(X, y)
        
        # Train
        trainer.fit(X, y, epochs=50, batch_size=8, verbose=0)
        
        # Get final accuracy
        _, final_acc = trainer.evaluate(X, y)
        
        # Accuracy should improve
        assert final_acc > initial_acc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
