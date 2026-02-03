"""
Unit Tests for Model Serialization

Tests for saving, loading, and exporting models.
"""

import pytest
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from core.serialization import (
    ModelSerializer,
    CheckpointManager,
    ModelExporter,
    save_model,
    load_model,
    save_weights,
    load_weights
)
from core.network import NeuralNetwork, NetworkBuilder
from core.layers import DenseLayer, ActivationLayer
from core.optimizers import Adam, SGD


class TestModelSerializer:
    """Tests for ModelSerializer class."""
    
    @pytest.fixture
    def simple_network(self):
        """Create a simple network for testing."""
        np.random.seed(42)
        return (NetworkBuilder()
                .input(10)
                .dense(5, activation='relu')
                .dense(3, activation='softmax')
                .build())
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_initialization(self, temp_dir):
        """Test serializer initialization."""
        serializer = ModelSerializer(temp_dir)
        assert serializer.models_dir.exists()
    
    def test_save_weights(self, simple_network, temp_dir):
        """Test saving weights."""
        serializer = ModelSerializer(temp_dir)
        filepath = os.path.join(temp_dir, "weights.npz")
        
        saved_path = serializer.save_weights(simple_network, filepath)
        
        assert os.path.exists(saved_path)
        assert saved_path.endswith(".npz")
    
    def test_load_weights(self, simple_network, temp_dir):
        """Test loading weights."""
        serializer = ModelSerializer(temp_dir)
        filepath = os.path.join(temp_dir, "weights.npz")
        
        # Get original weights
        orig_weights = []
        for layer in simple_network.layers:
            if hasattr(layer, 'weights') and layer.weights is not None:
                orig_weights.append(layer.weights.copy())
        
        # Save weights
        serializer.save_weights(simple_network, filepath)
        
        # Create new network with different weights
        np.random.seed(123)
        new_network = (NetworkBuilder()
                       .input(10)
                       .dense(5, activation='relu')
                       .dense(3, activation='softmax')
                       .build())
        
        # Load weights
        serializer.load_weights(new_network, filepath)
        
        # Verify weights match
        loaded_idx = 0
        for layer in new_network.layers:
            if hasattr(layer, 'weights') and layer.weights is not None:
                np.testing.assert_array_almost_equal(
                    layer.weights, orig_weights[loaded_idx]
                )
                loaded_idx += 1
    
    def test_save_weights_with_optimizer(self, simple_network, temp_dir):
        """Test saving weights with optimizer state."""
        serializer = ModelSerializer(temp_dir)
        filepath = os.path.join(temp_dir, "weights_opt.npz")
        
        optimizer = Adam(learning_rate=0.001)
        
        saved_path = serializer.save_weights(
            simple_network, filepath,
            include_optimizer=True,
            optimizer=optimizer
        )
        
        data = np.load(saved_path, allow_pickle=True)
        assert 'optimizer_type' in data or 'optimizer_learning_rate' in data
    
    def test_save_model(self, simple_network, temp_dir):
        """Test saving complete model."""
        serializer = ModelSerializer(temp_dir)
        filepath = os.path.join(temp_dir, "model.npz")
        
        metadata = {'accuracy': 0.95, 'epochs': 10}
        saved_path = serializer.save_model(
            simple_network, filepath,
            metadata=metadata
        )
        
        assert os.path.exists(saved_path)
        
        # Verify content
        data = np.load(saved_path, allow_pickle=True)
        assert 'architecture' in data
        assert 'metadata' in data
        assert 'version' in data
    
    def test_load_model(self, simple_network, temp_dir):
        """Test loading complete model."""
        serializer = ModelSerializer(temp_dir)
        filepath = os.path.join(temp_dir, "model.npz")
        
        # Save original weights
        orig_weights = []
        for layer in simple_network.layers:
            if hasattr(layer, 'weights') and layer.weights is not None:
                orig_weights.append(layer.weights.copy())
        
        # Save model
        metadata = {'test_key': 'test_value'}
        serializer.save_model(simple_network, filepath, metadata=metadata)
        
        # Load model
        loaded_network, opt_state, loaded_metadata = serializer.load_model(filepath)
        
        # Verify structure
        assert loaded_network is not None
        assert loaded_metadata is not None
        assert 'test_key' in loaded_metadata
        
        # Verify weights
        loaded_idx = 0
        for layer in loaded_network.layers:
            if hasattr(layer, 'weights') and layer.weights is not None:
                np.testing.assert_array_almost_equal(
                    layer.weights, orig_weights[loaded_idx]
                )
                loaded_idx += 1
    
    def test_model_hash(self, simple_network, temp_dir):
        """Test model hash computation."""
        serializer = ModelSerializer(temp_dir)
        
        hash1 = serializer._compute_model_hash(simple_network)
        
        # Same network should have same hash
        hash2 = serializer._compute_model_hash(simple_network)
        assert hash1 == hash2
        
        # Different network should have different hash
        np.random.seed(99)
        different_network = (NetworkBuilder()
                            .input(10)
                            .dense(5, activation='relu')
                            .dense(3, activation='softmax')
                            .build())
        
        hash3 = serializer._compute_model_hash(different_network)
        assert hash1 != hash3


class TestCheckpointManager:
    """Tests for CheckpointManager."""
    
    @pytest.fixture
    def simple_network(self):
        """Create simple network."""
        np.random.seed(42)
        return (NetworkBuilder()
                .input(10)
                .dense(5, activation='relu')
                .dense(3, activation='softmax')
                .build())
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_initialization(self, temp_dir):
        """Test checkpoint manager initialization."""
        manager = CheckpointManager(temp_dir, max_checkpoints=3)
        
        assert manager.checkpoint_dir.exists()
        assert manager.max_checkpoints == 3
    
    def test_save_checkpoint(self, simple_network, temp_dir):
        """Test saving checkpoint."""
        manager = CheckpointManager(
            temp_dir, 
            save_best_only=False
        )
        
        metrics = {'loss': 0.5, 'accuracy': 0.8}
        path = manager.save_checkpoint(simple_network, epoch=1, metrics=metrics)
        
        assert path is not None
        assert os.path.exists(path)
    
    def test_save_best_only(self, simple_network, temp_dir):
        """Test saving best checkpoint only."""
        manager = CheckpointManager(
            temp_dir,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
        
        # First checkpoint - should save
        path1 = manager.save_checkpoint(
            simple_network, epoch=1,
            metrics={'val_loss': 0.5}
        )
        assert path1 is not None
        
        # Worse checkpoint - should not save
        path2 = manager.save_checkpoint(
            simple_network, epoch=2,
            metrics={'val_loss': 0.6}
        )
        assert path2 is None
        
        # Better checkpoint - should save
        path3 = manager.save_checkpoint(
            simple_network, epoch=3,
            metrics={'val_loss': 0.4}
        )
        assert path3 is not None
    
    def test_max_checkpoints(self, simple_network, temp_dir):
        """Test checkpoint cleanup."""
        manager = CheckpointManager(
            temp_dir,
            max_checkpoints=2,
            save_best_only=False
        )
        
        # Save 4 checkpoints
        for i in range(4):
            manager.save_checkpoint(
                simple_network, epoch=i+1,
                metrics={'loss': 0.5 - i*0.1}
            )
        
        # Should only have 2 checkpoints
        checkpoints = list(Path(temp_dir).glob("checkpoint_*.npz"))
        assert len(checkpoints) <= 3  # max_checkpoints + best
    
    def test_load_checkpoint(self, simple_network, temp_dir):
        """Test loading checkpoint."""
        manager = CheckpointManager(temp_dir, save_best_only=False)
        
        metrics = {'loss': 0.5, 'accuracy': 0.8}
        path = manager.save_checkpoint(simple_network, epoch=5, metrics=metrics)
        
        loaded_network, epoch, loaded_metrics = manager.load_checkpoint(path)
        
        assert loaded_network is not None
        assert epoch == 5
        assert loaded_metrics['loss'] == 0.5
    
    def test_get_best_checkpoint(self, simple_network, temp_dir):
        """Test getting best checkpoint."""
        manager = CheckpointManager(
            temp_dir,
            save_best_only=False,
            monitor='val_loss',
            mode='min'
        )
        
        manager.save_checkpoint(simple_network, 1, {'val_loss': 0.5})
        manager.save_checkpoint(simple_network, 2, {'val_loss': 0.3})
        manager.save_checkpoint(simple_network, 3, {'val_loss': 0.4})
        
        best = manager.get_best_checkpoint()
        assert best is not None
        assert 'epoch0002' in best  # Epoch 2 had lowest loss
    
    def test_get_latest_checkpoint(self, simple_network, temp_dir):
        """Test getting latest checkpoint."""
        manager = CheckpointManager(temp_dir, save_best_only=False)
        
        manager.save_checkpoint(simple_network, 1, {'loss': 0.5})
        manager.save_checkpoint(simple_network, 2, {'loss': 0.4})
        
        latest = manager.get_latest_checkpoint()
        assert latest is not None
        assert 'epoch0002' in latest


class TestModelExporter:
    """Tests for ModelExporter."""
    
    @pytest.fixture
    def simple_network(self):
        """Create simple network."""
        np.random.seed(42)
        return (NetworkBuilder()
                .input(10)
                .dense(5, activation='relu')
                .dense(3, activation='softmax')
                .build())
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_export_to_json(self, simple_network, temp_dir):
        """Test exporting model to JSON."""
        filepath = os.path.join(temp_dir, "model.json")
        
        exported_path = ModelExporter.export_to_json(simple_network, filepath)
        
        assert os.path.exists(exported_path)
        
        # Verify JSON structure
        import json
        with open(exported_path, 'r') as f:
            data = json.load(f)
        
        assert 'format' in data
        assert 'layers' in data
        assert len(data['layers']) > 0
    
    def test_import_from_json(self, simple_network, temp_dir):
        """Test importing model from JSON."""
        filepath = os.path.join(temp_dir, "model.json")
        
        # Get original weights
        orig_weights = []
        for layer in simple_network.layers:
            if hasattr(layer, 'weights') and layer.weights is not None:
                orig_weights.append(layer.weights.copy())
        
        # Export
        ModelExporter.export_to_json(simple_network, filepath)
        
        # Create new network
        np.random.seed(99)
        new_network = (NetworkBuilder()
                       .input(10)
                       .dense(5, activation='relu')
                       .dense(3, activation='softmax')
                       .build())
        
        # Import
        ModelExporter.import_from_json(filepath, new_network)
        
        # Verify weights
        loaded_idx = 0
        for layer in new_network.layers:
            if hasattr(layer, 'weights') and layer.weights is not None:
                np.testing.assert_array_almost_equal(
                    layer.weights, orig_weights[loaded_idx]
                )
                loaded_idx += 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    @pytest.fixture
    def simple_network(self):
        """Create simple network."""
        np.random.seed(42)
        return (NetworkBuilder()
                .input(10)
                .dense(5, activation='relu')
                .dense(3, activation='softmax')
                .build())
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_save_and_load_model(self, simple_network, temp_dir):
        """Test save_model and load_model functions."""
        filepath = os.path.join(temp_dir, "model.npz")
        
        # Save
        save_model(simple_network, filepath)
        assert os.path.exists(filepath)
        
        # Load
        loaded_net, _, _ = load_model(filepath)
        assert loaded_net is not None
    
    def test_save_and_load_weights(self, simple_network, temp_dir):
        """Test save_weights and load_weights functions."""
        filepath = os.path.join(temp_dir, "weights.npz")
        
        # Get original weights
        orig_weights = []
        for layer in simple_network.layers:
            if hasattr(layer, 'weights') and layer.weights is not None:
                orig_weights.append(layer.weights.copy())
        
        # Save
        save_weights(simple_network, filepath)
        
        # Create new network
        np.random.seed(99)
        new_network = (NetworkBuilder()
                       .input(10)
                       .dense(5, activation='relu')
                       .dense(3, activation='softmax')
                       .build())
        
        # Load
        load_weights(new_network, filepath)
        
        # Verify
        loaded_idx = 0
        for layer in new_network.layers:
            if hasattr(layer, 'weights') and layer.weights is not None:
                np.testing.assert_array_almost_equal(
                    layer.weights, orig_weights[loaded_idx]
                )
                loaded_idx += 1


class TestSerializationIntegration:
    """Integration tests for serialization."""
    
    def test_full_workflow(self):
        """Test complete save/load workflow."""
        np.random.seed(42)
        
        # Create and "train" network
        network = (NetworkBuilder()
                   .input(10)
                   .dense(8, activation='relu')
                   .dense(4, activation='softmax')
                   .build())
        
        # Get original weights before save
        orig_weights = []
        for layer in network.layers:
            if hasattr(layer, 'weights') and layer.weights is not None:
                orig_weights.append(layer.weights.copy())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.npz")
            
            # Save
            save_model(network, filepath, metadata={'test': True})
            
            # Load into new network
            loaded_net, _, metadata = load_model(filepath)
            
            # Verify weights match
            loaded_idx = 0
            for layer in loaded_net.layers:
                if hasattr(layer, 'weights') and layer.weights is not None:
                    np.testing.assert_array_almost_equal(
                        layer.weights, orig_weights[loaded_idx]
                    )
                    loaded_idx += 1
            
            # Verify metadata
            assert metadata['test'] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
