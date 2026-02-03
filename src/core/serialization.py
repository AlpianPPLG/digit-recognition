"""
Model Serialization and Persistence

This module handles saving and loading neural network models, including:
- Saving/loading model weights
- Saving/loading model architecture
- Complete model checkpointing
- Model versioning
"""

import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union
import hashlib


class ModelSerializer:
    """
    Handles serialization of neural network models.
    
    Supports:
    - Weights-only saving/loading (NPZ format)
    - Full model saving (weights + architecture)
    - Checkpoint management
    - Model metadata
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model serializer.
        
        Args:
            models_dir: Directory for saving models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def save_weights(self, 
                     network, 
                     filepath: str,
                     include_optimizer: bool = False,
                     optimizer = None) -> str:
        """
        Save only model weights to NPZ file.
        
        Args:
            network: Neural network with layers
            filepath: Path to save weights
            include_optimizer: Whether to save optimizer state
            optimizer: Optimizer instance (required if include_optimizer=True)
            
        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Collect weights from all layers
        weights_dict = {}
        for i, layer in enumerate(network.layers):
            if hasattr(layer, 'weights') and layer.weights is not None:
                weights_dict[f'layer_{i}_weights'] = layer.weights
            if hasattr(layer, 'bias') and layer.bias is not None:
                weights_dict[f'layer_{i}_bias'] = layer.bias
        
        # Include optimizer state if requested
        if include_optimizer and optimizer is not None:
            optimizer_state = self._get_optimizer_state(optimizer)
            for key, value in optimizer_state.items():
                weights_dict[f'optimizer_{key}'] = np.array([value]) if not isinstance(value, np.ndarray) else value
        
        np.savez(filepath, **weights_dict)
        return str(filepath)
    
    def load_weights(self, 
                     network, 
                     filepath: str,
                     load_optimizer: bool = False,
                     optimizer = None) -> None:
        """
        Load weights into network from NPZ file.
        
        Args:
            network: Neural network to load weights into
            filepath: Path to weights file
            load_optimizer: Whether to load optimizer state
            optimizer: Optimizer instance (required if load_optimizer=True)
        """
        data = np.load(filepath, allow_pickle=True)
        
        for i, layer in enumerate(network.layers):
            weight_key = f'layer_{i}_weights'
            bias_key = f'layer_{i}_bias'
            
            if weight_key in data:
                if hasattr(layer, 'weights'):
                    layer.weights = data[weight_key]
            if bias_key in data:
                if hasattr(layer, 'bias'):
                    layer.bias = data[bias_key]
        
        # Load optimizer state if requested
        if load_optimizer and optimizer is not None:
            self._set_optimizer_state(optimizer, data)
    
    def save_model(self,
                   network,
                   filepath: str,
                   optimizer = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   training_history = None) -> str:
        """
        Save complete model (architecture + weights + metadata).
        
        Args:
            network: Neural network to save
            filepath: Path to save model
            optimizer: Optimizer instance (optional)
            metadata: Additional metadata to save
            training_history: Training history object (optional)
            
        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Build save dictionary
        save_dict = {
            'version': self.VERSION,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save architecture
        architecture = self._extract_architecture(network)
        save_dict['architecture'] = np.array([json.dumps(architecture)])
        
        # Save weights
        for i, layer in enumerate(network.layers):
            if hasattr(layer, 'weights') and layer.weights is not None:
                save_dict[f'layer_{i}_weights'] = layer.weights
            if hasattr(layer, 'bias') and layer.bias is not None:
                save_dict[f'layer_{i}_bias'] = layer.bias
        
        # Save optimizer state
        if optimizer is not None:
            optimizer_state = self._get_optimizer_state(optimizer)
            save_dict['optimizer_state'] = np.array([json.dumps(optimizer_state)])
            save_dict['optimizer_type'] = np.array([type(optimizer).__name__])
        
        # Save metadata
        if metadata is None:
            metadata = {}
        metadata['model_hash'] = self._compute_model_hash(network)
        save_dict['metadata'] = np.array([json.dumps(metadata)])
        
        # Save training history
        if training_history is not None:
            history_dict = {}
            for key in training_history.history:
                history_dict[key] = training_history.history[key]
            save_dict['training_history'] = np.array([json.dumps(history_dict)])
        
        np.savez_compressed(filepath, **save_dict)
        return str(filepath)
    
    def load_model(self, 
                   filepath: str,
                   network_builder = None) -> Tuple[Any, Optional[Dict], Optional[Dict]]:
        """
        Load complete model from file.
        
        Args:
            filepath: Path to model file
            network_builder: Optional builder for creating network
            
        Returns:
            Tuple of (network, optimizer_state, metadata)
        """
        data = np.load(filepath, allow_pickle=True)
        
        # Load architecture
        architecture = json.loads(str(data['architecture'][0]))
        
        # Create network from architecture
        if network_builder is not None:
            network = self._build_network_from_architecture(architecture, network_builder)
        else:
            # Return architecture for manual building
            from core.network import NetworkBuilder
            network = self._build_network_from_architecture(architecture, NetworkBuilder())
        
        # Collect weight keys and their indices
        weight_keys = sorted([
            (int(k.split('_')[1]), k) 
            for k in data.files 
            if k.startswith('layer_') and k.endswith('_weights')
        ])
        
        # Get layers that have weights
        weight_layers = [
            layer for layer in network.layers 
            if hasattr(layer, 'weights') and layer.weights is not None
        ]
        
        # Load weights by matching shapes
        for layer_idx, layer in enumerate(weight_layers):
            for saved_idx, weight_key in weight_keys:
                saved_weights = data[weight_key]
                if saved_weights.shape == layer.weights.shape:
                    layer.weights = saved_weights
                    bias_key = f'layer_{saved_idx}_bias'
                    if bias_key in data and hasattr(layer, 'bias'):
                        if data[bias_key].shape == layer.bias.shape:
                            layer.bias = data[bias_key]
                    # Remove used key
                    weight_keys.remove((saved_idx, weight_key))
                    break
        
        # Load optimizer state
        optimizer_state = None
        if 'optimizer_state' in data:
            optimizer_state = json.loads(str(data['optimizer_state'][0]))
        
        # Load metadata
        metadata = None
        if 'metadata' in data:
            metadata = json.loads(str(data['metadata'][0]))
        
        return network, optimizer_state, metadata
    
    def _extract_architecture(self, network) -> List[Dict[str, Any]]:
        """Extract architecture from network."""
        architecture = []
        
        for layer in network.layers:
            layer_config = {
                'type': type(layer).__name__,
            }
            
            # Dense layer specifics
            if hasattr(layer, 'input_size'):
                layer_config['input_size'] = layer.input_size
            if hasattr(layer, 'output_size'):
                layer_config['output_size'] = layer.output_size
            
            # Activation specifics
            if hasattr(layer, 'activation_name'):
                layer_config['activation'] = layer.activation_name
            
            # Dropout specifics  
            if hasattr(layer, 'rate'):
                layer_config['dropout_rate'] = layer.rate
            
            architecture.append(layer_config)
        
        return architecture
    
    def _build_network_from_architecture(self, architecture: List[Dict], builder) -> Any:
        """Build network from architecture config."""
        for i, layer_config in enumerate(architecture):
            layer_type = layer_config['type']
            
            if layer_type == 'DenseLayer':
                if i == 0:
                    # First dense layer - need input size
                    builder.input(layer_config.get('input_size', 784))
                builder.dense(
                    layer_config['output_size'],
                    activation=None  # Activation added separately
                )
            elif layer_type == 'ActivationLayer':
                # Skip - handled with dense layer
                pass
            elif layer_type == 'DropoutLayer':
                builder.dropout(layer_config.get('dropout_rate', 0.5))
        
        return builder.build()
    
    def _get_optimizer_state(self, optimizer) -> Dict[str, Any]:
        """Get serializable optimizer state."""
        state = {
            'type': type(optimizer).__name__,
            'learning_rate': getattr(optimizer, 'learning_rate', None),
        }
        
        # Adam specific
        if hasattr(optimizer, 'beta1'):
            state['beta1'] = optimizer.beta1
        if hasattr(optimizer, 'beta2'):
            state['beta2'] = optimizer.beta2
        if hasattr(optimizer, 't'):
            state['t'] = optimizer.t
        
        # Momentum specific
        if hasattr(optimizer, 'momentum'):
            state['momentum'] = optimizer.momentum
        
        return state
    
    def _set_optimizer_state(self, optimizer, data) -> None:
        """Set optimizer state from loaded data."""
        # Learning rate
        if 'optimizer_learning_rate' in data:
            optimizer.learning_rate = float(data['optimizer_learning_rate'][0])
        
        # Adam specific
        if hasattr(optimizer, 't') and 'optimizer_t' in data:
            optimizer.t = int(data['optimizer_t'][0])
    
    def _compute_model_hash(self, network) -> str:
        """Compute hash of model weights for versioning."""
        hasher = hashlib.md5()
        
        for layer in network.layers:
            if hasattr(layer, 'weights') and layer.weights is not None:
                hasher.update(layer.weights.tobytes())
            if hasattr(layer, 'bias') and layer.bias is not None:
                hasher.update(layer.bias.tobytes())
        
        return hasher.hexdigest()[:12]


class CheckpointManager:
    """
    Manages model checkpoints during training.
    
    Features:
    - Automatic checkpoint saving
    - Best model tracking
    - Checkpoint cleanup
    - Resume from checkpoint
    """
    
    def __init__(self,
                 checkpoint_dir: str,
                 max_checkpoints: int = 5,
                 save_best_only: bool = True,
                 monitor: str = 'val_loss',
                 mode: str = 'min'):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Only save when model improves
            monitor: Metric to monitor for improvement
            mode: 'min' or 'max' for improvement direction
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        
        self.serializer = ModelSerializer(str(checkpoint_dir))
        self.checkpoints: List[Tuple[str, float]] = []
        
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.best_checkpoint = None
    
    def save_checkpoint(self,
                        network,
                        epoch: int,
                        metrics: Dict[str, float],
                        optimizer = None) -> Optional[str]:
        """
        Save checkpoint if conditions are met.
        
        Args:
            network: Network to checkpoint
            epoch: Current epoch number
            metrics: Training metrics
            optimizer: Optimizer (optional)
            
        Returns:
            Path to checkpoint if saved, None otherwise
        """
        current_value = metrics.get(self.monitor, 0)
        
        # Check if we should save
        is_better = False
        if self.mode == 'min':
            is_better = current_value < self.best_value
        else:
            is_better = current_value > self.best_value
        
        if self.save_best_only and not is_better:
            return None
        
        # Update best
        if is_better:
            self.best_value = current_value
        
        # Generate checkpoint name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch{epoch:04d}_{timestamp}.npz"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save checkpoint
        self.serializer.save_model(
            network,
            str(checkpoint_path),
            optimizer=optimizer,
            metadata={
                'epoch': epoch,
                'metrics': metrics
            }
        )
        
        # Track checkpoint
        self.checkpoints.append((str(checkpoint_path), current_value))
        
        if is_better:
            self.best_checkpoint = str(checkpoint_path)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, 
                        checkpoint_path: str,
                        network_builder = None) -> Tuple[Any, int, Dict]:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            network_builder: Optional network builder
            
        Returns:
            Tuple of (network, epoch, metrics)
        """
        network, optimizer_state, metadata = self.serializer.load_model(
            checkpoint_path, network_builder
        )
        
        epoch = metadata.get('epoch', 0) if metadata else 0
        metrics = metadata.get('metrics', {}) if metadata else {}
        
        return network, epoch, metrics
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        return self.best_checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        if self.checkpoints:
            return self.checkpoints[-1][0]
        
        # Search directory
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.npz"))
        if checkpoints:
            return str(checkpoints[-1])
        
        return None
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        while len(self.checkpoints) > self.max_checkpoints:
            old_path, _ = self.checkpoints.pop(0)
            
            # Don't delete best checkpoint
            if old_path != self.best_checkpoint:
                try:
                    os.remove(old_path)
                except OSError:
                    pass


class ModelExporter:
    """
    Export models to different formats for deployment.
    """
    
    @staticmethod
    def export_to_json(network, filepath: str) -> str:
        """
        Export model to JSON format (weights as base64).
        
        Args:
            network: Network to export
            filepath: Output path
            
        Returns:
            Path to exported file
        """
        import base64
        
        model_dict = {
            'format': 'aidigit_json_v1',
            'timestamp': datetime.now().isoformat(),
            'layers': []
        }
        
        for i, layer in enumerate(network.layers):
            layer_dict = {
                'type': type(layer).__name__,
                'index': i
            }
            
            if hasattr(layer, 'weights') and layer.weights is not None:
                layer_dict['weights'] = {
                    'data': base64.b64encode(layer.weights.tobytes()).decode('ascii'),
                    'shape': list(layer.weights.shape),
                    'dtype': str(layer.weights.dtype)
                }
            
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer_dict['bias'] = {
                    'data': base64.b64encode(layer.bias.tobytes()).decode('ascii'),
                    'shape': list(layer.bias.shape),
                    'dtype': str(layer.bias.dtype)
                }
            
            if hasattr(layer, 'activation_name'):
                layer_dict['activation'] = layer.activation_name
            
            model_dict['layers'].append(layer_dict)
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(model_dict, f, indent=2)
        
        return str(filepath)
    
    @staticmethod
    def import_from_json(filepath: str, network) -> None:
        """
        Import model from JSON format.
        
        Args:
            filepath: Path to JSON file
            network: Network to load weights into
        """
        import base64
        
        with open(filepath, 'r') as f:
            model_dict = json.load(f)
        
        for layer_dict in model_dict['layers']:
            idx = layer_dict['index']
            if idx >= len(network.layers):
                continue
            
            layer = network.layers[idx]
            
            if 'weights' in layer_dict and hasattr(layer, 'weights'):
                w_data = layer_dict['weights']
                weights = np.frombuffer(
                    base64.b64decode(w_data['data']),
                    dtype=np.dtype(w_data['dtype'])
                ).reshape(w_data['shape'])
                layer.weights = weights
            
            if 'bias' in layer_dict and hasattr(layer, 'bias'):
                b_data = layer_dict['bias']
                bias = np.frombuffer(
                    base64.b64decode(b_data['data']),
                    dtype=np.dtype(b_data['dtype'])
                ).reshape(b_data['shape'])
                layer.bias = bias


# Convenience functions
def save_model(network, filepath: str, **kwargs) -> str:
    """
    Convenience function to save model.
    
    Args:
        network: Network to save
        filepath: Path to save to
        **kwargs: Additional arguments for ModelSerializer.save_model
        
    Returns:
        Path to saved model
    """
    serializer = ModelSerializer()
    return serializer.save_model(network, filepath, **kwargs)


def load_model(filepath: str, **kwargs) -> Tuple[Any, Optional[Dict], Optional[Dict]]:
    """
    Convenience function to load model.
    
    Args:
        filepath: Path to model file
        **kwargs: Additional arguments for ModelSerializer.load_model
        
    Returns:
        Tuple of (network, optimizer_state, metadata)
    """
    serializer = ModelSerializer()
    return serializer.load_model(filepath, **kwargs)


def save_weights(network, filepath: str) -> str:
    """
    Convenience function to save weights only.
    
    Args:
        network: Network to save
        filepath: Path to save to
        
    Returns:
        Path to saved weights
    """
    serializer = ModelSerializer()
    return serializer.save_weights(network, filepath)


def load_weights(network, filepath: str) -> None:
    """
    Convenience function to load weights.
    
    Args:
        network: Network to load into
        filepath: Path to weights file
    """
    serializer = ModelSerializer()
    serializer.load_weights(network, filepath)
