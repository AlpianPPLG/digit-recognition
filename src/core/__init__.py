"""
Core Module - Neural Network Implementation
===========================================

This module contains the core neural network components built from scratch:
- Network architecture (network.py)
- Layer implementations (layers.py)
- Activation functions (activations.py)
- Loss functions (losses.py)
- Optimizers (optimizers.py)
- Weight initializers (initializers.py)
- Regularizers (regularizers.py)
- Evaluation metrics (metrics.py)
"""

from .network import NeuralNetwork
from .layers import Layer, DenseLayer, ActivationLayer, DropoutLayer
from .activations import sigmoid, relu, leaky_relu, softmax
from .losses import CrossEntropyLoss, MSELoss
from .optimizers import SGD, SGDMomentum, Adam, RMSprop
from .initializers import xavier_init, he_init, random_init
from .metrics import accuracy, precision, recall, f1_score, confusion_matrix
from .serialization import (
    ModelSerializer, CheckpointManager, ModelExporter,
    save_model, load_model, save_weights, load_weights
)

__all__ = [
    # Network
    'NeuralNetwork',
    
    # Layers
    'Layer',
    'DenseLayer',
    'ActivationLayer',
    'DropoutLayer',
    
    # Activations
    'sigmoid',
    'relu',
    'leaky_relu',
    'softmax',
    
    # Losses
    'CrossEntropyLoss',
    'MSELoss',
    
    # Optimizers
    'SGD',
    'SGDMomentum',
    'Adam',
    'RMSprop',
    
    # Initializers
    'xavier_init',
    'he_init',
    'random_init',
    
    # Metrics
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'confusion_matrix',
    
    # Serialization
    'ModelSerializer',
    'CheckpointManager',
    'ModelExporter',
    'save_model',
    'load_model',
    'save_weights',
    'load_weights',
]
