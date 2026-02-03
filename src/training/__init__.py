"""
Training Module - Model Training Utilities

This module contains training-related functionality:
- Training loop management
- MNIST data loading
- Batch generation
- Learning rate scheduling
- Model checkpointing
- Training callbacks
"""

from .trainer import (
    EarlyStopping,
    LearningRateScheduler,
    TrainingHistory,
    Trainer,
    train_model,
)


__all__ = [
    'EarlyStopping',
    'LearningRateScheduler',
    'TrainingHistory',
    'Trainer',
    'train_model',
]
