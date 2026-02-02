"""
Metrics Module for AiDigit Neural Network

This module provides evaluation metrics for assessing model performance:
- Accuracy: Proportion of correct predictions
- Precision: Positive predictive value
- Recall: True positive rate (sensitivity)
- F1 Score: Harmonic mean of precision and recall
- Confusion Matrix: Detailed classification breakdown

All metrics work with both one-hot encoded and integer label formats.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Union


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate classification accuracy.
    
    Accuracy = (Number of correct predictions) / (Total predictions)
    
    Args:
        y_true: True labels (one-hot encoded or integer class labels)
        y_pred: Predicted labels (probabilities or integer class labels)
    
    Returns:
        Accuracy as a float in range [0, 1]
    
    Example:
        >>> y_true = np.array([0, 1, 2, 0, 1])
        >>> y_pred = np.array([0, 1, 2, 0, 0])
        >>> accuracy(y_true, y_pred)
        0.8
    """
    # Convert one-hot to class indices if needed
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    
    return np.mean(y_true == y_pred)


def precision(y_true: np.ndarray, y_pred: np.ndarray, 
              average: str = 'macro') -> Union[float, np.ndarray]:
    """
    Calculate precision (positive predictive value).
    
    Precision = TP / (TP + FP)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method:
            - 'macro': Unweighted mean of per-class precision
            - 'micro': Global precision (TP_total / (TP + FP)_total)
            - 'weighted': Weighted mean by support
            - None: Return per-class precision
    
    Returns:
        Precision score(s)
    
    Example:
        >>> y_true = np.array([0, 1, 2, 0, 1, 2])
        >>> y_pred = np.array([0, 1, 1, 0, 0, 2])
        >>> precision(y_true, y_pred, average='macro')
    """
    # Convert one-hot to class indices if needed
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    # Calculate per-class precision
    precisions = np.zeros(n_classes)
    supports = np.zeros(n_classes)
    
    for i, c in enumerate(classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        supports[i] = np.sum(y_true == c)
        
        if tp + fp > 0:
            precisions[i] = tp / (tp + fp)
        else:
            precisions[i] = 0.0
    
    if average is None:
        return precisions
    elif average == 'macro':
        return np.mean(precisions)
    elif average == 'micro':
        # Global precision
        tp_total = np.sum([
            np.sum((y_true == c) & (y_pred == c)) for c in classes
        ])
        fp_total = np.sum([
            np.sum((y_true != c) & (y_pred == c)) for c in classes
        ])
        return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    elif average == 'weighted':
        total_support = np.sum(supports)
        if total_support > 0:
            return np.sum(precisions * supports) / total_support
        return 0.0
    else:
        raise ValueError(f"Unknown average method: {average}")


def recall(y_true: np.ndarray, y_pred: np.ndarray,
           average: str = 'macro') -> Union[float, np.ndarray]:
    """
    Calculate recall (sensitivity, true positive rate).
    
    Recall = TP / (TP + FN)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('macro', 'micro', 'weighted', None)
    
    Returns:
        Recall score(s)
    
    Example:
        >>> y_true = np.array([0, 1, 2, 0, 1, 2])
        >>> y_pred = np.array([0, 1, 1, 0, 0, 2])
        >>> recall(y_true, y_pred, average='macro')
    """
    # Convert one-hot to class indices if needed
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    # Calculate per-class recall
    recalls = np.zeros(n_classes)
    supports = np.zeros(n_classes)
    
    for i, c in enumerate(classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        supports[i] = np.sum(y_true == c)
        
        if tp + fn > 0:
            recalls[i] = tp / (tp + fn)
        else:
            recalls[i] = 0.0
    
    if average is None:
        return recalls
    elif average == 'macro':
        return np.mean(recalls)
    elif average == 'micro':
        # Global recall = global precision for multi-class
        tp_total = np.sum([
            np.sum((y_true == c) & (y_pred == c)) for c in classes
        ])
        fn_total = np.sum([
            np.sum((y_true == c) & (y_pred != c)) for c in classes
        ])
        return tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    elif average == 'weighted':
        total_support = np.sum(supports)
        if total_support > 0:
            return np.sum(recalls * supports) / total_support
        return 0.0
    else:
        raise ValueError(f"Unknown average method: {average}")


def f1_score(y_true: np.ndarray, y_pred: np.ndarray,
             average: str = 'macro') -> Union[float, np.ndarray]:
    """
    Calculate F1 score (harmonic mean of precision and recall).
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('macro', 'micro', 'weighted', None)
    
    Returns:
        F1 score(s)
    
    Example:
        >>> y_true = np.array([0, 1, 2, 0, 1, 2])
        >>> y_pred = np.array([0, 1, 1, 0, 0, 2])
        >>> f1_score(y_true, y_pred, average='macro')
    """
    p = precision(y_true, y_pred, average=average)
    r = recall(y_true, y_pred, average=average)
    
    if average is None:
        # Per-class F1
        f1 = np.zeros_like(p)
        mask = (p + r) > 0
        f1[mask] = 2 * p[mask] * r[mask] / (p[mask] + r[mask])
        return f1
    else:
        if p + r > 0:
            return 2 * p * r / (p + r)
        return 0.0


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     num_classes: Optional[int] = None) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Entry [i, j] is the number of samples with true label i
    and predicted label j.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes (inferred if None)
    
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    
    Example:
        >>> y_true = np.array([0, 0, 1, 1, 2])
        >>> y_pred = np.array([0, 1, 1, 1, 0])
        >>> confusion_matrix(y_true, y_pred)
        array([[1, 1, 0],
               [0, 2, 0],
               [1, 0, 0]])
    """
    # Convert one-hot to class indices if needed
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    
    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    return cm


def classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: Optional[list] = None) -> Dict:
    """
    Generate a detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names
    
    Returns:
        Dictionary containing per-class and overall metrics
    
    Example:
        >>> y_true = np.array([0, 1, 2, 0, 1, 2])
        >>> y_pred = np.array([0, 1, 1, 0, 0, 2])
        >>> report = classification_report(y_true, y_pred, ['A', 'B', 'C'])
    """
    # Convert one-hot to class indices if needed
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    if class_names is None:
        class_names = [str(c) for c in classes]
    
    # Per-class metrics
    p_per_class = precision(y_true, y_pred, average=None)
    r_per_class = recall(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    # Support (number of true instances per class)
    supports = np.array([np.sum(y_true == c) for c in classes])
    
    report = {
        'per_class': {},
        'accuracy': accuracy(y_true, y_pred),
        'macro_avg': {
            'precision': np.mean(p_per_class),
            'recall': np.mean(r_per_class),
            'f1_score': np.mean(f1_per_class),
        },
        'weighted_avg': {
            'precision': precision(y_true, y_pred, average='weighted'),
            'recall': recall(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
        },
        'total_samples': len(y_true),
    }
    
    for i, name in enumerate(class_names[:n_classes]):
        report['per_class'][name] = {
            'precision': float(p_per_class[i]),
            'recall': float(r_per_class[i]),
            'f1_score': float(f1_per_class[i]),
            'support': int(supports[i]),
        }
    
    return report


def top_k_accuracy(y_true: np.ndarray, y_pred_probs: np.ndarray, 
                   k: int = 5) -> float:
    """
    Calculate top-k accuracy.
    
    A prediction is considered correct if the true class is among
    the top k predicted classes.
    
    Args:
        y_true: True labels (class indices or one-hot)
        y_pred_probs: Predicted probabilities (batch_size, num_classes)
        k: Number of top predictions to consider
    
    Returns:
        Top-k accuracy as a float in range [0, 1]
    
    Example:
        >>> y_true = np.array([0, 1, 2])
        >>> y_pred = np.array([[0.9, 0.05, 0.05],
        ...                    [0.1, 0.6, 0.3],
        ...                    [0.4, 0.4, 0.2]])
        >>> top_k_accuracy(y_true, y_pred, k=2)
    """
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)
    
    # Get indices of top-k predictions for each sample
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
    
    # Check if true label is in top-k
    correct = np.any(top_k_preds == y_true[:, np.newaxis], axis=1)
    
    return np.mean(correct)


def log_loss(y_true: np.ndarray, y_pred: np.ndarray, 
             epsilon: float = 1e-15) -> float:
    """
    Calculate log loss (cross-entropy loss).
    
    Log Loss = -mean(sum(y_true * log(y_pred)))
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities
        epsilon: Small value for numerical stability
    
    Returns:
        Log loss value
    """
    # Ensure one-hot encoding
    if y_true.ndim == 1:
        n_samples = len(y_true)
        n_classes = y_pred.shape[1]
        y_true_onehot = np.zeros((n_samples, n_classes))
        y_true_onehot[np.arange(n_samples), y_true] = 1
        y_true = y_true_onehot
    
    # Clip predictions for numerical stability
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                       num_classes: Optional[int] = None) -> np.ndarray:
    """
    Calculate accuracy for each class individually.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes (inferred if None)
    
    Returns:
        Array of per-class accuracies
    """
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    
    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    
    accuracies = np.zeros(num_classes)
    
    for c in range(num_classes):
        mask = y_true == c
        if np.sum(mask) > 0:
            accuracies[c] = np.mean(y_pred[mask] == c)
        else:
            accuracies[c] = 0.0
    
    return accuracies


# Convenience exports
__all__ = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'confusion_matrix',
    'classification_report',
    'top_k_accuracy',
    'log_loss',
    'per_class_accuracy',
]
