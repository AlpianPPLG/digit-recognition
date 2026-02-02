"""
Unit Tests for Metrics

Tests for src/core/metrics.py
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from core.metrics import (
    accuracy, precision, recall, f1_score, 
    confusion_matrix, classification_report,
    top_k_accuracy, log_loss, per_class_accuracy
)


class TestAccuracy:
    """Tests for accuracy metric."""
    
    def test_perfect_accuracy(self):
        """Test 100% accuracy."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        assert accuracy(y_true, y_pred) == 1.0
    
    def test_zero_accuracy(self):
        """Test 0% accuracy."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        assert accuracy(y_true, y_pred) == 0.0
    
    def test_partial_accuracy(self):
        """Test partial accuracy."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 0])  # 4/5 correct
        assert accuracy(y_true, y_pred) == 0.8
    
    def test_one_hot_input(self):
        """Test accuracy with one-hot encoded labels."""
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[0.9, 0.05, 0.05], 
                          [0.1, 0.8, 0.1], 
                          [0.1, 0.1, 0.8]])
        assert accuracy(y_true, y_pred) == 1.0
    
    def test_binary_classification(self):
        """Test binary classification accuracy."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])  # 2/4 correct
        assert accuracy(y_true, y_pred) == 0.5


class TestPrecision:
    """Tests for precision metric."""
    
    def test_perfect_precision(self):
        """Test 100% precision."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        assert precision(y_true, y_pred, average='macro') == 1.0
    
    def test_macro_precision(self):
        """Test macro-averaged precision."""
        # y_true = [0, 0, 1, 1, 2]
        # y_pred = [0, 0, 1, 0, 2]
        # Class 0: TP=2 (true=0,pred=0), FP=1 (true=1,pred=0) -> precision=2/3
        # Class 1: TP=1 (true=1,pred=1), FP=0 -> precision=1.0
        # Class 2: TP=1, FP=0 -> precision=1.0
        # Macro: (2/3 + 1.0 + 1.0) / 3 = 8/9
        y_true = np.array([0, 0, 1, 1, 2])
        y_pred = np.array([0, 0, 1, 0, 2])
        
        p = precision(y_true, y_pred, average='macro')
        np.testing.assert_almost_equal(p, 8/9)  # (2/3 + 1 + 1) / 3
    
    def test_micro_precision(self):
        """Test micro-averaged precision."""
        y_true = np.array([0, 0, 1, 1, 2])
        y_pred = np.array([0, 0, 1, 0, 2])
        
        # Total TP = 4, Total FP = 1
        p = precision(y_true, y_pred, average='micro')
        np.testing.assert_almost_equal(p, 4/5)
    
    def test_per_class_precision(self):
        """Test per-class precision."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        
        p = precision(y_true, y_pred, average=None)
        
        # Class 0: TP=1, FP=0 -> 1.0
        # Class 1: TP=2, FP=1 -> 2/3
        np.testing.assert_almost_equal(p[0], 1.0)
        np.testing.assert_almost_equal(p[1], 2/3)


class TestRecall:
    """Tests for recall metric."""
    
    def test_perfect_recall(self):
        """Test 100% recall."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        assert recall(y_true, y_pred, average='macro') == 1.0
    
    def test_macro_recall(self):
        """Test macro-averaged recall."""
        y_true = np.array([0, 0, 1, 1, 2])
        y_pred = np.array([0, 0, 1, 0, 2])
        
        # Class 0: TP=2, FN=0 -> recall=1.0
        # Class 1: TP=1, FN=1 -> recall=0.5
        # Class 2: TP=1, FN=0 -> recall=1.0
        r = recall(y_true, y_pred, average='macro')
        np.testing.assert_almost_equal(r, 5/6)
    
    def test_per_class_recall(self):
        """Test per-class recall."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        
        r = recall(y_true, y_pred, average=None)
        
        # Class 0: TP=1, FN=1 -> 0.5
        # Class 1: TP=2, FN=0 -> 1.0
        np.testing.assert_almost_equal(r[0], 0.5)
        np.testing.assert_almost_equal(r[1], 1.0)


class TestF1Score:
    """Tests for F1 score metric."""
    
    def test_perfect_f1(self):
        """Test perfect F1 score."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        assert f1_score(y_true, y_pred, average='macro') == 1.0
    
    def test_f1_formula(self):
        """Test F1 = 2 * (P * R) / (P + R)."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        
        p = precision(y_true, y_pred, average='macro')
        r = recall(y_true, y_pred, average='macro')
        expected_f1 = 2 * p * r / (p + r)
        
        actual_f1 = f1_score(y_true, y_pred, average='macro')
        np.testing.assert_almost_equal(actual_f1, expected_f1)
    
    def test_per_class_f1(self):
        """Test per-class F1 scores."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        
        f1 = f1_score(y_true, y_pred, average=None)
        assert len(f1) == 2


class TestConfusionMatrix:
    """Tests for confusion matrix."""
    
    def test_perfect_classification(self):
        """Test perfect classification produces diagonal matrix."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        
        cm = confusion_matrix(y_true, y_pred)
        
        expected = np.array([[2, 0, 0],
                             [0, 2, 0],
                             [0, 0, 2]])
        np.testing.assert_array_equal(cm, expected)
    
    def test_misclassification(self):
        """Test confusion matrix with misclassifications."""
        y_true = np.array([0, 0, 1, 1, 2])
        y_pred = np.array([0, 1, 1, 1, 0])  # 2 errors
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Row = true, Col = pred
        # [0,0]=1 (true=0,pred=0), [0,1]=1 (true=0,pred=1)
        # [1,1]=2 (true=1,pred=1)
        # [2,0]=1 (true=2,pred=0)
        expected = np.array([[1, 1, 0],
                             [0, 2, 0],
                             [1, 0, 0]])
        np.testing.assert_array_equal(cm, expected)
    
    def test_explicit_num_classes(self):
        """Test with explicit number of classes."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        
        cm = confusion_matrix(y_true, y_pred, num_classes=5)
        
        assert cm.shape == (5, 5)
        assert cm[0, 0] == 1
        assert cm[1, 1] == 1
    
    def test_one_hot_input(self):
        """Test confusion matrix with one-hot encoded input."""
        y_true = np.array([[1, 0], [0, 1], [1, 0]])
        y_pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])
        
        cm = confusion_matrix(y_true, y_pred)
        
        # True: [0, 1, 0], Pred: [0, 1, 1]
        expected = np.array([[1, 1],
                             [0, 1]])
        np.testing.assert_array_equal(cm, expected)


class TestClassificationReport:
    """Tests for classification report."""
    
    def test_report_structure(self):
        """Test report contains expected keys."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        
        report = classification_report(y_true, y_pred)
        
        assert 'per_class' in report
        assert 'accuracy' in report
        assert 'macro_avg' in report
        assert 'weighted_avg' in report
        assert 'total_samples' in report
    
    def test_per_class_metrics(self):
        """Test per-class metrics in report."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        
        report = classification_report(y_true, y_pred, ['A', 'B'])
        
        assert 'A' in report['per_class']
        assert 'B' in report['per_class']
        assert 'precision' in report['per_class']['A']
        assert 'recall' in report['per_class']['A']
        assert 'f1_score' in report['per_class']['A']
        assert 'support' in report['per_class']['A']
    
    def test_accuracy_in_report(self):
        """Test accuracy is calculated correctly."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])  # 3/4 correct
        
        report = classification_report(y_true, y_pred)
        
        assert report['accuracy'] == 0.75
        assert report['total_samples'] == 4


class TestTopKAccuracy:
    """Tests for top-k accuracy."""
    
    def test_top_1_accuracy(self):
        """Test top-1 accuracy equals standard accuracy."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([[0.9, 0.05, 0.05],
                          [0.1, 0.8, 0.1],
                          [0.1, 0.1, 0.8]])
        
        top1 = top_k_accuracy(y_true, y_pred, k=1)
        assert top1 == 1.0
    
    def test_top_2_accuracy(self):
        """Test top-2 accuracy."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([[0.4, 0.5, 0.1],   # true=0, top-2=[1,0] - correct
                          [0.3, 0.5, 0.2],    # true=1, top-2=[1,0] - correct
                          [0.4, 0.35, 0.25]])  # true=2, top-2=[0,1] - wrong
        
        top2 = top_k_accuracy(y_true, y_pred, k=2)
        np.testing.assert_almost_equal(top2, 2/3)
    
    def test_top_k_with_one_hot(self):
        """Test top-k with one-hot encoded labels."""
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_pred = np.array([[0.3, 0.5, 0.2], [0.1, 0.8, 0.1]])
        
        top2 = top_k_accuracy(y_true, y_pred, k=2)
        assert top2 == 1.0


class TestLogLoss:
    """Tests for log loss (cross-entropy)."""
    
    def test_perfect_predictions(self):
        """Test log loss with near-perfect predictions."""
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[0.999, 0.001], [0.001, 0.999]])
        
        loss = log_loss(y_true, y_pred)
        assert loss < 0.01
    
    def test_poor_predictions(self):
        """Test log loss with poor predictions."""
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[0.1, 0.9], [0.9, 0.1]])
        
        loss = log_loss(y_true, y_pred)
        assert loss > 2.0
    
    def test_integer_labels(self):
        """Test log loss with integer labels."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([[0.9, 0.05, 0.05],
                          [0.05, 0.9, 0.05],
                          [0.05, 0.05, 0.9]])
        
        loss = log_loss(y_true, y_pred)
        assert loss < 0.2


class TestPerClassAccuracy:
    """Tests for per-class accuracy."""
    
    def test_all_classes_correct(self):
        """Test perfect classification."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        
        acc = per_class_accuracy(y_true, y_pred)
        
        np.testing.assert_array_equal(acc, [1.0, 1.0, 1.0])
    
    def test_varying_accuracy(self):
        """Test different accuracy per class."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 0, 2, 0])  # class 0: 100%, class 1: 50%, class 2: 50%
        
        acc = per_class_accuracy(y_true, y_pred)
        
        np.testing.assert_almost_equal(acc[0], 1.0)
        np.testing.assert_almost_equal(acc[1], 0.5)
        np.testing.assert_almost_equal(acc[2], 0.5)
    
    def test_explicit_num_classes(self):
        """Test with explicit number of classes."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        
        acc = per_class_accuracy(y_true, y_pred, num_classes=5)
        
        assert len(acc) == 5
        assert acc[0] == 1.0
        assert acc[1] == 1.0
        assert acc[2] == 0.0  # No samples, defaults to 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
