# 🧪 Testing Strategy - Digit Recognition

**Version**: 1.0  
**Date**: 1 Feb 2026  
**Status**: Planning

---

## 1. Overview

### 1.1 Testing Philosophy

Testing adalah bagian integral dari pengembangan untuk memastikan:

- **Correctness** - Algoritma bekerja sesuai matematika
- **Reliability** - Aplikasi stabil dan bebas bug
- **Performance** - Memenuhi target speed dan accuracy
- **Usability** - User experience yang baik

### 1.2 Testing Pyramid

```
                    ┌───────────┐
                    │   E2E     │  ← Few, slow, high coverage
                    │  Tests    │
                    ├───────────┤
                    │Integration│  ← Medium amount
                    │  Tests    │
                    ├───────────┤
                    │           │
                    │   Unit    │  ← Many, fast, focused
                    │   Tests   │
                    └───────────┘
```

### 1.3 Testing Goals

| Goal                       | Target   | Priority |
| -------------------------- | -------- | -------- |
| Code Coverage              | ≥ 80%    | P0       |
| Unit Test Pass Rate        | 100%     | P0       |
| Integration Test Pass Rate | 100%     | P0       |
| Model Accuracy             | ≥ 97%    | P0       |
| Performance Benchmarks     | All pass | P1       |

---

## 2. Test Categories

### 2.1 Test Types Overview

| Type            | Scope                 | Speed  | When to Run  |
| --------------- | --------------------- | ------ | ------------ |
| **Unit**        | Single function/class | Fast   | Every commit |
| **Integration** | Multiple components   | Medium | Every PR     |
| **Performance** | Speed & memory        | Slow   | Pre-release  |
| **Accuracy**    | Model quality         | Slow   | Pre-release  |
| **E2E**         | Full application      | Slow   | Pre-release  |

### 2.2 Test Organization

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── unit/
│   ├── __init__.py
│   ├── test_layers.py       # Layer unit tests
│   ├── test_activations.py  # Activation functions
│   ├── test_loss.py         # Loss functions
│   ├── test_optimizers.py   # Optimizers
│   ├── test_network.py      # Network class
│   └── test_preprocessing.py # Preprocessing
├── integration/
│   ├── __init__.py
│   ├── test_training.py     # Training pipeline
│   ├── test_prediction.py   # Prediction pipeline
│   └── test_data_flow.py    # Data flow integration
├── performance/
│   ├── __init__.py
│   ├── test_speed.py        # Speed benchmarks
│   └── test_memory.py       # Memory usage
├── accuracy/
│   ├── __init__.py
│   └── test_model_accuracy.py
└── e2e/
    ├── __init__.py
    └── test_full_workflow.py
```

---

## 3. Unit Tests

### 3.1 Layer Tests

```python
# tests/unit/test_layers.py

import pytest
import numpy as np
from core.layers import DenseLayer

class TestDenseLayer:
    """Unit tests for Dense/Fully Connected Layer"""

    @pytest.fixture
    def layer(self):
        """Create a standard dense layer for testing"""
        np.random.seed(42)  # For reproducibility
        return DenseLayer(input_size=4, output_size=3)

    def test_initialization_shapes(self, layer):
        """Test that weights and bias have correct shapes"""
        assert layer.weights.shape == (4, 3)
        assert layer.bias.shape == (1, 3)

    def test_forward_pass_shape(self, layer):
        """Test forward pass output shape"""
        x = np.random.randn(10, 4)  # batch of 10, 4 features
        output = layer.forward(x)

        assert output.shape == (10, 3)

    def test_forward_pass_computation(self):
        """Test forward pass mathematical correctness"""
        layer = DenseLayer(2, 2)
        layer.weights = np.array([[1, 2], [3, 4]])
        layer.bias = np.array([[0.5, 0.5]])

        x = np.array([[1, 1]])
        output = layer.forward(x)

        # Expected: x @ W + b = [1,1] @ [[1,2],[3,4]] + [0.5, 0.5]
        #         = [4, 6] + [0.5, 0.5] = [4.5, 6.5]
        expected = np.array([[4.5, 6.5]])
        np.testing.assert_array_almost_equal(output, expected)

    def test_backward_pass_shape(self, layer):
        """Test backward pass gradient shapes"""
        x = np.random.randn(10, 4)
        layer.forward(x)

        grad = np.random.randn(10, 3)
        grad_input = layer.backward(grad)

        assert grad_input.shape == (10, 4)
        assert layer.dW.shape == (4, 3)
        assert layer.db.shape == (1, 3)

    def test_gradient_computation(self):
        """Test gradient computation correctness using numerical gradient"""
        layer = DenseLayer(3, 2)
        x = np.random.randn(5, 3)

        # Forward pass
        output = layer.forward(x)

        # Backward pass with unit gradient
        grad = np.ones_like(output)
        layer.backward(grad)

        # Numerical gradient check
        epsilon = 1e-5
        numerical_dW = np.zeros_like(layer.weights)

        for i in range(layer.weights.shape[0]):
            for j in range(layer.weights.shape[1]):
                layer.weights[i, j] += epsilon
                out_plus = layer.forward(x)

                layer.weights[i, j] -= 2 * epsilon
                out_minus = layer.forward(x)

                layer.weights[i, j] += epsilon  # Restore

                numerical_dW[i, j] = np.sum(out_plus - out_minus) / (2 * epsilon)

        # Compare numerical and analytical gradients
        np.testing.assert_array_almost_equal(
            layer.dW, numerical_dW / x.shape[0], decimal=5
        )

    def test_weight_initialization_he(self):
        """Test He initialization produces correct variance"""
        np.random.seed(42)
        layer = DenseLayer(1000, 500, initializer='he')

        # He initialization: std = sqrt(2/n_in)
        expected_std = np.sqrt(2.0 / 1000)
        actual_std = np.std(layer.weights)

        # Allow 10% tolerance
        assert abs(actual_std - expected_std) / expected_std < 0.1
```

### 3.2 Activation Function Tests

```python
# tests/unit/test_activations.py

import pytest
import numpy as np
from core.activations import ReLULayer, SigmoidLayer, SoftmaxLayer

class TestReLU:
    """Unit tests for ReLU activation"""

    def test_forward_positive(self):
        """Test ReLU passes through positive values"""
        relu = ReLULayer()
        x = np.array([[1, 2, 3]])
        output = relu.forward(x)

        np.testing.assert_array_equal(output, x)

    def test_forward_negative(self):
        """Test ReLU zeros negative values"""
        relu = ReLULayer()
        x = np.array([[-1, -2, -3]])
        output = relu.forward(x)

        expected = np.array([[0, 0, 0]])
        np.testing.assert_array_equal(output, expected)

    def test_forward_mixed(self):
        """Test ReLU with mixed values"""
        relu = ReLULayer()
        x = np.array([[-2, -1, 0, 1, 2]])
        output = relu.forward(x)

        expected = np.array([[0, 0, 0, 1, 2]])
        np.testing.assert_array_equal(output, expected)

    def test_backward_gradient(self):
        """Test ReLU backward pass"""
        relu = ReLULayer()
        x = np.array([[-2, -1, 0, 1, 2]])
        relu.forward(x)

        grad = np.array([[1, 1, 1, 1, 1]])
        grad_input = relu.backward(grad)

        # Gradient is 1 where x > 0, 0 elsewhere
        expected = np.array([[0, 0, 0, 1, 1]])
        np.testing.assert_array_equal(grad_input, expected)


class TestSigmoid:
    """Unit tests for Sigmoid activation"""

    def test_forward_range(self):
        """Test sigmoid output is in (0, 1)"""
        sigmoid = SigmoidLayer()
        x = np.random.randn(100, 50)
        output = sigmoid.forward(x)

        assert np.all(output > 0)
        assert np.all(output < 1)

    def test_forward_known_values(self):
        """Test sigmoid at known points"""
        sigmoid = SigmoidLayer()
        x = np.array([[0]])
        output = sigmoid.forward(x)

        # sigmoid(0) = 0.5
        np.testing.assert_almost_equal(output[0, 0], 0.5)

    def test_backward_derivative(self):
        """Test sigmoid derivative: σ(x) * (1 - σ(x))"""
        sigmoid = SigmoidLayer()
        x = np.array([[0, 1, -1]])
        out = sigmoid.forward(x)

        grad = np.ones_like(out)
        grad_input = sigmoid.backward(grad)

        # Derivative is σ(x) * (1 - σ(x))
        expected = out * (1 - out)
        np.testing.assert_array_almost_equal(grad_input, expected)


class TestSoftmax:
    """Unit tests for Softmax activation"""

    def test_sum_to_one(self):
        """Test softmax outputs sum to 1"""
        softmax = SoftmaxLayer()
        x = np.random.randn(10, 5)
        output = softmax.forward(x)

        row_sums = np.sum(output, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(10))

    def test_positive_outputs(self):
        """Test all softmax outputs are positive"""
        softmax = SoftmaxLayer()
        x = np.random.randn(10, 5) * 10  # Large values
        output = softmax.forward(x)

        assert np.all(output > 0)

    def test_numerical_stability(self):
        """Test softmax is numerically stable with large inputs"""
        softmax = SoftmaxLayer()
        x = np.array([[1000, 1001, 1002]])  # Very large values
        output = softmax.forward(x)

        # Should not produce NaN or Inf
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

        # Should still sum to 1
        assert abs(np.sum(output) - 1.0) < 1e-6
```

### 3.3 Loss Function Tests

```python
# tests/unit/test_loss.py

import pytest
import numpy as np
from core.loss import CrossEntropyLoss, MSELoss

class TestCrossEntropyLoss:
    """Unit tests for Cross-Entropy Loss"""

    def test_perfect_prediction(self):
        """Test loss is low for correct predictions"""
        loss_fn = CrossEntropyLoss()

        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0.99, 0.005, 0.005]])

        loss = loss_fn(y_true, y_pred)

        # Loss should be close to 0 for perfect prediction
        assert loss < 0.02

    def test_wrong_prediction(self):
        """Test loss is high for wrong predictions"""
        loss_fn = CrossEntropyLoss()

        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0.01, 0.99, 0.00]])

        loss = loss_fn(y_true, y_pred)

        # Loss should be high for wrong prediction
        assert loss > 2.0

    def test_batch_computation(self):
        """Test loss computation with batch"""
        loss_fn = CrossEntropyLoss()

        y_true = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        y_pred = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9]
        ])

        loss = loss_fn(y_true, y_pred)

        # Should be average loss across batch
        assert isinstance(loss, float)
        assert loss > 0

    def test_gradient_shape(self):
        """Test gradient has correct shape"""
        loss_fn = CrossEntropyLoss()

        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0.9, 0.05, 0.05]])

        grad = loss_fn.gradient(y_true, y_pred)

        assert grad.shape == y_pred.shape

    def test_gradient_direction(self):
        """Test gradient points in right direction"""
        loss_fn = CrossEntropyLoss()

        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0.5, 0.3, 0.2]])

        grad = loss_fn.gradient(y_true, y_pred)

        # For softmax+CE, gradient = y_pred - y_true
        expected = y_pred - y_true
        np.testing.assert_array_almost_equal(grad, expected)


class TestMSELoss:
    """Unit tests for Mean Squared Error Loss"""

    def test_zero_loss_identical(self):
        """Test MSE is 0 when prediction equals target"""
        loss_fn = MSELoss()

        y_true = np.array([[1, 2, 3]])
        y_pred = np.array([[1, 2, 3]])

        loss = loss_fn(y_true, y_pred)

        np.testing.assert_almost_equal(loss, 0)

    def test_known_value(self):
        """Test MSE computation with known values"""
        loss_fn = MSELoss()

        y_true = np.array([[0, 0, 0]])
        y_pred = np.array([[1, 2, 3]])

        loss = loss_fn(y_true, y_pred)

        # MSE = mean([1, 4, 9]) = 14/3 ≈ 4.67
        expected = (1 + 4 + 9) / 3
        np.testing.assert_almost_equal(loss, expected)
```

### 3.4 Optimizer Tests

```python
# tests/unit/test_optimizers.py

import pytest
import numpy as np
from core.optimizers import SGD, SGDMomentum, Adam

class TestSGD:
    """Unit tests for SGD optimizer"""

    def test_update_direction(self):
        """Test SGD updates in negative gradient direction"""
        sgd = SGD(learning_rate=0.1)

        params = {'weights': np.array([1.0, 2.0, 3.0])}
        grads = {'weights': np.array([1.0, 1.0, 1.0])}

        sgd.update(params, grads)

        # weights should decrease by lr * grad
        expected = np.array([0.9, 1.9, 2.9])
        np.testing.assert_array_almost_equal(params['weights'], expected)

    def test_learning_rate_effect(self):
        """Test different learning rates"""
        params1 = {'weights': np.array([1.0])}
        params2 = {'weights': np.array([1.0])}
        grads = {'weights': np.array([1.0])}

        sgd_small = SGD(learning_rate=0.01)
        sgd_large = SGD(learning_rate=0.1)

        sgd_small.update(params1, grads)
        sgd_large.update(params2, grads)

        # Larger LR should move more
        assert abs(params2['weights'][0] - 1.0) > abs(params1['weights'][0] - 1.0)


class TestAdam:
    """Unit tests for Adam optimizer"""

    def test_first_update(self):
        """Test Adam first update works correctly"""
        adam = Adam(learning_rate=0.001)

        params = {'weights': np.array([1.0, 2.0])}
        grads = {'weights': np.array([0.1, 0.2])}

        adam.update(params, grads, layer_id=0)

        # Weights should change
        assert not np.allclose(params['weights'], [1.0, 2.0])

    def test_momentum_accumulation(self):
        """Test Adam accumulates momentum correctly"""
        adam = Adam(learning_rate=0.001, beta1=0.9)

        params = {'weights': np.array([1.0])}
        grads = {'weights': np.array([1.0])}

        # Multiple updates with same gradient
        for _ in range(10):
            adam.update(params, grads, layer_id=0)

        # After multiple updates, momentum should build up
        assert adam.t == 10

    def test_bias_correction(self):
        """Test Adam bias correction is applied"""
        adam = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)

        params = {'weights': np.array([0.0])}
        grads = {'weights': np.array([1.0])}

        # First update should have significant bias correction
        adam.update(params, grads, layer_id=0)

        # Weight change should be larger than raw gradient * lr
        # due to bias correction
        assert abs(params['weights'][0]) > 0.001
```

---

## 4. Integration Tests

### 4.1 Training Pipeline Tests

```python
# tests/integration/test_training.py

import pytest
import numpy as np
from core.network import NeuralNetwork, NetworkBuilder
from core.trainer import Trainer
from core.optimizers import Adam
from core.loss import CrossEntropyLoss

class TestTrainingPipeline:
    """Integration tests for complete training pipeline"""

    @pytest.fixture
    def simple_dataset(self):
        """Create simple XOR-like dataset"""
        np.random.seed(42)

        # Simple 2-class classification
        X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ], dtype=np.float32)

        y = np.array([
            [1, 0],  # Class 0
            [0, 1],  # Class 1
            [0, 1],  # Class 1
            [1, 0]   # Class 0
        ], dtype=np.float32)

        return X, y

    @pytest.fixture
    def network(self):
        """Create simple network for testing"""
        return NetworkBuilder() \
            .input(2) \
            .dense(8, activation='relu') \
            .dense(2, activation='softmax') \
            .build()

    def test_training_reduces_loss(self, simple_dataset, network):
        """Test that training reduces loss"""
        X, y = simple_dataset

        optimizer = Adam(learning_rate=0.1)
        loss_fn = CrossEntropyLoss()
        trainer = Trainer(network, optimizer, loss_fn)

        # Initial loss
        initial_pred = network.forward(X)
        initial_loss = loss_fn(y, initial_pred)

        # Train
        history = trainer.train(X, y, epochs=100, batch_size=4)

        # Final loss
        final_loss = history['train_loss'][-1]

        assert final_loss < initial_loss

    def test_training_improves_accuracy(self, simple_dataset, network):
        """Test that training improves accuracy"""
        X, y = simple_dataset

        optimizer = Adam(learning_rate=0.1)
        loss_fn = CrossEntropyLoss()
        trainer = Trainer(network, optimizer, loss_fn)

        history = trainer.train(X, y, epochs=200, batch_size=4)

        final_accuracy = history['train_acc'][-1]

        # Should achieve high accuracy on simple dataset
        assert final_accuracy > 0.9

    def test_validation_evaluation(self, network):
        """Test validation evaluation during training"""
        np.random.seed(42)

        # Larger dataset with train/val split
        X = np.random.randn(100, 2)
        y = np.zeros((100, 2))
        y[np.arange(100), (X[:, 0] > 0).astype(int)] = 1

        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        optimizer = Adam(learning_rate=0.01)
        loss_fn = CrossEntropyLoss()
        trainer = Trainer(network, optimizer, loss_fn)

        history = trainer.train(
            X_train, y_train,
            X_val, y_val,
            epochs=50,
            batch_size=16
        )

        assert 'val_loss' in history
        assert 'val_acc' in history
        assert len(history['val_loss']) == 50


class TestMNISTTraining:
    """Integration tests with actual MNIST data"""

    @pytest.fixture
    def mnist_subset(self):
        """Load small subset of MNIST for testing"""
        from data.mnist_loader import MNISTLoader

        loader = MNISTLoader()
        X_train, y_train, X_test, y_test = loader.load()

        # Use only 1000 samples for speed
        return X_train[:1000], y_train[:1000], X_test[:200], y_test[:200]

    def test_mnist_training_convergence(self, mnist_subset):
        """Test model can learn from MNIST"""
        X_train, y_train, X_test, y_test = mnist_subset

        network = NetworkBuilder() \
            .input(784) \
            .dense(64, activation='relu') \
            .dense(10, activation='softmax') \
            .build()

        optimizer = Adam(learning_rate=0.001)
        loss_fn = CrossEntropyLoss()
        trainer = Trainer(network, optimizer, loss_fn)

        history = trainer.train(X_train, y_train, epochs=10, batch_size=32)

        # Should show learning
        assert history['train_loss'][-1] < history['train_loss'][0]
        assert history['train_acc'][-1] > 0.8
```

### 4.2 Prediction Pipeline Tests

```python
# tests/integration/test_prediction.py

import pytest
import numpy as np
from core.network import NetworkBuilder
from preprocessing.pipeline import PreprocessingPipeline
from PIL import Image

class TestPredictionPipeline:
    """Integration tests for prediction pipeline"""

    @pytest.fixture
    def trained_network(self):
        """Create and minimally train a network"""
        from data.mnist_loader import MNISTLoader
        from core.trainer import Trainer
        from core.optimizers import Adam
        from core.loss import CrossEntropyLoss

        # Load data
        loader = MNISTLoader()
        X_train, y_train, _, _ = loader.load()

        # Create and train network
        network = NetworkBuilder() \
            .input(784) \
            .dense(64, activation='relu') \
            .dense(10, activation='softmax') \
            .build()

        trainer = Trainer(network, Adam(0.001), CrossEntropyLoss())
        trainer.train(X_train[:1000], y_train[:1000], epochs=5, batch_size=32)

        return network

    def test_canvas_to_prediction(self, trained_network):
        """Test full pipeline from canvas image to prediction"""
        # Simulate canvas image (white digit on black background)
        canvas_image = np.zeros((280, 280), dtype=np.uint8)
        # Draw a simple "1" shape
        canvas_image[50:230, 130:150] = 255

        # Preprocess
        pipeline = PreprocessingPipeline()
        processed = pipeline.process_canvas(canvas_image)

        assert processed.shape == (784,)

        # Predict
        prediction = trained_network.forward(processed.reshape(1, -1))

        assert prediction.shape == (1, 10)
        assert abs(np.sum(prediction) - 1.0) < 1e-6

    def test_image_file_to_prediction(self, trained_network, tmp_path):
        """Test prediction from image file"""
        # Create test image
        img = Image.new('L', (100, 100), color=0)
        # Draw something
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([30, 10, 70, 90], fill=255)

        # Save to temp file
        filepath = tmp_path / "test_digit.png"
        img.save(filepath)

        # Load and predict
        pipeline = PreprocessingPipeline()
        processed = pipeline.process_image_file(str(filepath))

        prediction = trained_network.forward(processed.reshape(1, -1))

        predicted_digit = np.argmax(prediction)
        assert 0 <= predicted_digit <= 9
```

---

## 5. Performance Tests

### 5.1 Speed Benchmarks

```python
# tests/performance/test_speed.py

import pytest
import numpy as np
import time
from core.network import NetworkBuilder

class TestSpeedBenchmarks:
    """Performance tests for speed requirements"""

    @pytest.fixture
    def network(self):
        """Create standard digit recognition network"""
        return NetworkBuilder() \
            .input(784) \
            .dense(128, activation='relu') \
            .dense(64, activation='relu') \
            .dense(10, activation='softmax') \
            .build()

    def test_single_inference_speed(self, network):
        """Test single image inference time < 50ms"""
        x = np.random.randn(1, 784)

        # Warm up
        for _ in range(10):
            network.forward(x)

        # Measure
        times = []
        for _ in range(100):
            start = time.perf_counter()
            network.forward(x)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        avg_time = np.mean(times)

        assert avg_time < 50, f"Inference too slow: {avg_time:.2f}ms"

    def test_batch_inference_speed(self, network):
        """Test batch inference maintains efficiency"""
        batch_sizes = [1, 10, 32, 100]

        for batch_size in batch_sizes:
            x = np.random.randn(batch_size, 784)

            start = time.perf_counter()
            network.forward(x)
            end = time.perf_counter()

            time_per_sample = (end - start) * 1000 / batch_size

            # Should be faster per sample with larger batches
            assert time_per_sample < 10, \
                f"Batch {batch_size}: {time_per_sample:.2f}ms per sample"

    def test_training_epoch_speed(self):
        """Test training epoch completes in reasonable time"""
        from core.trainer import Trainer
        from core.optimizers import Adam
        from core.loss import CrossEntropyLoss

        # Create data
        X = np.random.randn(1000, 784)
        y = np.zeros((1000, 10))
        y[np.arange(1000), np.random.randint(0, 10, 1000)] = 1

        network = NetworkBuilder() \
            .input(784) \
            .dense(128, activation='relu') \
            .dense(64, activation='relu') \
            .dense(10, activation='softmax') \
            .build()

        trainer = Trainer(network, Adam(0.001), CrossEntropyLoss())

        start = time.perf_counter()
        trainer.train(X, y, epochs=1, batch_size=32)
        end = time.perf_counter()

        epoch_time = end - start

        # One epoch on 1000 samples should be < 5 seconds
        assert epoch_time < 5, f"Epoch too slow: {epoch_time:.2f}s"


class TestMemoryUsage:
    """Tests for memory usage"""

    def test_memory_during_training(self):
        """Test memory usage stays within bounds during training"""
        import tracemalloc

        tracemalloc.start()

        # Create large-ish network
        network = NetworkBuilder() \
            .input(784) \
            .dense(256, activation='relu') \
            .dense(128, activation='relu') \
            .dense(10, activation='softmax') \
            .build()

        # Training data
        X = np.random.randn(10000, 784).astype(np.float32)
        y = np.zeros((10000, 10), dtype=np.float32)
        y[np.arange(10000), np.random.randint(0, 10, 10000)] = 1

        from core.trainer import Trainer
        from core.optimizers import Adam
        from core.loss import CrossEntropyLoss

        trainer = Trainer(network, Adam(0.001), CrossEntropyLoss())
        trainer.train(X, y, epochs=2, batch_size=32)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024

        # Peak memory should be < 500 MB
        assert peak_mb < 500, f"Peak memory too high: {peak_mb:.1f}MB"
```

---

## 6. Accuracy Tests

### 6.1 Model Accuracy Tests

```python
# tests/accuracy/test_model_accuracy.py

import pytest
import numpy as np

class TestModelAccuracy:
    """Tests for model accuracy requirements"""

    @pytest.fixture(scope="class")
    def trained_model(self):
        """Train model on full MNIST (cached for class)"""
        from data.mnist_loader import MNISTLoader
        from core.network import NetworkBuilder
        from core.trainer import Trainer
        from core.optimizers import Adam
        from core.loss import CrossEntropyLoss

        loader = MNISTLoader()
        X_train, y_train, X_test, y_test = loader.load()

        network = NetworkBuilder() \
            .input(784) \
            .dense(128, activation='relu') \
            .dense(64, activation='relu') \
            .dense(10, activation='softmax') \
            .build()

        trainer = Trainer(network, Adam(0.001), CrossEntropyLoss())
        trainer.train(X_train, y_train, epochs=20, batch_size=32)

        return network, X_test, y_test

    def test_overall_accuracy(self, trained_model):
        """Test model achieves >= 97% accuracy on test set"""
        network, X_test, y_test = trained_model

        predictions = network.forward(X_test)
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y_test, axis=1)

        accuracy = np.mean(pred_labels == true_labels)

        assert accuracy >= 0.97, f"Accuracy {accuracy:.2%} below 97% target"

    def test_per_digit_accuracy(self, trained_model):
        """Test accuracy for each digit is acceptable"""
        network, X_test, y_test = trained_model

        predictions = network.forward(X_test)
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y_test, axis=1)

        for digit in range(10):
            mask = true_labels == digit
            digit_accuracy = np.mean(pred_labels[mask] == digit)

            # Each digit should have at least 90% accuracy
            assert digit_accuracy >= 0.90, \
                f"Digit {digit} accuracy {digit_accuracy:.2%} too low"

    def test_confidence_calibration(self, trained_model):
        """Test that high confidence predictions are more accurate"""
        network, X_test, y_test = trained_model

        predictions = network.forward(X_test)
        pred_labels = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        true_labels = np.argmax(y_test, axis=1)

        # High confidence (>95%) predictions
        high_conf_mask = confidences > 0.95
        high_conf_accuracy = np.mean(
            pred_labels[high_conf_mask] == true_labels[high_conf_mask]
        )

        # Low confidence (<50%) predictions
        low_conf_mask = confidences < 0.50
        if np.sum(low_conf_mask) > 0:
            low_conf_accuracy = np.mean(
                pred_labels[low_conf_mask] == true_labels[low_conf_mask]
            )

            # High confidence should be more accurate
            assert high_conf_accuracy > low_conf_accuracy

        # High confidence predictions should be very accurate
        assert high_conf_accuracy >= 0.98
```

---

## 7. Test Configuration

### 7.1 pytest Configuration

```python
# tests/conftest.py

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility"""
    np.random.seed(42)
    yield


@pytest.fixture(scope="session")
def mnist_data():
    """Load MNIST data once per test session"""
    from data.mnist_loader import MNISTLoader

    loader = MNISTLoader(data_dir='data/mnist')
    return loader.load()


@pytest.fixture
def small_network():
    """Small network for quick tests"""
    from core.network import NetworkBuilder

    return NetworkBuilder() \
        .input(784) \
        .dense(32, activation='relu') \
        .dense(10, activation='softmax') \
        .build()


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
```

### 7.2 pytest.ini

```ini
# pytest.ini

[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests requiring GPU
    integration: marks integration tests
    performance: marks performance tests

# Default options
addopts =
    -v
    --tb=short
    --strict-markers
    -ra

# Coverage settings
[coverage:run]
source = core,preprocessing,data,gui
omit = tests/*

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
```

---

## 8. Running Tests

### 8.1 Basic Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_layers.py

# Run specific test class
pytest tests/unit/test_layers.py::TestDenseLayer

# Run specific test function
pytest tests/unit/test_layers.py::TestDenseLayer::test_forward_pass_shape

# Run tests matching pattern
pytest -k "forward"

# Run excluding slow tests
pytest -m "not slow"
```

### 8.2 Coverage Commands

```bash
# Run with coverage
pytest --cov=core --cov=preprocessing

# Generate HTML coverage report
pytest --cov=core --cov-report=html

# Show missing lines
pytest --cov=core --cov-report=term-missing
```

### 8.3 CI/CD Integration

```yaml
# .github/workflows/tests.yml

name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: |
          pytest --cov=core --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## 9. Test Coverage Goals

### 9.1 Coverage Targets

| Module                | Target Coverage | Priority |
| --------------------- | --------------- | -------- |
| `core/layers.py`      | 95%             | P0       |
| `core/activations.py` | 100%            | P0       |
| `core/loss.py`        | 100%            | P0       |
| `core/optimizers.py`  | 90%             | P0       |
| `core/network.py`     | 90%             | P0       |
| `core/trainer.py`     | 85%             | P1       |
| `preprocessing/`      | 80%             | P1       |
| `data/`               | 80%             | P1       |
| `gui/`                | 60%             | P2       |

### 9.2 Current Coverage Report Template

```
---------- coverage: ----------
Name                          Stmts   Miss  Cover
-------------------------------------------------
core/activations.py              45      0   100%
core/layers.py                   82      4    95%
core/loss.py                     38      0   100%
core/network.py                  65      7    89%
core/optimizers.py               78      8    90%
core/trainer.py                 102     15    85%
preprocessing/pipeline.py        56      8    86%
data/mnist_loader.py             48     10    79%
-------------------------------------------------
TOTAL                           514     52    90%
```

---

**Document Status**: ✅ Complete  
**Related Documents**:

- [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md)
- [PERFORMANCE_BENCHMARKS.md](PERFORMANCE_BENCHMARKS.md)
