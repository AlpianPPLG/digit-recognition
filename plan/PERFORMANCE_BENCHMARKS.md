# 📊 Performance Benchmarks - Digit Recognition

**Version**: 1.0  
**Date**: 1 Feb 2026  
**Status**: Planning

---

## 1. Overview

### 1.1 Tujuan Benchmarking

Dokumen ini mendefinisikan target performance dan metodologi pengukuran untuk:

- **Model Accuracy** - Akurasi prediksi digit
- **Speed Performance** - Waktu training dan inference
- **Memory Efficiency** - Penggunaan memori
- **Scalability** - Performa dengan berbagai ukuran data

### 1.2 Key Performance Indicators

| KPI                        | Target  | Minimum  | Priority |
| -------------------------- | ------- | -------- | -------- |
| Test Accuracy              | ≥ 97%   | 95%      | P0       |
| Single Inference           | < 50ms  | < 100ms  | P0       |
| Training Time (full MNIST) | < 5 min | < 10 min | P1       |
| Memory Usage               | < 500MB | < 1GB    | P1       |
| Model Size                 | < 5MB   | < 10MB   | P2       |

---

## 2. Accuracy Benchmarks

### 2.1 Overall Accuracy Targets

```
┌──────────────────────────────────────────────────────────────┐
│                    ACCURACY TARGETS                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Target:   ████████████████████████████████████████ 97%     │
│                                                              │
│  Minimum:  ██████████████████████████████████████░░ 95%     │
│                                                              │
│  Baseline: ████████████████████████████████░░░░░░░░ 80%     │
│  (untrained)                                                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Per-Digit Accuracy Requirements

| Digit | Target | Minimum | Common Confusion |
| ----- | ------ | ------- | ---------------- |
| 0     | 98%    | 95%     | 6                |
| 1     | 99%    | 97%     | 7                |
| 2     | 97%    | 94%     | 7                |
| 3     | 96%    | 93%     | 5, 8             |
| 4     | 97%    | 94%     | 9                |
| 5     | 96%    | 93%     | 3, 6             |
| 6     | 97%    | 94%     | 0, 5             |
| 7     | 98%    | 95%     | 1, 2             |
| 8     | 96%    | 93%     | 3, 5             |
| 9     | 97%    | 94%     | 4, 7             |

### 2.3 Confusion Matrix Target

```python
# Target confusion matrix (normalized, per row)
# Each row should have >95% on diagonal

#    Predicted
#    0    1    2    3    4    5    6    7    8    9
# 0 [.98  .00  .00  .00  .00  .00  .01  .00  .01  .00]
# 1 [.00  .99  .00  .00  .00  .00  .00  .01  .00  .00]
# 2 [.00  .01  .97  .00  .00  .00  .00  .01  .01  .00]
# 3 [.00  .00  .01  .96  .00  .02  .00  .00  .01  .00]
# 4 [.00  .00  .00  .00  .97  .00  .00  .00  .00  .03]
# 5 [.00  .00  .00  .02  .00  .96  .01  .00  .01  .00]
# 6 [.01  .00  .00  .00  .00  .01  .97  .00  .01  .00]
# 7 [.00  .01  .01  .00  .00  .00  .00  .98  .00  .00]
# 8 [.01  .00  .01  .01  .00  .01  .00  .00  .96  .00]
# 9 [.00  .00  .00  .00  .02  .00  .00  .01  .00  .97]
```

### 2.4 Accuracy Benchmarking Script

```python
# benchmark/accuracy_benchmark.py

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class AccuracyBenchmark:
    """Accuracy benchmarking for digit recognition model"""

    overall_accuracy: float
    per_digit_accuracy: Dict[int, float]
    confusion_matrix: np.ndarray
    precision: Dict[int, float]
    recall: Dict[int, float]
    f1_scores: Dict[int, float]

def benchmark_accuracy(network, X_test: np.ndarray, y_test: np.ndarray) -> AccuracyBenchmark:
    """
    Comprehensive accuracy benchmarking

    Args:
        network: Trained neural network
        X_test: Test images (N, 784)
        y_test: Test labels one-hot (N, 10)

    Returns:
        AccuracyBenchmark with detailed metrics
    """
    # Get predictions
    predictions = network.forward(X_test)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    # Overall accuracy
    overall_accuracy = np.mean(pred_labels == true_labels)

    # Per-digit accuracy
    per_digit_accuracy = {}
    for digit in range(10):
        mask = true_labels == digit
        if np.sum(mask) > 0:
            per_digit_accuracy[digit] = np.mean(pred_labels[mask] == digit)
        else:
            per_digit_accuracy[digit] = 0.0

    # Confusion matrix
    confusion_matrix = np.zeros((10, 10), dtype=np.int32)
    for true, pred in zip(true_labels, pred_labels):
        confusion_matrix[true, pred] += 1

    # Precision, Recall, F1
    precision = {}
    recall = {}
    f1_scores = {}

    for digit in range(10):
        tp = confusion_matrix[digit, digit]
        fp = np.sum(confusion_matrix[:, digit]) - tp
        fn = np.sum(confusion_matrix[digit, :]) - tp

        precision[digit] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[digit] = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision[digit] + recall[digit] > 0:
            f1_scores[digit] = 2 * (precision[digit] * recall[digit]) / \
                              (precision[digit] + recall[digit])
        else:
            f1_scores[digit] = 0

    return AccuracyBenchmark(
        overall_accuracy=overall_accuracy,
        per_digit_accuracy=per_digit_accuracy,
        confusion_matrix=confusion_matrix,
        precision=precision,
        recall=recall,
        f1_scores=f1_scores
    )

def verify_accuracy_targets(benchmark: AccuracyBenchmark) -> Dict[str, bool]:
    """
    Verify if benchmark meets target requirements

    Returns:
        Dictionary of passed/failed checks
    """
    results = {}

    # Overall accuracy target
    results['overall_accuracy_97'] = benchmark.overall_accuracy >= 0.97
    results['overall_accuracy_95'] = benchmark.overall_accuracy >= 0.95

    # Per-digit minimum
    results['all_digits_above_90'] = all(
        acc >= 0.90 for acc in benchmark.per_digit_accuracy.values()
    )

    # Average F1 score
    avg_f1 = np.mean(list(benchmark.f1_scores.values()))
    results['avg_f1_above_95'] = avg_f1 >= 0.95

    return results

def print_accuracy_report(benchmark: AccuracyBenchmark) -> None:
    """Print formatted accuracy report"""

    print("=" * 60)
    print("ACCURACY BENCHMARK REPORT")
    print("=" * 60)
    print()

    # Overall
    print(f"Overall Accuracy: {benchmark.overall_accuracy:.2%}")
    status = "✅ PASS" if benchmark.overall_accuracy >= 0.97 else "❌ FAIL"
    print(f"Target (97%): {status}")
    print()

    # Per-digit
    print("Per-Digit Accuracy:")
    print("-" * 40)
    for digit in range(10):
        acc = benchmark.per_digit_accuracy[digit]
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        status = "✅" if acc >= 0.95 else "⚠️" if acc >= 0.90 else "❌"
        print(f"  Digit {digit}: {bar} {acc:.1%} {status}")
    print()

    # Confusion matrix
    print("Confusion Matrix:")
    print("-" * 40)
    print("      ", end="")
    for i in range(10):
        print(f"{i:5d}", end="")
    print()
    for i in range(10):
        print(f"  {i}:  ", end="")
        for j in range(10):
            val = benchmark.confusion_matrix[i, j]
            if i == j:
                print(f"\033[92m{val:5d}\033[0m", end="")  # Green diagonal
            elif val > 10:
                print(f"\033[91m{val:5d}\033[0m", end="")  # Red for errors
            else:
                print(f"{val:5d}", end="")
        print()
```

---

## 3. Speed Benchmarks

### 3.1 Inference Speed Targets

```
┌──────────────────────────────────────────────────────────────┐
│                    INFERENCE SPEED                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Single Image:                                               │
│  ├─ Target:    <  50ms ████████████████████░░░░░░░░░░░░░░░░ │
│  ├─ Acceptable: < 100ms ████████████████████████████████████ │
│  └─ Current:   ~  15ms ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│                                                              │
│  Batch (32 images):                                          │
│  ├─ Target:    < 200ms ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│  └─ Per image: <   7ms ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Training Speed Targets

| Phase         | Target      | Measurement             |
| ------------- | ----------- | ----------------------- |
| Data Loading  | < 5s        | Full MNIST (70k images) |
| Single Epoch  | < 15s       | 60k images, batch 32    |
| Full Training | < 5 min     | 20 epochs               |
| Convergence   | < 10 epochs | To 95% accuracy         |

### 3.3 Speed Benchmarking Script

```python
# benchmark/speed_benchmark.py

import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Callable
import gc

@dataclass
class TimingResult:
    """Single timing measurement result"""
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    samples: int

@dataclass
class SpeedBenchmark:
    """Complete speed benchmark results"""

    single_inference: TimingResult
    batch_inference: Dict[int, TimingResult]  # batch_size -> timing
    forward_pass: TimingResult
    backward_pass: TimingResult
    epoch_time: float
    full_training_time: float
    data_loading_time: float

def time_function(func: Callable, warmup: int = 5,
                  samples: int = 100) -> TimingResult:
    """
    Time a function with warmup and multiple samples

    Args:
        func: Function to time (no arguments)
        warmup: Number of warmup iterations
        samples: Number of timed iterations

    Returns:
        TimingResult with statistics
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Collect garbage before timing
    gc.collect()

    # Time samples
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return TimingResult(
        mean_ms=np.mean(times),
        std_ms=np.std(times),
        min_ms=np.min(times),
        max_ms=np.max(times),
        samples=samples
    )

def benchmark_inference_speed(network, input_size: int = 784) -> SpeedBenchmark:
    """
    Benchmark inference speed at various batch sizes

    Args:
        network: Neural network to benchmark
        input_size: Input feature size

    Returns:
        SpeedBenchmark with inference timings
    """
    results = {}

    # Single inference
    single_input = np.random.randn(1, input_size)
    single_result = time_function(
        lambda: network.forward(single_input),
        warmup=10,
        samples=100
    )

    # Batch inference at various sizes
    batch_results = {}
    for batch_size in [1, 8, 16, 32, 64, 128]:
        batch_input = np.random.randn(batch_size, input_size)
        batch_results[batch_size] = time_function(
            lambda: network.forward(batch_input),
            warmup=5,
            samples=50
        )

    return SpeedBenchmark(
        single_inference=single_result,
        batch_inference=batch_results,
        forward_pass=single_result,  # Same for this simple case
        backward_pass=None,  # Filled by training benchmark
        epoch_time=0,
        full_training_time=0,
        data_loading_time=0
    )

def benchmark_training_speed(network, X_train, y_train,
                            batch_size: int = 32) -> Dict:
    """
    Benchmark training speed

    Args:
        network: Network to train
        X_train: Training data
        y_train: Training labels
        batch_size: Batch size for training

    Returns:
        Dictionary with training timings
    """
    from core.trainer import Trainer
    from core.optimizers import Adam
    from core.loss import CrossEntropyLoss

    trainer = Trainer(network, Adam(0.001), CrossEntropyLoss())

    results = {}

    # Single epoch
    gc.collect()
    start = time.perf_counter()
    trainer.train(X_train, y_train, epochs=1, batch_size=batch_size, verbose=False)
    end = time.perf_counter()
    results['single_epoch'] = end - start

    # Forward + backward timing (one batch)
    batch_X = X_train[:batch_size]
    batch_y = y_train[:batch_size]

    def forward_backward():
        output = network.forward(batch_X)
        network.backward(batch_y - output)

    fb_result = time_function(forward_backward, warmup=5, samples=50)
    results['forward_backward_batch'] = fb_result.mean_ms

    return results

def print_speed_report(benchmark: SpeedBenchmark) -> None:
    """Print formatted speed benchmark report"""

    print("=" * 60)
    print("SPEED BENCHMARK REPORT")
    print("=" * 60)
    print()

    # Single inference
    print("Single Image Inference:")
    print("-" * 40)
    result = benchmark.single_inference
    target_status = "✅ PASS" if result.mean_ms < 50 else "❌ FAIL"
    print(f"  Mean:   {result.mean_ms:6.2f} ms  (target: < 50ms) {target_status}")
    print(f"  Std:    {result.std_ms:6.2f} ms")
    print(f"  Min:    {result.min_ms:6.2f} ms")
    print(f"  Max:    {result.max_ms:6.2f} ms")
    print()

    # Batch inference
    print("Batch Inference:")
    print("-" * 40)
    print(f"  {'Batch Size':>10} | {'Total (ms)':>10} | {'Per Image (ms)':>12}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*12}")
    for batch_size, result in sorted(benchmark.batch_inference.items()):
        per_image = result.mean_ms / batch_size
        print(f"  {batch_size:>10} | {result.mean_ms:>10.2f} | {per_image:>12.3f}")
    print()

    # Training
    if benchmark.epoch_time > 0:
        print("Training Speed:")
        print("-" * 40)
        print(f"  Single Epoch:   {benchmark.epoch_time:.2f} s")
        print(f"  Full Training:  {benchmark.full_training_time:.2f} s")
        print(f"  Data Loading:   {benchmark.data_loading_time:.2f} s")

def verify_speed_targets(benchmark: SpeedBenchmark) -> Dict[str, bool]:
    """Verify if benchmark meets speed targets"""

    results = {}

    # Single inference < 50ms
    results['single_inference_target'] = benchmark.single_inference.mean_ms < 50
    results['single_inference_acceptable'] = benchmark.single_inference.mean_ms < 100

    # Batch efficiency (per-image should decrease with batch size)
    if len(benchmark.batch_inference) >= 2:
        sizes = sorted(benchmark.batch_inference.keys())
        per_image_1 = benchmark.batch_inference[sizes[0]].mean_ms / sizes[0]
        per_image_n = benchmark.batch_inference[sizes[-1]].mean_ms / sizes[-1]
        results['batch_efficiency'] = per_image_n < per_image_1

    return results
```

---

## 4. Memory Benchmarks

### 4.1 Memory Usage Targets

```
┌──────────────────────────────────────────────────────────────┐
│                    MEMORY USAGE                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Model Parameters:                                           │
│  └─ Target: < 5 MB   ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│                                                              │
│  Runtime (Training):                                         │
│  └─ Target: < 500 MB ██████████████████████░░░░░░░░░░░░░░░░ │
│                                                              │
│  Runtime (Inference):                                        │
│  └─ Target: < 100 MB ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│                                                              │
│  Peak Memory:                                                │
│  └─ Target: < 1 GB   ██████████████████████████████████████ │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Memory Breakdown

| Component                    | Size    | Calculation              |
| ---------------------------- | ------- | ------------------------ |
| **Model Parameters**         |         |                          |
| ├─ Layer 1 (784→128)         | 401 KB  | 784×128×4 + 128×4        |
| ├─ Layer 2 (128→64)          | 33 KB   | 128×64×4 + 64×4          |
| └─ Layer 3 (64→10)           | 2.6 KB  | 64×10×4 + 10×4           |
| **Total Parameters**         | ~437 KB | 109,386 params × 4 bytes |
|                              |         |                          |
| **Training Data**            |         |                          |
| ├─ X_train (60k)             | 188 MB  | 60000×784×4              |
| └─ y_train (60k)             | 2.4 MB  | 60000×10×4               |
| **Total Training Data**      | ~190 MB |                          |
|                              |         |                          |
| **Intermediate Activations** |         |                          |
| └─ Per batch (32)            | ~400 KB | Activations + gradients  |

### 4.3 Memory Benchmarking Script

```python
# benchmark/memory_benchmark.py

import numpy as np
import tracemalloc
import gc
from dataclasses import dataclass
from typing import Dict

@dataclass
class MemoryBenchmark:
    """Memory usage benchmark results"""

    model_size_bytes: int
    parameter_count: int
    inference_peak_bytes: int
    training_peak_bytes: int
    data_memory_bytes: int

def count_parameters(network) -> int:
    """Count total trainable parameters in network"""
    total = 0
    for layer in network.layers:
        if hasattr(layer, 'weights'):
            total += layer.weights.size
        if hasattr(layer, 'bias'):
            total += layer.bias.size
    return total

def measure_model_size(network) -> int:
    """Measure model memory size in bytes"""
    total_bytes = 0
    for layer in network.layers:
        if hasattr(layer, 'weights'):
            total_bytes += layer.weights.nbytes
        if hasattr(layer, 'bias'):
            total_bytes += layer.bias.nbytes
    return total_bytes

def benchmark_memory(network, X_train, y_train, batch_size: int = 32) -> MemoryBenchmark:
    """
    Comprehensive memory benchmarking

    Args:
        network: Neural network to benchmark
        X_train: Training data
        y_train: Training labels
        batch_size: Batch size for training benchmark

    Returns:
        MemoryBenchmark with all measurements
    """
    # Model size
    model_size = measure_model_size(network)
    param_count = count_parameters(network)

    # Inference memory
    gc.collect()
    tracemalloc.start()

    for _ in range(10):
        network.forward(X_train[:1])

    _, inference_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Training memory
    from core.trainer import Trainer
    from core.optimizers import Adam
    from core.loss import CrossEntropyLoss

    gc.collect()
    tracemalloc.start()

    trainer = Trainer(network, Adam(0.001), CrossEntropyLoss())
    trainer.train(
        X_train[:1000], y_train[:1000],
        epochs=1, batch_size=batch_size, verbose=False
    )

    _, training_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Data memory
    data_memory = X_train.nbytes + y_train.nbytes

    return MemoryBenchmark(
        model_size_bytes=model_size,
        parameter_count=param_count,
        inference_peak_bytes=inference_peak,
        training_peak_bytes=training_peak,
        data_memory_bytes=data_memory
    )

def format_bytes(size: int) -> str:
    """Format bytes to human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"

def print_memory_report(benchmark: MemoryBenchmark) -> None:
    """Print formatted memory benchmark report"""

    print("=" * 60)
    print("MEMORY BENCHMARK REPORT")
    print("=" * 60)
    print()

    print("Model Size:")
    print("-" * 40)
    print(f"  Parameters:     {benchmark.parameter_count:,}")
    print(f"  Memory:         {format_bytes(benchmark.model_size_bytes)}")

    target = "✅" if benchmark.model_size_bytes < 5 * 1024 * 1024 else "❌"
    print(f"  Target (<5MB):  {target}")
    print()

    print("Runtime Memory:")
    print("-" * 40)
    print(f"  Inference Peak: {format_bytes(benchmark.inference_peak_bytes)}")
    print(f"  Training Peak:  {format_bytes(benchmark.training_peak_bytes)}")
    print(f"  Data Memory:    {format_bytes(benchmark.data_memory_bytes)}")
    print()

    # Memory efficiency
    print("Memory Efficiency:")
    print("-" * 40)
    bytes_per_param = benchmark.model_size_bytes / benchmark.parameter_count
    print(f"  Bytes per parameter: {bytes_per_param:.1f}")

def verify_memory_targets(benchmark: MemoryBenchmark) -> Dict[str, bool]:
    """Verify if benchmark meets memory targets"""

    MB = 1024 * 1024

    return {
        'model_size_under_5mb': benchmark.model_size_bytes < 5 * MB,
        'inference_under_100mb': benchmark.inference_peak_bytes < 100 * MB,
        'training_under_500mb': benchmark.training_peak_bytes < 500 * MB,
    }
```

---

## 5. Scalability Analysis

### 5.1 Scaling with Data Size

```python
# benchmark/scalability_benchmark.py

import numpy as np
import time
from typing import List, Dict

def benchmark_data_scaling(network_builder, data_sizes: List[int]) -> Dict:
    """
    Benchmark how training scales with data size

    Args:
        network_builder: Function to create fresh network
        data_sizes: List of training set sizes to test

    Returns:
        Dictionary with scaling results
    """
    from core.trainer import Trainer
    from core.optimizers import Adam
    from core.loss import CrossEntropyLoss

    results = {
        'sizes': data_sizes,
        'training_times': [],
        'epochs_to_converge': [],
        'final_accuracy': []
    }

    for size in data_sizes:
        # Generate synthetic data
        X = np.random.randn(size, 784).astype(np.float32)
        y = np.zeros((size, 10), dtype=np.float32)
        y[np.arange(size), np.random.randint(0, 10, size)] = 1

        # Fresh network
        network = network_builder()
        trainer = Trainer(network, Adam(0.001), CrossEntropyLoss())

        # Train and measure
        start = time.perf_counter()
        history = trainer.train(X, y, epochs=10, batch_size=32, verbose=False)
        end = time.perf_counter()

        results['training_times'].append(end - start)
        results['final_accuracy'].append(history['train_acc'][-1])

        # Find epoch where accuracy > 80%
        converge_epoch = 10
        for i, acc in enumerate(history['train_acc']):
            if acc > 0.8:
                converge_epoch = i + 1
                break
        results['epochs_to_converge'].append(converge_epoch)

    return results

def print_scaling_report(results: Dict) -> None:
    """Print scaling analysis report"""

    print("=" * 60)
    print("SCALABILITY ANALYSIS")
    print("=" * 60)
    print()

    print(f"{'Data Size':>12} | {'Train Time':>12} | {'Converge':>10} | {'Accuracy':>10}")
    print(f"{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")

    for i, size in enumerate(results['sizes']):
        time_s = results['training_times'][i]
        epochs = results['epochs_to_converge'][i]
        acc = results['final_accuracy'][i]

        print(f"{size:>12,} | {time_s:>10.2f}s | {epochs:>10} | {acc:>9.1%}")

    print()

    # Scaling analysis
    if len(results['sizes']) >= 2:
        time_ratio = results['training_times'][-1] / results['training_times'][0]
        size_ratio = results['sizes'][-1] / results['sizes'][0]
        scaling_factor = np.log(time_ratio) / np.log(size_ratio)

        print(f"Scaling Factor: O(n^{scaling_factor:.2f})")
        if scaling_factor < 1.2:
            print("✅ Near-linear scaling achieved")
        elif scaling_factor < 1.5:
            print("⚠️ Slightly super-linear scaling")
        else:
            print("❌ Poor scaling - optimization needed")
```

### 5.2 Scaling with Network Size

```python
def benchmark_network_scaling(layer_configs: List[List[int]],
                             X_train, y_train) -> Dict:
    """
    Benchmark how performance scales with network size

    Args:
        layer_configs: List of layer configurations, e.g., [[784, 64, 10], [784, 128, 64, 10]]
        X_train: Training data
        y_train: Training labels

    Returns:
        Dictionary with scaling results
    """
    from core.network import NetworkBuilder
    from core.trainer import Trainer
    from core.optimizers import Adam
    from core.loss import CrossEntropyLoss
    import tracemalloc

    results = {
        'configs': [],
        'param_counts': [],
        'training_times': [],
        'inference_times': [],
        'memory_usage': [],
        'accuracy': []
    }

    for config in layer_configs:
        # Build network
        builder = NetworkBuilder().input(config[0])
        for hidden_size in config[1:-1]:
            builder.dense(hidden_size, activation='relu')
        builder.dense(config[-1], activation='softmax')
        network = builder.build()

        # Count parameters
        param_count = sum(
            layer.weights.size + layer.bias.size
            for layer in network.layers
            if hasattr(layer, 'weights')
        )

        # Measure inference time
        single_input = X_train[:1]
        times = []
        for _ in range(100):
            start = time.perf_counter()
            network.forward(single_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        inference_time = np.mean(times)

        # Measure training time and memory
        tracemalloc.start()
        trainer = Trainer(network, Adam(0.001), CrossEntropyLoss())
        start = time.perf_counter()
        history = trainer.train(X_train[:5000], y_train[:5000],
                               epochs=5, batch_size=32, verbose=False)
        training_time = time.perf_counter() - start
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results['configs'].append(config)
        results['param_counts'].append(param_count)
        results['training_times'].append(training_time)
        results['inference_times'].append(inference_time)
        results['memory_usage'].append(peak_memory)
        results['accuracy'].append(history['train_acc'][-1])

    return results
```

---

## 6. Benchmark Automation

### 6.1 Complete Benchmark Suite

```python
# benchmark/run_all_benchmarks.py

import argparse
from datetime import datetime
import json

from accuracy_benchmark import benchmark_accuracy, print_accuracy_report
from speed_benchmark import benchmark_inference_speed, benchmark_training_speed, print_speed_report
from memory_benchmark import benchmark_memory, print_memory_report
from scalability_benchmark import benchmark_data_scaling, print_scaling_report

def run_complete_benchmark(network, X_train, y_train, X_test, y_test,
                          output_file: str = None):
    """
    Run complete benchmark suite

    Args:
        network: Trained neural network
        X_train, y_train: Training data
        X_test, y_test: Test data
        output_file: Optional JSON output file
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " COMPLETE BENCHMARK SUITE ".center(58) + "║")
    print("╠" + "═" * 58 + "╣")
    print(f"║ Timestamp: {timestamp}".ljust(59) + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    results = {
        'timestamp': timestamp,
        'accuracy': {},
        'speed': {},
        'memory': {},
        'status': {}
    }

    # Accuracy Benchmark
    print("\n[1/3] Running Accuracy Benchmark...")
    print("-" * 60)
    acc_benchmark = benchmark_accuracy(network, X_test, y_test)
    print_accuracy_report(acc_benchmark)
    results['accuracy'] = {
        'overall': acc_benchmark.overall_accuracy,
        'per_digit': acc_benchmark.per_digit_accuracy
    }

    # Speed Benchmark
    print("\n[2/3] Running Speed Benchmark...")
    print("-" * 60)
    speed_benchmark = benchmark_inference_speed(network)
    training_speed = benchmark_training_speed(network, X_train, y_train)
    speed_benchmark.epoch_time = training_speed['single_epoch']
    print_speed_report(speed_benchmark)
    results['speed'] = {
        'single_inference_ms': speed_benchmark.single_inference.mean_ms,
        'batch_32_ms': speed_benchmark.batch_inference[32].mean_ms,
        'epoch_time_s': training_speed['single_epoch']
    }

    # Memory Benchmark
    print("\n[3/3] Running Memory Benchmark...")
    print("-" * 60)
    mem_benchmark = benchmark_memory(network, X_train, y_train)
    print_memory_report(mem_benchmark)
    results['memory'] = {
        'model_size_bytes': mem_benchmark.model_size_bytes,
        'parameter_count': mem_benchmark.parameter_count,
        'inference_peak_bytes': mem_benchmark.inference_peak_bytes,
        'training_peak_bytes': mem_benchmark.training_peak_bytes
    }

    # Final Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    checks = {
        'Accuracy ≥ 97%': acc_benchmark.overall_accuracy >= 0.97,
        'Inference < 50ms': speed_benchmark.single_inference.mean_ms < 50,
        'Memory < 500MB': mem_benchmark.training_peak_bytes < 500 * 1024 * 1024
    }

    all_pass = True
    for check, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
        if not passed:
            all_pass = False

    results['status']['all_pass'] = all_pass
    results['status']['checks'] = checks

    print()
    if all_pass:
        print("🎉 ALL BENCHMARKS PASSED!")
    else:
        print("⚠️ SOME BENCHMARKS FAILED - Review and optimize")

    # Save to file
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark suite")
    parser.add_argument('--model', type=str, help='Path to saved model')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output JSON file')
    args = parser.parse_args()

    # Load model and data
    from core.network import Network
    from data.mnist_loader import MNISTLoader

    network = Network.load(args.model)
    loader = MNISTLoader()
    X_train, y_train, X_test, y_test = loader.load()

    # Run benchmarks
    run_complete_benchmark(network, X_train, y_train, X_test, y_test, args.output)
```

---

## 7. Benchmark Results Template

### 7.1 Results Summary Format

```
╔════════════════════════════════════════════════════════════╗
║               BENCHMARK RESULTS - v1.0                      ║
╠════════════════════════════════════════════════════════════╣
║ Date: 2026-02-01 14:30:00                                  ║
║ Model: MLP 784-128-64-10                                   ║
║ Parameters: 109,386                                         ║
╠════════════════════════════════════════════════════════════╣
║                        ACCURACY                             ║
║ ─────────────────────────────────────────────────────────  ║
║ Overall:          97.32%    ✅ PASS (target: ≥97%)         ║
║ Per-digit min:    95.10%    ✅ PASS (target: ≥90%)         ║
║ F1 Score (avg):   97.15%    ✅ PASS (target: ≥95%)         ║
╠════════════════════════════════════════════════════════════╣
║                         SPEED                               ║
║ ─────────────────────────────────────────────────────────  ║
║ Single inference: 12.45ms   ✅ PASS (target: <50ms)        ║
║ Batch-32:         45.20ms   ✅ PASS (1.41ms/image)         ║
║ Training epoch:   13.2s     ✅ PASS (target: <15s)         ║
║ Full training:    4.4min    ✅ PASS (target: <5min)        ║
╠════════════════════════════════════════════════════════════╣
║                        MEMORY                               ║
║ ─────────────────────────────────────────────────────────  ║
║ Model size:       437 KB    ✅ PASS (target: <5MB)         ║
║ Inference peak:   52 MB     ✅ PASS (target: <100MB)       ║
║ Training peak:    312 MB    ✅ PASS (target: <500MB)       ║
╠════════════════════════════════════════════════════════════╣
║                    FINAL STATUS                             ║
║ ─────────────────────────────────────────────────────────  ║
║              🎉 ALL BENCHMARKS PASSED                       ║
╚════════════════════════════════════════════════════════════╝
```

---

## 8. Optimization Recommendations

### 8.1 Jika Accuracy Rendah

| Issue                | Possible Cause         | Solution                            |
| -------------------- | ---------------------- | ----------------------------------- |
| < 90% overall        | Underfitting           | Increase network size, train longer |
| < 90% overall        | Learning rate too high | Reduce LR, add scheduler            |
| Specific digit low   | Imbalanced data        | Data augmentation, weighted loss    |
| High train, low test | Overfitting            | Add dropout, regularization         |

### 8.2 Jika Inference Lambat

| Issue              | Possible Cause    | Solution               |
| ------------------ | ----------------- | ---------------------- |
| > 50ms single      | Network too large | Reduce layer sizes     |
| > 50ms single      | Inefficient code  | Vectorize operations   |
| Poor batch scaling | Memory issues     | Optimize memory access |

### 8.3 Jika Memory Tinggi

| Issue            | Possible Cause      | Solution            |
| ---------------- | ------------------- | ------------------- |
| > 500MB training | Large batch         | Reduce batch size   |
| > 500MB training | Too many params     | Compress network    |
| Memory leaks     | Retained references | Clear cache, use gc |

---

**Document Status**: ✅ Complete  
**Related Documents**:

- [TESTING_STRATEGY.md](TESTING_STRATEGY.md)
- [TRAINING_STRATEGY.md](TRAINING_STRATEGY.md)
