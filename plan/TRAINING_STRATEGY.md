# 🎯 Training Strategy - Digit Recognition

**Version**: 1.0  
**Date**: 1 Feb 2026  
**Status**: Planning

---

## 1. Overview

Dokumen ini menjelaskan strategi lengkap untuk melatih neural network digit recognition, termasuk penanganan dataset, teknik optimisasi, dan best practices untuk mencapai akurasi tinggi.

### 1.1 Training Goals

| Goal           | Target                   | Priority |
| -------------- | ------------------------ | -------- |
| Accuracy       | ≥ 97% pada test set      | P0       |
| Training Time  | < 5 menit (full dataset) | P1       |
| Convergence    | Stable loss decrease     | P0       |
| Generalization | Low overfitting          | P0       |

### 1.2 Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                    │
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Load    │───►│  Preproc │───►│  Batch   │───►│  Train   │              │
│  │  MNIST   │    │   Data   │    │ Generator│    │   Loop   │              │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘              │
│                                                        │                    │
│                  ┌────────────────────────────────────┘                    │
│                  │                                                          │
│                  ▼                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Save    │◄───│ Evaluate │◄───│  Update  │◄───│  Forward │              │
│  │  Model   │    │ Metrics  │    │  Weights │    │ /Backward│              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Dataset: MNIST

### 2.1 Dataset Overview

MNIST (Modified National Institute of Standards and Technology) adalah dataset standar untuk digit recognition.

| Property             | Value           |
| -------------------- | --------------- |
| **Training Samples** | 60,000          |
| **Test Samples**     | 10,000          |
| **Image Size**       | 28 × 28 pixels  |
| **Channels**         | 1 (Grayscale)   |
| **Classes**          | 10 (digits 0-9) |
| **Pixel Range**      | 0-255           |

### 2.2 Data Distribution

```
Training Set Distribution:
┌────────────────────────────────────────────────────────────┐
│ Digit │ Count  │ Percentage │ Visual                       │
├────────────────────────────────────────────────────────────┤
│   0   │  5,923 │   9.87%    │ ████████████████████         │
│   1   │  6,742 │  11.24%    │ ██████████████████████       │
│   2   │  5,958 │   9.93%    │ ████████████████████         │
│   3   │  6,131 │  10.22%    │ ████████████████████         │
│   4   │  5,842 │   9.74%    │ ███████████████████          │
│   5   │  5,421 │   9.04%    │ ██████████████████           │
│   6   │  5,918 │   9.86%    │ ████████████████████         │
│   7   │  6,265 │  10.44%    │ █████████████████████        │
│   8   │  5,851 │   9.75%    │ ███████████████████          │
│   9   │  5,949 │   9.92%    │ ████████████████████         │
└────────────────────────────────────────────────────────────┘
```

### 2.3 MNIST Loader Implementation

```python
import numpy as np
import gzip
import os
from urllib import request

class MNISTLoader:
    """
    MNIST dataset loader with auto-download capability

    Usage:
        loader = MNISTLoader(data_dir='data/mnist')
        X_train, y_train, X_test, y_test = loader.load()
    """

    BASE_URL = "http://yann.lecun.com/exdb/mnist/"

    FILES = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }

    def __init__(self, data_dir: str = 'data/mnist'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def download(self):
        """Download MNIST files if not present"""
        for name, filename in self.FILES.items():
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                url = self.BASE_URL + filename
                request.urlretrieve(url, filepath)
                print(f"Downloaded {filename}")

    def _load_images(self, filename: str) -> np.ndarray:
        """Load images from gzipped file"""
        filepath = os.path.join(self.data_dir, filename)
        with gzip.open(filepath, 'rb') as f:
            # Skip header (magic number, num images, rows, cols)
            f.read(16)
            # Read image data
            data = np.frombuffer(f.read(), dtype=np.uint8)
            # Reshape to (num_images, 784) and normalize
            return data.reshape(-1, 784).astype(np.float32) / 255.0

    def _load_labels(self, filename: str) -> np.ndarray:
        """Load labels from gzipped file"""
        filepath = os.path.join(self.data_dir, filename)
        with gzip.open(filepath, 'rb') as f:
            # Skip header (magic number, num labels)
            f.read(8)
            # Read label data
            return np.frombuffer(f.read(), dtype=np.uint8)

    def load(self, normalize: bool = True, one_hot: bool = True):
        """
        Load complete MNIST dataset

        Args:
            normalize: Normalize pixel values to [0, 1]
            one_hot: Convert labels to one-hot encoding

        Returns:
            X_train, y_train, X_test, y_test
        """
        self.download()

        X_train = self._load_images(self.FILES['train_images'])
        y_train = self._load_labels(self.FILES['train_labels'])
        X_test = self._load_images(self.FILES['test_images'])
        y_test = self._load_labels(self.FILES['test_labels'])

        if one_hot:
            y_train = self._to_one_hot(y_train)
            y_test = self._to_one_hot(y_test)

        return X_train, y_train, X_test, y_test

    def _to_one_hot(self, labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
        """Convert integer labels to one-hot encoding"""
        one_hot = np.zeros((labels.shape[0], num_classes))
        one_hot[np.arange(labels.shape[0]), labels] = 1
        return one_hot
```

### 2.4 Data Splitting

```python
def train_val_split(X: np.ndarray, y: np.ndarray,
                    val_ratio: float = 0.1,
                    shuffle: bool = True,
                    seed: int = 42) -> tuple:
    """
    Split training data into train and validation sets

    Args:
        X: Input features
        y: Labels
        val_ratio: Fraction for validation
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducibility

    Returns:
        X_train, y_train, X_val, y_val
    """
    n_samples = X.shape[0]
    n_val = int(n_samples * val_ratio)

    if shuffle:
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]


# Usage
X_train, y_train, X_val, y_val = train_val_split(X_train_full, y_train_full)

# Final split:
# - Training: 54,000 samples (90%)
# - Validation: 6,000 samples (10%)
# - Test: 10,000 samples (separate)
```

---

## 3. Batch Processing

### 3.1 Why Mini-Batches?

| Approach           | Pros                    | Cons               |
| ------------------ | ----------------------- | ------------------ |
| **Full Batch**     | Stable gradients        | Slow, memory-heavy |
| **Stochastic (1)** | Fast updates            | Noisy, unstable    |
| **Mini-Batch**     | Balance speed/stability | Needs tuning size  |

**Recommended**: Mini-batch size of 32-128

### 3.2 Batch Generator

```python
class BatchGenerator:
    """
    Generate mini-batches for training

    Features:
    - Shuffling per epoch
    - Handles incomplete final batch
    - Memory efficient (generator-based)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 batch_size: int = 32, shuffle: bool = True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]
        self.n_batches = int(np.ceil(self.n_samples / batch_size))

    def __iter__(self):
        """Generate batches for one epoch"""
        indices = np.arange(self.n_samples)

        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.X[batch_indices], self.y[batch_indices]

    def __len__(self):
        return self.n_batches


# Usage
batch_gen = BatchGenerator(X_train, y_train, batch_size=32)

for epoch in range(num_epochs):
    for batch_X, batch_y in batch_gen:
        # Training step
        pass
```

### 3.3 Batch Size Considerations

```
Batch Size vs Performance:
┌────────────────────────────────────────────────────────────┐
│ Batch Size │ Memory │ Speed │ Gradient Quality │ Accuracy │
├────────────────────────────────────────────────────────────┤
│     16     │  Low   │ Slow  │     Noisy        │  ~97.0%  │
│     32     │ Medium │ Good  │     Good         │  ~97.5%  │ ◄── Recommended
│     64     │ Medium │ Fast  │     Smooth       │  ~97.3%  │
│    128     │  High  │ Fast  │     Smooth       │  ~97.0%  │
│    256     │  High  │ V.Fast│     Too Smooth   │  ~96.5%  │
└────────────────────────────────────────────────────────────┘
```

---

## 4. Training Loop

### 4.1 Core Training Algorithm

```python
class Trainer:
    """
    Neural network trainer with comprehensive features

    Features:
    - Mini-batch training
    - Validation evaluation
    - Early stopping
    - Model checkpointing
    - Training history
    - Progress callbacks
    """

    def __init__(self, network, optimizer, loss_fn):
        self.network = network
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs: int = 20, batch_size: int = 32,
              callbacks: list = None):
        """
        Train the network

        Args:
            X_train: Training features
            y_train: Training labels (one-hot)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            callbacks: List of callback functions
        """
        callbacks = callbacks or []
        batch_gen = BatchGenerator(X_train, y_train, batch_size)

        for epoch in range(epochs):
            # Training phase
            epoch_loss = 0
            epoch_correct = 0
            n_samples = 0

            for batch_X, batch_y in batch_gen:
                # Forward pass
                predictions = self.network.forward(batch_X)

                # Compute loss
                loss = self.loss_fn(batch_y, predictions)
                epoch_loss += loss * batch_X.shape[0]

                # Compute accuracy
                pred_labels = np.argmax(predictions, axis=1)
                true_labels = np.argmax(batch_y, axis=1)
                epoch_correct += np.sum(pred_labels == true_labels)
                n_samples += batch_X.shape[0]

                # Backward pass
                self.network.backward(batch_y, predictions)

                # Update weights
                self._update_weights()

            # Compute epoch metrics
            train_loss = epoch_loss / n_samples
            train_acc = epoch_correct / n_samples

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validation phase
            if X_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

            # Callbacks
            for callback in callbacks:
                callback.on_epoch_end(epoch, self.history)

            # Print progress
            self._print_progress(epoch, epochs)

        return self.history

    def _update_weights(self):
        """Update network weights using optimizer"""
        for i, layer in enumerate(self.network.layers):
            if hasattr(layer, 'parameters') and layer.parameters:
                self.optimizer.update(
                    layer.parameters,
                    layer.gradients,
                    layer_id=i
                )

    def evaluate(self, X, y):
        """Evaluate model on given data"""
        predictions = self.network.forward(X)
        loss = self.loss_fn(y, predictions)

        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(pred_labels == true_labels)

        return loss, accuracy

    def _print_progress(self, epoch, total_epochs):
        """Print training progress"""
        train_loss = self.history['train_loss'][-1]
        train_acc = self.history['train_acc'][-1]

        msg = f"Epoch {epoch+1}/{total_epochs} - "
        msg += f"loss: {train_loss:.4f} - acc: {train_acc:.4f}"

        if self.history['val_loss']:
            val_loss = self.history['val_loss'][-1]
            val_acc = self.history['val_acc'][-1]
            msg += f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"

        print(msg)
```

### 4.2 Training Flow Visualization

```
Epoch Loop:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  for epoch in range(num_epochs):                                            │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     BATCH LOOP                                       │    │
│  │                                                                      │    │
│  │  for batch_X, batch_y in batches:                                   │    │
│  │      │                                                               │    │
│  │      ├──► Forward Pass ──► Loss ──► Backward Pass ──► Update        │    │
│  │      │                                                               │    │
│  │      └──► Accumulate metrics                                        │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Validation Evaluation (if val data provided)                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Callbacks: Early Stopping, Checkpointing, Logging                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Learning Rate Strategies

### 5.1 Learning Rate Overview

Learning rate adalah hyperparameter paling penting dalam training.

```
Learning Rate Effects:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Too High (0.1)          Just Right (0.001)      Too Low (0.00001)          │
│                                                                              │
│  Loss                    Loss                    Loss                        │
│    │    ╱╲  ╱╲             │                       │                        │
│    │   ╱  ╲╱  ╲            │╲                      │                        │
│    │  ╱        ╲           │ ╲                     │\                       │
│    │ ╱          ╲          │  ╲                    │ \                      │
│    │╱            ╲         │   ╲____              │  \___________          │
│    └────────────────      └────────────────      └────────────────         │
│      Diverges              Converges              Stuck                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Learning Rate Schedules

```python
class LearningRateScheduler:
    """Collection of learning rate scheduling strategies"""

    @staticmethod
    def constant(initial_lr: float):
        """Constant learning rate"""
        def schedule(epoch):
            return initial_lr
        return schedule

    @staticmethod
    def step_decay(initial_lr: float, drop_rate: float = 0.5,
                   epochs_drop: int = 10):
        """
        Step decay: reduce LR by factor every N epochs

        lr = initial_lr * drop_rate^(epoch // epochs_drop)
        """
        def schedule(epoch):
            return initial_lr * (drop_rate ** (epoch // epochs_drop))
        return schedule

    @staticmethod
    def exponential_decay(initial_lr: float, decay_rate: float = 0.95):
        """
        Exponential decay: lr = initial_lr * decay_rate^epoch
        """
        def schedule(epoch):
            return initial_lr * (decay_rate ** epoch)
        return schedule

    @staticmethod
    def cosine_annealing(initial_lr: float, total_epochs: int,
                         min_lr: float = 0.0):
        """
        Cosine annealing: smooth decrease following cosine curve
        """
        def schedule(epoch):
            return min_lr + (initial_lr - min_lr) * \
                   (1 + np.cos(np.pi * epoch / total_epochs)) / 2
        return schedule

    @staticmethod
    def warmup_then_decay(initial_lr: float, warmup_epochs: int = 5,
                          total_epochs: int = 20):
        """
        Linear warmup followed by exponential decay
        """
        def schedule(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return initial_lr * (epoch + 1) / warmup_epochs
            else:
                # Exponential decay
                decay_epochs = epoch - warmup_epochs
                return initial_lr * (0.95 ** decay_epochs)
        return schedule


# Usage
scheduler = LearningRateScheduler.cosine_annealing(
    initial_lr=0.001,
    total_epochs=20
)

for epoch in range(20):
    current_lr = scheduler(epoch)
    optimizer.lr = current_lr
```

### 5.3 Learning Rate Visualization

```
Learning Rate Schedules Comparison:
┌─────────────────────────────────────────────────────────────────────────────┐
│  LR                                                                          │
│  0.001 ┬─────────────────────────────────────────────────────────────────── │
│        │ ─ Constant                                                          │
│        │ ────                                                                │
│        │     ────  Step Decay                                               │
│        │         ────                                                        │
│        │    ╲         ────                                                  │
│        │     ╲            ────                                              │
│        │      ╲    Exponential                                              │
│        │       ╲                                                             │
│        │        ╲    ╭────╮                                                 │
│        │         ╲  ╱      ╲  Cosine                                        │
│        │          ╲╱        ╲                                               │
│  0.000 ┴───────────────────────────────────────────────────────────────────  │
│        0          5         10         15         20  Epoch                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Early Stopping

### 6.1 Concept

Early stopping mencegah overfitting dengan menghentikan training ketika validation performance berhenti improve.

```
With vs Without Early Stopping:
┌─────────────────────────────────────────────────────────────────────────────┐
│  Accuracy                                                                    │
│     │                                                                        │
│ 100%│                          ╭──────────── Train (overfitting)            │
│     │                     ╭────╯                                            │
│     │                ╭────╯                                                 │
│     │           ╭────╯    ╭───────────────── Validation                     │
│     │      ╭────╯    ╭────╯     ↓ Best point (stop here!)                   │
│     │ ╭────╯    ╭────╯                                                      │
│     │╱     ╭────╯                                                           │
│     └────────────────────────────────────────────────────────────────────── │
│           │         │                                                        │
│           │    Early Stop                                                   │
│           │         │                                                        │
│     0     5        10        15        20        25        30   Epoch       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Implementation

```python
class EarlyStopping:
    """
    Stop training when validation metric stops improving

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy
        restore_best: Whether to restore best weights
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001,
                 mode: str = 'min', restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.best_weights = None
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(self, epoch: int, history: dict, network=None):
        """Check if training should stop"""
        # Get current validation metric
        current = history['val_loss'][-1] if self.mode == 'min' \
                  else history['val_acc'][-1]

        # Check for improvement
        if self._is_improvement(current):
            self.best_value = current
            self.best_epoch = epoch
            self.counter = 0

            # Save best weights
            if self.restore_best and network:
                self.best_weights = self._copy_weights(network)
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\nEarly stopping at epoch {epoch + 1}")
                print(f"Best epoch was {self.best_epoch + 1} "
                      f"with value {self.best_value:.4f}")

                # Restore best weights
                if self.restore_best and self.best_weights and network:
                    self._restore_weights(network, self.best_weights)

    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement"""
        if self.mode == 'min':
            return current < self.best_value - self.min_delta
        else:
            return current > self.best_value + self.min_delta

    def _copy_weights(self, network) -> list:
        """Deep copy network weights"""
        weights = []
        for layer in network.layers:
            if hasattr(layer, 'parameters') and layer.parameters:
                layer_weights = {}
                for key, value in layer.parameters.items():
                    layer_weights[key] = value.copy()
                weights.append(layer_weights)
        return weights

    def _restore_weights(self, network, weights: list):
        """Restore weights to network"""
        weight_idx = 0
        for layer in network.layers:
            if hasattr(layer, 'parameters') and layer.parameters:
                for key in layer.parameters:
                    layer.parameters[key] = weights[weight_idx][key].copy()
                weight_idx += 1


# Usage
early_stopping = EarlyStopping(patience=5, mode='min')

trainer.train(
    X_train, y_train,
    X_val, y_val,
    epochs=100,  # Set high, early stopping will handle it
    callbacks=[early_stopping]
)
```

---

## 7. Model Checkpointing

### 7.1 Checkpoint Strategy

```python
class ModelCheckpoint:
    """
    Save model weights during training

    Features:
    - Save best model only
    - Save every N epochs
    - Save on improvement
    """

    def __init__(self, filepath: str, monitor: str = 'val_loss',
                 mode: str = 'min', save_best_only: bool = True,
                 verbose: bool = True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose

        self.best_value = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(self, epoch: int, history: dict, network):
        """Check and save model"""
        current = history[self.monitor][-1]

        should_save = False

        if self.save_best_only:
            if self._is_improvement(current):
                self.best_value = current
                should_save = True
        else:
            should_save = True

        if should_save:
            # Generate filepath with epoch info
            filepath = self.filepath.format(
                epoch=epoch + 1,
                val_loss=history.get('val_loss', [0])[-1],
                val_acc=history.get('val_acc', [0])[-1]
            )

            self._save_model(network, filepath)

            if self.verbose:
                print(f"\nSaved model to {filepath}")

    def _is_improvement(self, current: float) -> bool:
        if self.mode == 'min':
            return current < self.best_value
        return current > self.best_value

    def _save_model(self, network, filepath: str):
        """Save model weights to file"""
        weights = {}
        for i, layer in enumerate(network.layers):
            if hasattr(layer, 'parameters') and layer.parameters:
                for key, value in layer.parameters.items():
                    weights[f'layer_{i}_{key}'] = value

        np.savez(filepath, **weights)


# Usage
checkpoint = ModelCheckpoint(
    filepath='models/best_model.npz',
    monitor='val_acc',
    mode='max',
    save_best_only=True
)
```

### 7.2 Model Persistence

```python
class ModelIO:
    """Save and load complete model state"""

    @staticmethod
    def save(network, filepath: str, include_optimizer: bool = False,
             optimizer=None):
        """
        Save complete model to file

        File format (NPZ):
        - architecture: layer configurations
        - weights: all trainable parameters
        - optimizer_state: optimizer momentum/velocity (optional)
        """
        data = {}

        # Save architecture
        architecture = []
        for layer in network.layers:
            layer_info = {
                'type': layer.__class__.__name__,
            }
            if hasattr(layer, 'input_size'):
                layer_info['input_size'] = layer.input_size
            if hasattr(layer, 'output_size'):
                layer_info['output_size'] = layer.output_size
            architecture.append(layer_info)

        data['architecture'] = np.array(str(architecture))

        # Save weights
        for i, layer in enumerate(network.layers):
            if hasattr(layer, 'parameters') and layer.parameters:
                for key, value in layer.parameters.items():
                    data[f'layer_{i}_{key}'] = value

        # Save optimizer state (optional)
        if include_optimizer and optimizer:
            if hasattr(optimizer, 'm'):  # Adam
                for key, value in optimizer.m.items():
                    data[f'optimizer_m_{key}'] = value
                for key, value in optimizer.v.items():
                    data[f'optimizer_v_{key}'] = value
                data['optimizer_t'] = np.array(optimizer.t)

        np.savez_compressed(filepath, **data)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath: str, network=None):
        """
        Load model weights from file

        If network is provided, load weights into it.
        Otherwise, return weights dictionary.
        """
        data = np.load(filepath, allow_pickle=True)

        if network:
            # Load weights into network
            for i, layer in enumerate(network.layers):
                if hasattr(layer, 'parameters') and layer.parameters:
                    for key in layer.parameters:
                        param_key = f'layer_{i}_{key}'
                        if param_key in data:
                            layer.parameters[key] = data[param_key]
            print(f"Model loaded from {filepath}")
            return network
        else:
            # Return raw weights
            return dict(data)


# Usage
# Save
ModelIO.save(network, 'models/digit_recognition.npz')

# Load
network = create_network()
ModelIO.load('models/digit_recognition.npz', network)
```

---

## 8. Regularization Techniques

### 8.1 L2 Regularization (Weight Decay)

```python
class L2Regularization:
    """
    L2 regularization adds penalty for large weights

    Loss_total = Loss_data + λ * Σ||W||²
    """

    def __init__(self, lambda_: float = 0.001):
        self.lambda_ = lambda_

    def loss(self, network) -> float:
        """Compute regularization loss"""
        reg_loss = 0
        for layer in network.layers:
            if hasattr(layer, 'weights'):
                reg_loss += np.sum(layer.weights ** 2)
        return self.lambda_ * reg_loss

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Compute regularization gradient"""
        return 2 * self.lambda_ * weights


# Integrate with training
class RegularizedTrainer(Trainer):
    def __init__(self, network, optimizer, loss_fn, l2_lambda=0.001):
        super().__init__(network, optimizer, loss_fn)
        self.l2_reg = L2Regularization(l2_lambda)

    def _compute_loss(self, y_true, y_pred):
        data_loss = self.loss_fn(y_true, y_pred)
        reg_loss = self.l2_reg.loss(self.network)
        return data_loss + reg_loss
```

### 8.2 Dropout

```python
class Dropout:
    """
    Dropout regularization

    During training: randomly set activations to 0
    During inference: use all activations (scaled)
    """

    def __init__(self, rate: float = 0.5):
        self.rate = rate
        self.mask = None
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training:
            return x

        # Create and apply dropout mask
        self.mask = np.random.binomial(1, 1 - self.rate, size=x.shape)
        return x * self.mask / (1 - self.rate)  # Inverted dropout

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if not self.training:
            return grad
        return grad * self.mask / (1 - self.rate)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
```

### 8.3 Regularization Comparison

| Technique             | Effect                   | When to Use           |
| --------------------- | ------------------------ | --------------------- |
| **L2**                | Prevents large weights   | Always recommended    |
| **L1**                | Creates sparse weights   | Feature selection     |
| **Dropout**           | Prevents co-adaptation   | Large networks        |
| **Early Stopping**    | Prevents overtraining    | Always recommended    |
| **Data Augmentation** | Increases effective data | Limited training data |

---

## 9. Data Augmentation

### 9.1 Augmentation Techniques

```python
class DataAugmenter:
    """
    Data augmentation for MNIST digits

    Techniques:
    - Rotation (small angles)
    - Translation (shift)
    - Scaling
    - Noise injection
    - Elastic deformation
    """

    def __init__(self, rotation_range: float = 15,
                 shift_range: float = 0.1,
                 scale_range: tuple = (0.9, 1.1),
                 noise_std: float = 0.1):
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.noise_std = noise_std

    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentation to image

        Args:
            image: 1D array of shape (784,) or 2D (28, 28)

        Returns:
            Augmented image same shape as input
        """
        # Reshape to 2D if needed
        original_shape = image.shape
        if image.ndim == 1:
            image = image.reshape(28, 28)

        # Apply random augmentations
        if np.random.random() < 0.5:
            image = self._rotate(image)
        if np.random.random() < 0.5:
            image = self._shift(image)
        if np.random.random() < 0.5:
            image = self._scale(image)
        if np.random.random() < 0.3:
            image = self._add_noise(image)

        # Restore original shape
        if len(original_shape) == 1:
            image = image.flatten()

        return image

    def _rotate(self, image: np.ndarray) -> np.ndarray:
        """Rotate image by random angle"""
        from scipy.ndimage import rotate
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        return rotate(image, angle, reshape=False, mode='constant')

    def _shift(self, image: np.ndarray) -> np.ndarray:
        """Shift image by random amount"""
        from scipy.ndimage import shift
        max_shift = int(28 * self.shift_range)
        dx = np.random.randint(-max_shift, max_shift + 1)
        dy = np.random.randint(-max_shift, max_shift + 1)
        return shift(image, (dy, dx), mode='constant')

    def _scale(self, image: np.ndarray) -> np.ndarray:
        """Scale image by random factor"""
        from scipy.ndimage import zoom
        scale = np.random.uniform(*self.scale_range)
        scaled = zoom(image, scale, mode='constant')

        # Crop or pad to original size
        result = np.zeros((28, 28))
        h, w = scaled.shape

        # Center the scaled image
        start_y = (28 - h) // 2
        start_x = (28 - w) // 2

        if scale > 1:
            # Crop
            crop_y = (h - 28) // 2
            crop_x = (w - 28) // 2
            result = scaled[crop_y:crop_y+28, crop_x:crop_x+28]
        else:
            # Pad
            result[start_y:start_y+h, start_x:start_x+w] = scaled

        return result

    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise"""
        noise = np.random.normal(0, self.noise_std, image.shape)
        return np.clip(image + noise, 0, 1)


# Usage with training
augmenter = DataAugmenter()

for epoch in range(epochs):
    for batch_X, batch_y in batches:
        # Augment batch
        augmented_X = np.array([augmenter.augment(x) for x in batch_X])

        # Train on augmented data
        # ...
```

### 9.2 Augmentation Visualization

```
Original vs Augmented:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   Original       Rotated       Shifted       Scaled        Noisy            │
│                                                                              │
│   ┌─────┐       ┌─────┐       ┌─────┐       ┌─────┐       ┌─────┐          │
│   │  ▄  │       │   ▄ │       │     │       │  ▄  │       │ ▄▒▄ │          │
│   │ █▀█ │       │ █▀█ │       │  ▄  │       │ ███ │       │█▒██▒│          │
│   │   █ │       │  █  │       │ █▀█ │       │   █ │       │ ▒ █▒│          │
│   │   █ │       │  █  │       │   █ │       │   █ │       │  ▒█ │          │
│   └─────┘       └─────┘       └─────┘       └─────┘       └─────┘          │
│     "7"           "7"           "7"           "7"           "7"            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Training Monitoring

### 10.1 Metrics Tracking

```python
class TrainingMonitor:
    """
    Monitor and visualize training progress

    Tracks:
    - Loss (train/val)
    - Accuracy (train/val)
    - Learning rate
    - Gradient norms
    - Weight statistics
    """

    def __init__(self):
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': [],
            'gradient_norm': []
        }

    def log(self, epoch: int, train_loss: float, val_loss: float,
            train_acc: float, val_acc: float, lr: float = None,
            grad_norm: float = None):
        """Log metrics for one epoch"""
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_acc'].append(val_acc)
        if lr:
            self.metrics['learning_rate'].append(lr)
        if grad_norm:
            self.metrics['gradient_norm'].append(grad_norm)

    def plot_loss(self):
        """Plot loss curves"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(self.metrics['epoch'], self.metrics['train_loss'],
                 label='Train Loss')
        plt.plot(self.metrics['epoch'], self.metrics['val_loss'],
                 label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_accuracy(self):
        """Plot accuracy curves"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(self.metrics['epoch'], self.metrics['train_acc'],
                 label='Train Accuracy')
        plt.plot(self.metrics['epoch'], self.metrics['val_acc'],
                 label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

    def summary(self):
        """Print training summary"""
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"Total Epochs: {len(self.metrics['epoch'])}")
        print(f"Final Train Loss: {self.metrics['train_loss'][-1]:.4f}")
        print(f"Final Val Loss: {self.metrics['val_loss'][-1]:.4f}")
        print(f"Final Train Acc: {self.metrics['train_acc'][-1]:.4f}")
        print(f"Final Val Acc: {self.metrics['val_acc'][-1]:.4f}")
        print(f"Best Val Acc: {max(self.metrics['val_acc']):.4f}")
        print("=" * 50)
```

### 10.2 Progress Bar

```python
class ProgressBar:
    """Training progress bar with metrics display"""

    def __init__(self, total: int, width: int = 50):
        self.total = total
        self.width = width
        self.current = 0

    def update(self, current: int = None, **metrics):
        """Update progress bar"""
        if current is not None:
            self.current = current
        else:
            self.current += 1

        # Calculate progress
        progress = self.current / self.total
        filled = int(self.width * progress)
        bar = '█' * filled + '░' * (self.width - filled)

        # Format metrics
        metrics_str = ' - '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])

        # Print progress
        print(f'\r[{bar}] {self.current}/{self.total} {metrics_str}', end='')

        if self.current >= self.total:
            print()  # New line when complete

    def reset(self):
        """Reset progress bar"""
        self.current = 0


# Usage
progress = ProgressBar(total=len(batches))

for batch_X, batch_y in batches:
    # ... training step ...
    progress.update(loss=loss, acc=accuracy)
```

---

## 11. Hyperparameter Tuning

### 11.1 Recommended Hyperparameters

| Hyperparameter | Default   | Range to Try   | Notes                   |
| -------------- | --------- | -------------- | ----------------------- |
| Learning Rate  | 0.001     | [0.0001, 0.01] | Most important          |
| Batch Size     | 32        | [16, 128]      | Affects speed & quality |
| Hidden Units   | [128, 64] | [64-256]       | More = more capacity    |
| Epochs         | 20        | [10, 50]       | Use early stopping      |
| L2 Lambda      | 0.001     | [0.0001, 0.01] | Regularization strength |
| Dropout Rate   | 0.5       | [0.2, 0.7]     | If using dropout        |

### 11.2 Grid Search

```python
class HyperparameterTuner:
    """Simple grid search for hyperparameter tuning"""

    def __init__(self, param_grid: dict):
        self.param_grid = param_grid
        self.results = []

    def search(self, X_train, y_train, X_val, y_val):
        """Run grid search"""
        from itertools import product

        # Generate all combinations
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(product(*values))

        print(f"Testing {len(combinations)} combinations...")

        for combo in combinations:
            params = dict(zip(keys, combo))
            print(f"\nTesting: {params}")

            # Create and train model with these params
            network = self._create_network(params)
            trainer = Trainer(
                network,
                optimizer=Adam(learning_rate=params.get('learning_rate', 0.001)),
                loss_fn=CrossEntropyLoss()
            )

            history = trainer.train(
                X_train, y_train, X_val, y_val,
                epochs=params.get('epochs', 10),
                batch_size=params.get('batch_size', 32)
            )

            # Record results
            best_val_acc = max(history['val_acc'])
            self.results.append({
                'params': params,
                'val_acc': best_val_acc
            })

        # Sort by accuracy
        self.results.sort(key=lambda x: x['val_acc'], reverse=True)

        print("\n" + "=" * 50)
        print("BEST PARAMETERS:")
        print(self.results[0])
        print("=" * 50)

        return self.results[0]['params']

    def _create_network(self, params):
        """Create network with given parameters"""
        hidden_units = params.get('hidden_units', [128, 64])

        builder = NetworkBuilder().input(784)

        for units in hidden_units:
            builder.dense(units, activation='relu')

        builder.dense(10, activation='softmax')

        return builder.build()


# Usage
tuner = HyperparameterTuner({
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [32, 64],
    'hidden_units': [[64, 32], [128, 64], [256, 128]]
})

best_params = tuner.search(X_train, y_train, X_val, y_val)
```

---

## 12. Complete Training Example

```python
def train_digit_recognition():
    """Complete training pipeline for digit recognition"""

    # 1. Load data
    print("Loading MNIST dataset...")
    loader = MNISTLoader()
    X_train, y_train, X_test, y_test = loader.load()

    # 2. Split train/val
    X_train, y_train, X_val, y_val = train_val_split(
        X_train, y_train, val_ratio=0.1
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    # 3. Create network
    print("\nCreating neural network...")
    network = NetworkBuilder() \
        .input(784) \
        .dense(128, activation='relu') \
        .dense(64, activation='relu') \
        .dense(10, activation='softmax') \
        .build()

    network.summary()

    # 4. Setup training
    optimizer = Adam(learning_rate=0.001)
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(network, optimizer, loss_fn)

    # 5. Setup callbacks
    early_stopping = EarlyStopping(patience=5, mode='max',
                                    monitor='val_acc')
    checkpoint = ModelCheckpoint(
        filepath='models/best_model.npz',
        monitor='val_acc',
        mode='max'
    )

    # 6. Train
    print("\nStarting training...")
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, checkpoint]
    )

    # 7. Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = trainer.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # 8. Save final model
    ModelIO.save(network, 'models/final_model.npz')

    return network, history


if __name__ == '__main__':
    network, history = train_digit_recognition()
```

---

## 13. Expected Results

### 13.1 Training Progress

```
Epoch 1/20 - loss: 0.4521 - acc: 0.8742 - val_loss: 0.2134 - val_acc: 0.9387
Epoch 2/20 - loss: 0.1892 - acc: 0.9456 - val_loss: 0.1521 - val_acc: 0.9567
Epoch 3/20 - loss: 0.1342 - acc: 0.9612 - val_loss: 0.1234 - val_acc: 0.9645
...
Epoch 15/20 - loss: 0.0234 - acc: 0.9934 - val_loss: 0.0812 - val_acc: 0.9756

Test Accuracy: 97.42%
```

### 13.2 Performance Benchmarks

| Metric                    | Target  | Expected |
| ------------------------- | ------- | -------- |
| Test Accuracy             | ≥ 97%   | 97-98%   |
| Training Time (20 epochs) | < 5 min | ~3 min   |
| Inference Time            | < 50ms  | ~5ms     |
| Model Size                | < 5 MB  | ~1 MB    |

---

**Document Status**: ✅ Complete  
**Related Documents**:

- [NEURAL_NETWORK_DESIGN.md](NEURAL_NETWORK_DESIGN.md)
- [MATHEMATICAL_FOUNDATION.md](MATHEMATICAL_FOUNDATION.md)
- [PREPROCESSING_PIPELINE.md](PREPROCESSING_PIPELINE.md)
