# 🧮 Mathematical Foundation - Digit Recognition

**Version**: 1.0  
**Date**: 1 Feb 2026  
**Status**: Planning

---

## 1. Overview

Dokumen ini menjelaskan fondasi matematis yang diperlukan untuk membangun neural network dari nol. Setiap konsep dijelaskan dengan notasi formal, intuisi, dan implementasi Python/NumPy.

---

## 2. Linear Algebra Fundamentals

### 2.1 Vectors dan Matrices

#### Definisi Vector

Sebuah vector adalah array 1-dimensi dari bilangan real:

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n$$

**Implementasi NumPy:**

```python
import numpy as np

# Column vector (n, 1)
x = np.array([[1], [2], [3]])

# Row vector (1, n) - biasa untuk input
x = np.array([[1, 2, 3]])

# 1D array (n,) - paling umum digunakan
x = np.array([1, 2, 3])
```

#### Definisi Matrix

Matrix adalah array 2-dimensi dengan shape $(m, n)$:

$$\mathbf{W} = \begin{bmatrix} w_{11} & w_{12} & \cdots & w_{1n} \\ w_{21} & w_{22} & \cdots & w_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ w_{m1} & w_{m2} & \cdots & w_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

**Implementasi NumPy:**

```python
# Matrix 3x4 (3 rows, 4 columns)
W = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print(W.shape)  # (3, 4)
```

### 2.2 Matrix Operations

#### Matrix Multiplication (Dot Product)

Untuk matrices $\mathbf{A} \in \mathbb{R}^{m \times n}$ dan $\mathbf{B} \in \mathbb{R}^{n \times p}$:

$$(\mathbf{A} \cdot \mathbf{B})_{ij} = \sum_{k=1}^{n} a_{ik} \cdot b_{kj}$$

Hasil: $\mathbf{C} \in \mathbb{R}^{m \times p}$

**Visualisasi:**

```
A (m×n)     B (n×p)     C (m×p)
┌─────┐   ┌─────┐    ┌─────┐
│     │ × │     │ =  │     │
│     │   │     │    │     │
└─────┘   └─────┘    └─────┘
  m×n       n×p        m×p
```

**Implementasi NumPy:**

```python
A = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
B = np.array([[7, 8, 9], [10, 11, 12]])  # (2, 3)

# Matrix multiplication
C = np.dot(A, B)  # atau A @ B
# C shape: (3, 3)
```

#### Transpose

Transpose menukar rows dan columns:

$$(\mathbf{A}^T)_{ij} = \mathbf{A}_{ji}$$

**Implementasi:**

```python
A = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
A_T = A.T  # (3, 2)
```

#### Element-wise Operations (Hadamard Product)

$$(\mathbf{A} \odot \mathbf{B})_{ij} = a_{ij} \cdot b_{ij}$$

**Implementasi:**

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A * B  # Element-wise multiplication
```

### 2.3 Broadcasting

NumPy broadcasting memungkinkan operasi pada arrays dengan shapes berbeda:

```python
# Vector + scalar (broadcasting)
x = np.array([1, 2, 3])
y = x + 5  # [6, 7, 8]

# Matrix + vector (broadcasting along axis)
W = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
b = np.array([10, 20])  # (2,)
result = W + b  # Broadcasting: (3, 2) + (2,) = (3, 2)
```

---

## 3. Calculus for Neural Networks

### 3.1 Derivatives

#### Definisi

Derivative mengukur rate of change:

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

#### Common Derivatives

| Function                                | Derivative               |
| --------------------------------------- | ------------------------ |
| $f(x) = c$                              | $f'(x) = 0$              |
| $f(x) = x^n$                            | $f'(x) = nx^{n-1}$       |
| $f(x) = e^x$                            | $f'(x) = e^x$            |
| $f(x) = \ln(x)$                         | $f'(x) = \frac{1}{x}$    |
| $f(x) = \frac{1}{1 + e^{-x}}$ (sigmoid) | $f'(x) = f(x)(1 - f(x))$ |

### 3.2 Chain Rule

**The Most Important Rule for Backpropagation**

Jika $y = f(g(x))$, maka:

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

**Contoh:**

```
y = (2x + 3)²

Let g = 2x + 3
y = g²

dy/dx = dy/dg × dg/dx
      = 2g × 2
      = 2(2x + 3) × 2
      = 4(2x + 3)
```

**Dalam Neural Network:**

```
Input → Layer1 → Layer2 → Layer3 → Output → Loss

dLoss/dLayer1 = dLoss/dOutput × dOutput/dLayer3 × dLayer3/dLayer2 × dLayer2/dLayer1
```

### 3.3 Partial Derivatives

Untuk function dengan multiple variables:

$$f(x, y) = x^2 + 3xy + y^2$$

$$\frac{\partial f}{\partial x} = 2x + 3y$$

$$\frac{\partial f}{\partial y} = 3x + 2y$$

### 3.4 Gradient

Gradient adalah vector dari semua partial derivatives:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

Gradient menunjuk ke arah **steepest ascent** (kenaikan tercuram).

---

## 4. Activation Functions

### 4.1 Sigmoid Function

**Formula:**
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Derivative:**
$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$

**Properties:**

- Output range: $(0, 1)$
- Smooth gradient
- Problem: Vanishing gradient untuk |x| besar

**Graph:**

```
    1 ─────────────────────────
      │              ╭────────
      │            ╭─╯
  0.5 │ ──────────╭╯──────────
      │         ╭─╯
      │      ───╯
    0 ─────────────────────────
         -6    -3    0    3    6
```

**Implementasi:**

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
```

### 4.2 ReLU (Rectified Linear Unit)

**Formula:**
$$\text{ReLU}(x) = \max(0, x)$$

**Derivative:**
$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

**Properties:**

- Simple computation
- No vanishing gradient (untuk positive values)
- Sparse activation
- Problem: "Dying ReLU" (neurons stuck at 0)

**Graph:**

```
    y │
      │        ╱
      │      ╱
      │    ╱
      │  ╱
    ──┼────────── x
      │
```

**Implementasi:**

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
```

### 4.3 Leaky ReLU

**Formula:**
$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$

dimana $\alpha$ biasanya $0.01$

**Implementasi:**

```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)
```

### 4.4 Softmax Function

**Formula:**
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$$

**Properties:**

- Output range: $(0, 1)$
- Sum of all outputs = 1
- Digunakan untuk multi-class classification
- Converts scores to probabilities

**Numerical Stability:**

```python
def softmax(x):
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

**Contoh:**

```python
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
# Output: [0.659, 0.242, 0.099] (sums to 1.0)
```

### 4.5 Activation Comparison

| Function   | Output Range | Used For                     | Pros               | Cons                  |
| ---------- | ------------ | ---------------------------- | ------------------ | --------------------- |
| Sigmoid    | (0, 1)       | Binary classification, gates | Smooth             | Vanishing gradient    |
| Tanh       | (-1, 1)      | Hidden layers                | Zero-centered      | Vanishing gradient    |
| ReLU       | [0, ∞)       | Hidden layers                | Fast, no vanishing | Dying neurons         |
| Leaky ReLU | (-∞, ∞)      | Hidden layers                | No dying neurons   | Needs tuning α        |
| Softmax    | (0, 1)       | Multi-class output           | Probabilities      | Expensive computation |

---

## 5. Loss Functions

### 5.1 Mean Squared Error (MSE)

**Formula:**
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Derivative (per sample):**
$$\frac{\partial \text{MSE}}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)$$

**Implementasi:**

```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]
```

**Use Case:** Regression problems

### 5.2 Cross-Entropy Loss

**Binary Cross-Entropy:**
$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

**Categorical Cross-Entropy (Multi-class):**
$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

dimana $y$ adalah one-hot encoded label.

**Derivative (dengan softmax output):**
$$\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i$$

(Simplified form ketika combined dengan softmax)

**Implementasi:**

```python
def cross_entropy_loss(y_true, y_pred):
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # If y_true is one-hot encoded
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

def cross_entropy_derivative(y_true, y_pred):
    # When used with softmax
    return y_pred - y_true
```

**Use Case:** Classification problems

### 5.3 Loss Function Comparison

| Loss Function | Formula               | Use Case       | Properties            |
| ------------- | --------------------- | -------------- | --------------------- |
| MSE           | $(y - \hat{y})^2$     | Regression     | Sensitive to outliers |
| MAE           | $\|y - \hat{y}\|$     | Regression     | Robust to outliers    |
| Cross-Entropy | $-y\log(\hat{y})$     | Classification | Good with softmax     |
| Hinge Loss    | $\max(0, 1-y\hat{y})$ | SVM            | Margin-based          |

---

## 6. Forward Propagation

### 6.1 Single Layer Computation

Untuk layer dengan input $\mathbf{x}$, weights $\mathbf{W}$, bias $\mathbf{b}$:

**Linear transformation:**
$$\mathbf{z} = \mathbf{W} \cdot \mathbf{x} + \mathbf{b}$$

**Activation:**
$$\mathbf{a} = f(\mathbf{z})$$

dimana $f$ adalah activation function.

**Visualisasi:**

```
Input (x)      Weights (W)     Bias (b)    Activation
   │               │              │             │
   │    Linear     │              │   Apply f   │
   └───────────────┼──────────────┼─────────────┘
                   │              │
                   v              v
              z = Wx + b    a = f(z)
```

### 6.2 Multi-Layer Forward Pass

Untuk network dengan L layers:

$$\mathbf{a}^{[0]} = \mathbf{x} \text{ (input)}$$

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \cdot \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

$$\mathbf{a}^{[l]} = f^{[l]}(\mathbf{z}^{[l]})$$

$$\hat{\mathbf{y}} = \mathbf{a}^{[L]} \text{ (output)}$$

**Implementasi:**

```python
class Network:
    def forward(self, x):
        self.activations = [x]
        self.z_values = []

        a = x
        for layer in self.layers:
            z = np.dot(a, layer.weights) + layer.bias
            self.z_values.append(z)

            a = layer.activation(z)
            self.activations.append(a)

        return a
```

---

## 7. Backward Propagation

### 7.1 Chain Rule Application

Backpropagation menggunakan chain rule untuk menghitung gradients secara efisien.

**Goal:** Compute $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$ dan $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}}$ untuk setiap layer.

### 7.2 Gradient Computation

#### Output Layer Gradient

$$\delta^{[L]} = \nabla_{\mathbf{a}^{[L]}} \mathcal{L} \odot f'^{[L]}(\mathbf{z}^{[L]})$$

Untuk softmax + cross-entropy:
$$\delta^{[L]} = \mathbf{a}^{[L]} - \mathbf{y}$$

#### Hidden Layer Gradient

$$\delta^{[l]} = (\mathbf{W}^{[l+1]})^T \cdot \delta^{[l+1]} \odot f'^{[l]}(\mathbf{z}^{[l]})$$

#### Weight dan Bias Gradients

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = (\mathbf{a}^{[l-1]})^T \cdot \delta^{[l]}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \sum \delta^{[l]}$$

### 7.3 Backpropagation Algorithm

```
1. Forward pass: Compute semua z dan a

2. Output layer:
   δ[L] = a[L] - y  (for softmax + cross-entropy)

3. For l = L-1 to 1:
   δ[l] = (W[l+1].T @ δ[l+1]) * f'(z[l])

4. Compute gradients:
   dW[l] = a[l-1].T @ δ[l]
   db[l] = sum(δ[l], axis=0)
```

**Implementasi:**

```python
def backward(self, y_true):
    m = y_true.shape[0]  # batch size

    # Output layer gradient
    delta = self.activations[-1] - y_true  # softmax + cross-entropy

    self.gradients = []

    # Backpropagate through layers
    for l in reversed(range(len(self.layers))):
        # Compute weight and bias gradients
        dW = np.dot(self.activations[l].T, delta) / m
        db = np.sum(delta, axis=0, keepdims=True) / m

        self.gradients.insert(0, (dW, db))

        # Compute delta for previous layer
        if l > 0:
            delta = np.dot(delta, self.layers[l].weights.T)
            delta *= self.layers[l-1].activation_derivative(self.z_values[l-1])
```

### 7.4 Backprop Visualization

```
Forward: ─────────────────────────────────────────────►
         Input    Hidden1    Hidden2    Output    Loss
           │         │          │         │         │
           ▼         ▼          ▼         ▼         ▼
          x ──W1──► a1 ──W2──► a2 ──W3──► ŷ ─────► L
           │         │          │         │         │
Backward: ◄─────────────────────────────────────────
         dW1       dW2        dW3       δ[L]
```

---

## 8. Optimization Algorithms

### 8.1 Gradient Descent

**Update Rule:**
$$\mathbf{W} := \mathbf{W} - \eta \cdot \nabla_\mathbf{W} \mathcal{L}$$

dimana $\eta$ adalah learning rate.

### 8.2 Stochastic Gradient Descent (SGD)

- Update weights after each **single sample**
- Noisy but fast

```python
def sgd_update(weights, gradients, learning_rate):
    return weights - learning_rate * gradients
```

### 8.3 Mini-Batch Gradient Descent

- Update weights after **batch of samples**
- Balance antara speed dan stability

```python
def minibatch_update(weights, batch_gradients, learning_rate):
    avg_gradient = np.mean(batch_gradients, axis=0)
    return weights - learning_rate * avg_gradient
```

### 8.4 SGD with Momentum

**Intuition:** Tambahkan "inertia" untuk melewati local minima

$$\mathbf{v} := \beta \mathbf{v} + (1 - \beta) \nabla_\mathbf{W} \mathcal{L}$$
$$\mathbf{W} := \mathbf{W} - \eta \mathbf{v}$$

```python
class SGDMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, weights, gradients):
        if self.velocity is None:
            self.velocity = np.zeros_like(weights)

        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * gradients
        return weights - self.lr * self.velocity
```

### 8.5 Adam Optimizer

**Adam = Adaptive Moment Estimation**

Combines momentum + adaptive learning rates.

$$\mathbf{m} := \beta_1 \mathbf{m} + (1 - \beta_1) \mathbf{g}$$
$$\mathbf{v} := \beta_2 \mathbf{v} + (1 - \beta_2) \mathbf{g}^2$$
$$\hat{\mathbf{m}} := \frac{\mathbf{m}}{1 - \beta_1^t}$$
$$\hat{\mathbf{v}} := \frac{\mathbf{v}}{1 - \beta_2^t}$$
$$\mathbf{W} := \mathbf{W} - \eta \frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}} + \epsilon}$$

**Implementasi:**

```python
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, weights, gradients):
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)

        self.t += 1

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients

        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradients**2

        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Update weights
        return weights - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

### 8.6 Optimizer Comparison

| Optimizer | Pros                         | Cons                        | Best For           |
| --------- | ---------------------------- | --------------------------- | ------------------ |
| SGD       | Simple, well-understood      | Slow convergence            | Small datasets     |
| Momentum  | Faster, escapes local minima | Extra hyperparameter        | General use        |
| Adam      | Adaptive, fast               | May not converge to optimal | Most cases         |
| RMSprop   | Good for RNNs                | Less popular                | Recurrent networks |

---

## 9. Regularization Techniques

### 9.1 L2 Regularization (Weight Decay)

**Loss dengan L2:**
$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \sum_l \|\mathbf{W}^{[l]}\|^2$$

**Gradient update:**
$$\nabla_\mathbf{W} \mathcal{L}_{total} = \nabla_\mathbf{W} \mathcal{L}_{data} + 2\lambda \mathbf{W}$$

**Implementasi:**

```python
def l2_regularization_loss(weights, lambda_):
    return lambda_ * np.sum(weights ** 2)

def l2_regularization_gradient(weights, lambda_):
    return 2 * lambda_ * weights
```

### 9.2 L1 Regularization

**Loss dengan L1:**
$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \sum_l \|\mathbf{W}^{[l]}\|_1$$

**Effect:** Produces sparse weights (many zeros)

### 9.3 Dropout

**During Training:**

- Randomly set some activations to 0 dengan probability $p$
- Scale remaining activations by $\frac{1}{1-p}$

**During Inference:**

- Use all neurons (no dropout)

**Implementasi:**

```python
def dropout(x, p=0.5, training=True):
    if not training:
        return x

    mask = np.random.binomial(1, 1-p, size=x.shape) / (1-p)
    return x * mask
```

---

## 10. Weight Initialization

### 10.1 Xavier/Glorot Initialization

Untuk layer dengan $n_{in}$ inputs dan $n_{out}$ outputs:

$$\mathbf{W} \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

**Implementasi:**

```python
def xavier_init(n_in, n_out):
    std = np.sqrt(2.0 / (n_in + n_out))
    return np.random.randn(n_in, n_out) * std
```

### 10.2 He Initialization

Untuk ReLU activations:

$$\mathbf{W} \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$$

**Implementasi:**

```python
def he_init(n_in, n_out):
    std = np.sqrt(2.0 / n_in)
    return np.random.randn(n_in, n_out) * std
```

---

## 11. Mathematical Notation Summary

| Symbol             | Meaning                             |
| ------------------ | ----------------------------------- |
| $\mathbf{x}$       | Input vector                        |
| $\mathbf{y}$       | True labels                         |
| $\hat{\mathbf{y}}$ | Predicted output                    |
| $\mathbf{W}^{[l]}$ | Weights of layer $l$                |
| $\mathbf{b}^{[l]}$ | Bias of layer $l$                   |
| $\mathbf{z}^{[l]}$ | Linear combination (pre-activation) |
| $\mathbf{a}^{[l]}$ | Activation of layer $l$             |
| $\mathcal{L}$      | Loss function                       |
| $\eta$             | Learning rate                       |
| $\nabla$           | Gradient operator                   |
| $\odot$            | Element-wise multiplication         |
| $\delta^{[l]}$     | Error term for layer $l$            |

---

## 12. References

1. **Deep Learning** - Ian Goodfellow, Yoshua Bengio, Aaron Courville
2. **Neural Networks and Deep Learning** - Michael Nielsen (online book)
3. **CS231n: CNNs for Visual Recognition** - Stanford Course Notes
4. **Mathematics for Machine Learning** - Deisenroth, Faisal, Ong

---

**Document Status**: ✅ Complete  
**Related Documents**:

- [NEURAL_NETWORK_DESIGN.md](NEURAL_NETWORK_DESIGN.md)
- [TRAINING_STRATEGY.md](TRAINING_STRATEGY.md)
