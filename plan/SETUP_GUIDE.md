# 🛠️ Setup Guide - Digit Recognition

**Version**: 1.0  
**Date**: 1 Feb 2026  
**Status**: Planning

---

## 1. System Requirements

### 1.1 Minimum Requirements

| Component      | Minimum                                 | Recommended                          |
| -------------- | --------------------------------------- | ------------------------------------ |
| **OS**         | Windows 10 / macOS 10.14 / Ubuntu 18.04 | Windows 11 / macOS 13 / Ubuntu 22.04 |
| **Python**     | 3.10                                    | 3.11+                                |
| **RAM**        | 4 GB                                    | 8 GB                                 |
| **Disk Space** | 500 MB                                  | 1 GB                                 |
| **Display**    | 1280 x 720                              | 1920 x 1080                          |

### 1.2 Software Requirements

```
┌─────────────────────────────────────────────────────────────┐
│                 SOFTWARE REQUIREMENTS                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐                                        │
│  │   Python 3.10+  │  ← Core runtime                        │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ├──► NumPy 1.24+      (numerical computing)       │
│           ├──► Pillow 9.0+      (image processing)          │
│           ├──► Tkinter          (GUI framework)             │
│           └──► Matplotlib 3.6+  (visualization)             │
│                                                             │
│  Optional:                                                  │
│  ├──► CustomTkinter 5.0+  (modern UI)                       │
│  └──► pytest 7.0+         (testing)                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Installation Methods

### 2.1 Method 1: Quick Install (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/username/digit-recognition.git
cd digit-recognition

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run application
python main.py
```

### 2.2 Method 2: Manual Installation

#### Step 1: Install Python

**Windows:**

```powershell
# Download from python.org or use winget
winget install Python.Python.3.11

# Verify installation
python --version
```

**macOS:**

```bash
# Using Homebrew
brew install python@3.11

# Verify
python3 --version
```

**Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip python3-tk
```

#### Step 2: Create Virtual Environment

```bash
# Create
python -m venv digit_recognition_env

# Activate (Windows)
digit_recognition_env\Scripts\activate

# Activate (macOS/Linux)
source digit_recognition_env/bin/activate

# Verify
which python  # Should show venv path
```

#### Step 3: Install Dependencies

```bash
# Core dependencies
pip install numpy>=1.24.0
pip install pillow>=9.0.0
pip install matplotlib>=3.6.0

# Optional: Modern UI
pip install customtkinter>=5.0.0

# Optional: Testing
pip install pytest>=7.0.0
pip install pytest-cov>=4.0.0
```

#### Step 4: Verify Installation

```python
# verify_install.py
import sys
print(f"Python: {sys.version}")

import numpy as np
print(f"NumPy: {np.__version__}")

from PIL import Image
import PIL
print(f"Pillow: {PIL.__version__}")

import matplotlib
print(f"Matplotlib: {matplotlib.__version__}")

import tkinter as tk
print("Tkinter: OK")

try:
    import customtkinter as ctk
    print(f"CustomTkinter: {ctk.__version__}")
except ImportError:
    print("CustomTkinter: Not installed (optional)")

print("\n✅ All core dependencies installed!")
```

Run verification:

```bash
python verify_install.py
```

---

## 3. Project Setup

### 3.1 Directory Structure

Create the following directory structure:

```bash
# Create directories
mkdir -p digit_recognition/{core,data,gui,preprocessing,tests,models}

# Create __init__.py files
touch digit_recognition/__init__.py
touch digit_recognition/core/__init__.py
touch digit_recognition/data/__init__.py
touch digit_recognition/gui/__init__.py
touch digit_recognition/preprocessing/__init__.py
touch digit_recognition/tests/__init__.py
```

Result:

```
digit_recognition/
├── __init__.py
├── main.py
├── requirements.txt
├── README.md
├── core/
│   ├── __init__.py
│   ├── layers.py
│   ├── activations.py
│   ├── loss.py
│   ├── optimizers.py
│   ├── network.py
│   └── trainer.py
├── data/
│   ├── __init__.py
│   ├── mnist_loader.py
│   └── mnist/
│       ├── train-images-idx3-ubyte.gz
│       ├── train-labels-idx1-ubyte.gz
│       ├── t10k-images-idx3-ubyte.gz
│       └── t10k-labels-idx1-ubyte.gz
├── gui/
│   ├── __init__.py
│   ├── main_window.py
│   ├── canvas.py
│   └── components.py
├── preprocessing/
│   ├── __init__.py
│   └── pipeline.py
├── models/
│   └── pretrained.pkl
└── tests/
    ├── __init__.py
    ├── conftest.py
    └── test_*.py
```

### 3.2 Requirements File

```txt
# requirements.txt

# Core dependencies
numpy>=1.24.0,<2.0.0
pillow>=9.0.0,<11.0.0
matplotlib>=3.6.0,<4.0.0

# GUI (optional but recommended)
customtkinter>=5.0.0,<6.0.0

# Development dependencies
pytest>=7.0.0
pytest-cov>=4.0.0
flake8>=6.0.0

# Documentation
sphinx>=6.0.0
```

### 3.3 Setup.py (For Package Installation)

```python
# setup.py

from setuptools import setup, find_packages

setup(
    name="digit-recognition",
    version="1.0.0",
    author="Your Name",
    author_email="email@example.com",
    description="Digit recognition using neural network from scratch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/digit-recognition",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pillow>=9.0.0",
        "matplotlib>=3.6.0",
    ],
    extras_require={
        "gui": ["customtkinter>=5.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "digit-recognition=digit_recognition.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
```

---

## 4. MNIST Dataset Setup

### 4.1 Automatic Download

```python
# data/mnist_downloader.py

import os
import gzip
import urllib.request
from pathlib import Path

MNIST_URLS = {
    'train-images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'train-labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'test-images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'test-labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
}

def download_mnist(data_dir: str = 'data/mnist'):
    """Download MNIST dataset if not exists"""

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    for name, url in MNIST_URLS.items():
        filename = url.split('/')[-1]
        filepath = data_path / filename

        if filepath.exists():
            print(f"✓ {filename} already exists")
            continue

        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"✓ {filename} downloaded")

    print("\n✅ MNIST dataset ready!")

if __name__ == "__main__":
    download_mnist()
```

### 4.2 Manual Download

1. Visit: http://yann.lecun.com/exdb/mnist/
2. Download all 4 files:
   - `train-images-idx3-ubyte.gz`
   - `train-labels-idx1-ubyte.gz`
   - `t10k-images-idx3-ubyte.gz`
   - `t10k-labels-idx1-ubyte.gz`
3. Place in `data/mnist/` folder

### 4.3 Verify Dataset

```python
# verify_mnist.py

import gzip
import numpy as np
from pathlib import Path

def verify_mnist(data_dir: str = 'data/mnist'):
    """Verify MNIST dataset files"""

    data_path = Path(data_dir)

    files = {
        'train-images-idx3-ubyte.gz': (60000, 28, 28),
        'train-labels-idx1-ubyte.gz': (60000,),
        't10k-images-idx3-ubyte.gz': (10000, 28, 28),
        't10k-labels-idx1-ubyte.gz': (10000,),
    }

    print("Verifying MNIST dataset...")
    print("-" * 40)

    all_ok = True
    for filename, expected_shape in files.items():
        filepath = data_path / filename

        if not filepath.exists():
            print(f"❌ {filename}: NOT FOUND")
            all_ok = False
            continue

        # Check file size
        size_kb = filepath.stat().st_size / 1024

        # Try to read
        try:
            with gzip.open(filepath, 'rb') as f:
                if 'images' in filename:
                    magic = int.from_bytes(f.read(4), 'big')
                    n_items = int.from_bytes(f.read(4), 'big')
                    rows = int.from_bytes(f.read(4), 'big')
                    cols = int.from_bytes(f.read(4), 'big')
                    actual_shape = (n_items, rows, cols)
                else:
                    magic = int.from_bytes(f.read(4), 'big')
                    n_items = int.from_bytes(f.read(4), 'big')
                    actual_shape = (n_items,)

            if actual_shape == expected_shape:
                print(f"✅ {filename}: OK ({size_kb:.0f} KB)")
            else:
                print(f"⚠️ {filename}: Shape mismatch {actual_shape} != {expected_shape}")
                all_ok = False

        except Exception as e:
            print(f"❌ {filename}: Error reading - {e}")
            all_ok = False

    print("-" * 40)
    if all_ok:
        print("✅ All MNIST files verified!")
    else:
        print("❌ Some files have issues. Please re-download.")

    return all_ok

if __name__ == "__main__":
    verify_mnist()
```

---

## 5. IDE Setup

### 5.1 VS Code Setup

**Recommended Extensions:**

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.black-formatter",
    "ms-toolsai.jupyter",
    "njpwerner.autodocstring"
  ]
}
```

**Settings (.vscode/settings.json):**

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.analysis.typeCheckingMode": "basic",
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.rulers": [88],
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"]
}
```

**Launch Configuration (.vscode/launch.json):**

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run Application",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/main.py",
      "console": "integratedTerminal"
    },
    {
      "name": "Run Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["-v"],
      "console": "integratedTerminal"
    },
    {
      "name": "Debug Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal"
    }
  ]
}
```

### 5.2 PyCharm Setup

1. **Open Project:**
   - File → Open → Select project folder

2. **Configure Interpreter:**
   - File → Settings → Project → Python Interpreter
   - Add Interpreter → Local Interpreter
   - Select venv

3. **Install Plugins:**
   - Material Theme UI
   - Rainbow Brackets
   - .ignore

4. **Run Configuration:**
   - Run → Edit Configurations
   - Add Python
   - Script: main.py

---

## 6. Environment Configuration

### 6.1 Environment Variables

```bash
# .env file (optional)

# Data directory
MNIST_DATA_DIR=./data/mnist

# Model directory
MODEL_DIR=./models

# Logging level
LOG_LEVEL=INFO

# GUI theme
GUI_THEME=dark
```

### 6.2 Configuration File

```python
# config.py

import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    """Application configuration"""

    # Paths
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / 'data' / 'mnist'
    MODEL_DIR: Path = BASE_DIR / 'models'

    # Network defaults
    DEFAULT_ARCHITECTURE: list = None
    DEFAULT_LEARNING_RATE: float = 0.001
    DEFAULT_BATCH_SIZE: int = 32
    DEFAULT_EPOCHS: int = 20

    # GUI
    WINDOW_WIDTH: int = 1000
    WINDOW_HEIGHT: int = 700
    CANVAS_SIZE: int = 280
    THEME: str = 'dark'

    def __post_init__(self):
        if self.DEFAULT_ARCHITECTURE is None:
            self.DEFAULT_ARCHITECTURE = [784, 128, 64, 10]

        # Create directories
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls):
        """Create config from environment variables"""
        return cls(
            DATA_DIR=Path(os.getenv('MNIST_DATA_DIR', 'data/mnist')),
            MODEL_DIR=Path(os.getenv('MODEL_DIR', 'models')),
            THEME=os.getenv('GUI_THEME', 'dark'),
        )

# Global config instance
config = Config()
```

---

## 7. Development Environment

### 7.1 Git Setup

```bash
# Initialize git
git init

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
*.egg

# Virtual environment
venv/
env/
.venv/

# IDE
.idea/
.vscode/
*.swp

# Data
data/mnist/*.gz
*.pkl
*.h5

# Testing
.coverage
htmlcov/
.pytest_cache/

# OS
.DS_Store
Thumbs.db

# Logs
*.log
EOF

# Initial commit
git add .
git commit -m "Initial project setup"
```

### 7.2 Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88]
EOF

# Install hooks
pre-commit install
```

---

## 8. Troubleshooting

### 8.1 Common Issues

#### Issue: `ModuleNotFoundError: No module named 'tkinter'`

**Solution (Ubuntu/Debian):**

```bash
sudo apt install python3-tk
```

**Solution (macOS):**

```bash
brew install python-tk
```

**Solution (Windows):**
Re-install Python with "tcl/tk and IDLE" option checked.

---

#### Issue: `ImportError: numpy.core.multiarray failed to import`

**Solution:**

```bash
pip uninstall numpy
pip install numpy --force-reinstall
```

---

#### Issue: Slow training on large dataset

**Check:**

1. Ensure using NumPy operations (not Python loops)
2. Reduce batch size if memory limited
3. Consider using smaller network for testing

---

#### Issue: GUI not responding during training

**Solution:**
Training should run in a separate thread:

```python
import threading

def start_training():
    thread = threading.Thread(target=train_model, daemon=True)
    thread.start()
```

---

#### Issue: MNIST download fails

**Solution:**

1. Check internet connection
2. Try alternative mirror:
   ```python
   # Alternative: AWS mirror
   'https://s3.amazonaws.com/img-datasets/mnist/'
   ```
3. Download manually and place in data/mnist/

---

### 8.2 Verification Script

```python
# verify_setup.py

import sys
import platform

def check_python():
    """Check Python version"""
    v = sys.version_info
    if v.major >= 3 and v.minor >= 10:
        print(f"✅ Python {v.major}.{v.minor}.{v.micro}")
        return True
    else:
        print(f"❌ Python {v.major}.{v.minor} (need 3.10+)")
        return False

def check_package(name, min_version=None):
    """Check if package is installed"""
    try:
        pkg = __import__(name)
        version = getattr(pkg, '__version__', 'unknown')
        print(f"✅ {name} {version}")
        return True
    except ImportError:
        print(f"❌ {name} not installed")
        return False

def check_tkinter():
    """Check Tkinter availability"""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.destroy()
        print("✅ Tkinter working")
        return True
    except Exception as e:
        print(f"❌ Tkinter error: {e}")
        return False

def check_mnist():
    """Check MNIST data"""
    from pathlib import Path
    data_dir = Path('data/mnist')
    files = list(data_dir.glob('*.gz'))
    if len(files) >= 4:
        print(f"✅ MNIST data ({len(files)} files)")
        return True
    else:
        print(f"⚠️ MNIST data missing ({len(files)}/4 files)")
        return False

def main():
    print("=" * 50)
    print("SETUP VERIFICATION")
    print("=" * 50)
    print(f"OS: {platform.system()} {platform.release()}")
    print("-" * 50)

    checks = [
        check_python(),
        check_package('numpy'),
        check_package('PIL'),
        check_package('matplotlib'),
        check_tkinter(),
        check_mnist(),
    ]

    print("-" * 50)
    if all(checks):
        print("✅ All checks passed! Ready to run.")
        return 0
    else:
        print("⚠️ Some checks failed. Please fix issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

---

## 9. Quick Start After Setup

### 9.1 Running the Application

```bash
# 1. Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Download MNIST (if not done)
python -c "from data.mnist_downloader import download_mnist; download_mnist()"

# 3. Run application
python main.py
```

### 9.2 Training a New Model

```python
# train_model.py

from core.network import NetworkBuilder
from core.trainer import Trainer
from core.optimizers import Adam
from core.loss import CrossEntropyLoss
from data.mnist_loader import MNISTLoader

# Load data
print("Loading MNIST...")
loader = MNISTLoader()
X_train, y_train, X_test, y_test = loader.load()

# Build network
print("Building network...")
network = NetworkBuilder() \
    .input(784) \
    .dense(128, activation='relu') \
    .dense(64, activation='relu') \
    .dense(10, activation='softmax') \
    .build()

# Train
print("Training...")
trainer = Trainer(network, Adam(0.001), CrossEntropyLoss())
history = trainer.train(
    X_train, y_train,
    X_val=X_test, y_val=y_test,
    epochs=20,
    batch_size=32
)

# Save model
network.save('models/trained_model.pkl')
print("Model saved!")
```

### 9.3 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov-report=html

# Run specific test
pytest tests/test_layers.py -v
```

---

**Document Status**: ✅ Complete  
**Related Documents**:

- [USER_GUIDE.md](USER_GUIDE.md)
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
