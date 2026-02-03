"""
MNIST Dataset Loader

Downloads and loads the MNIST dataset for training and testing.
"""

import numpy as np
import gzip
import os
import urllib.request
from pathlib import Path
from typing import Tuple, Optional


class MNISTLoader:
    """
    MNIST Dataset Loader.
    
    Downloads and loads the MNIST handwritten digit dataset.
    The dataset contains 60,000 training images and 10,000 test images.
    
    Attributes:
        data_dir: Directory to store downloaded data
        
    Example:
        >>> loader = MNISTLoader()
        >>> (X_train, y_train), (X_test, y_test) = loader.load()
        >>> print(X_train.shape)  # (60000, 784)
        >>> print(y_train.shape)  # (60000,)
    """
    
    # MNIST dataset URLs (Yann LeCun's website mirror)
    BASE_URL = "http://yann.lecun.com/exdb/mnist/"
    
    FILES = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    # Alternative mirror (more reliable)
    MIRROR_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    
    def __init__(self, data_dir: str = "data/mnist"):
        """
        Initialize MNIST loader.
        
        Args:
            data_dir: Directory to store downloaded files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _download_file(self, filename: str) -> Path:
        """
        Download a single file if not exists.
        
        Args:
            filename: Name of file to download
            
        Returns:
            Path to downloaded file
        """
        filepath = self.data_dir / filename
        
        if filepath.exists():
            return filepath
        
        # Try mirror first (more reliable)
        urls = [
            self.MIRROR_URL + filename,
            self.BASE_URL + filename
        ]
        
        for url in urls:
            try:
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
                print(f"Downloaded {filename}")
                return filepath
            except Exception as e:
                print(f"Failed to download from {url}: {e}")
                continue
        
        raise RuntimeError(f"Failed to download {filename} from all sources")
    
    def _load_images(self, filepath: Path) -> np.ndarray:
        """
        Load images from gzipped IDX file.
        
        Args:
            filepath: Path to gzipped images file
            
        Returns:
            Images array of shape (N, 784)
        """
        with gzip.open(filepath, 'rb') as f:
            # Read header
            magic = int.from_bytes(f.read(4), 'big')
            if magic != 2051:
                raise ValueError(f"Invalid magic number for images: {magic}")
            
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            
            # Read image data
            data = np.frombuffer(f.read(), dtype=np.uint8)
            images = data.reshape(num_images, rows * cols)
        
        return images
    
    def _load_labels(self, filepath: Path) -> np.ndarray:
        """
        Load labels from gzipped IDX file.
        
        Args:
            filepath: Path to gzipped labels file
            
        Returns:
            Labels array of shape (N,)
        """
        with gzip.open(filepath, 'rb') as f:
            # Read header
            magic = int.from_bytes(f.read(4), 'big')
            if magic != 2049:
                raise ValueError(f"Invalid magic number for labels: {magic}")
            
            num_labels = int.from_bytes(f.read(4), 'big')
            
            # Read label data
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        return labels
    
    def download(self) -> None:
        """Download all MNIST files."""
        for name, filename in self.FILES.items():
            self._download_file(filename)
        print("All MNIST files downloaded successfully!")
    
    def load(self, normalize: bool = True, 
             flatten: bool = True,
             one_hot: bool = False) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                              Tuple[np.ndarray, np.ndarray]]:
        """
        Load MNIST dataset.
        
        Args:
            normalize: Normalize pixel values to [0, 1]
            flatten: Flatten images to (N, 784), else (N, 28, 28)
            one_hot: One-hot encode labels
            
        Returns:
            ((X_train, y_train), (X_test, y_test))
        """
        # Download if needed
        for filename in self.FILES.values():
            filepath = self.data_dir / filename
            if not filepath.exists():
                self._download_file(filename)
        
        # Load data
        X_train = self._load_images(self.data_dir / self.FILES['train_images'])
        y_train = self._load_labels(self.data_dir / self.FILES['train_labels'])
        X_test = self._load_images(self.data_dir / self.FILES['test_images'])
        y_test = self._load_labels(self.data_dir / self.FILES['test_labels'])
        
        # Convert to float
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        
        # Normalize
        if normalize:
            X_train = X_train / 255.0
            X_test = X_test / 255.0
        
        # Reshape
        if not flatten:
            X_train = X_train.reshape(-1, 28, 28)
            X_test = X_test.reshape(-1, 28, 28)
        
        # One-hot encode
        if one_hot:
            y_train = self._one_hot_encode(y_train, 10)
            y_test = self._one_hot_encode(y_test, 10)
        
        return (X_train, y_train), (X_test, y_test)
    
    def _one_hot_encode(self, labels: np.ndarray, 
                        num_classes: int) -> np.ndarray:
        """One-hot encode labels."""
        one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
        one_hot[np.arange(len(labels)), labels] = 1
        return one_hot
    
    def load_sample(self, n_samples: int = 1000,
                    normalize: bool = True,
                    one_hot: bool = False) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                     Tuple[np.ndarray, np.ndarray]]:
        """
        Load a smaller sample of MNIST for quick testing.
        
        Args:
            n_samples: Number of samples to load from each set
            normalize: Normalize pixel values
            one_hot: One-hot encode labels
            
        Returns:
            ((X_train, y_train), (X_test, y_test))
        """
        (X_train, y_train), (X_test, y_test) = self.load(
            normalize=normalize, one_hot=one_hot
        )
        
        # Shuffle and sample
        np.random.seed(42)
        train_idx = np.random.choice(len(X_train), min(n_samples, len(X_train)), replace=False)
        test_idx = np.random.choice(len(X_test), min(n_samples // 6, len(X_test)), replace=False)
        
        return (X_train[train_idx], y_train[train_idx]), (X_test[test_idx], y_test[test_idx])


def split_data(X: np.ndarray, y: np.ndarray, 
               val_split: float = 0.1,
               shuffle: bool = True,
               random_seed: Optional[int] = 42) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                          Tuple[np.ndarray, np.ndarray]]:
    """
    Split data into training and validation sets.
    
    Args:
        X: Feature array
        y: Label array
        val_split: Fraction of data for validation
        shuffle: Whether to shuffle before splitting
        random_seed: Random seed for reproducibility
        
    Returns:
        ((X_train, y_train), (X_val, y_val))
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    val_size = int(n_samples * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    return (X[train_indices], y[train_indices]), (X[val_indices], y[val_indices])


# Convenience exports
__all__ = ['MNISTLoader', 'split_data']
