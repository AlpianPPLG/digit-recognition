#!/usr/bin/env python
"""
Download MNIST Dataset

This script downloads the MNIST dataset from the official source
and saves it to the data/mnist directory.

Usage:
    python scripts/download_mnist.py
"""

import os
import gzip
import urllib.request
from pathlib import Path


BASE_URL = "http://yann.lecun.com/exdb/mnist/"

FILES = {
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
    'test_images': 't10k-images-idx3-ubyte.gz',
    'test_labels': 't10k-labels-idx1-ubyte.gz'
}


def download_mnist(data_dir: str = 'data/mnist'):
    """Download MNIST dataset files"""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    for name, filename in FILES.items():
        filepath = data_path / filename
        
        if filepath.exists():
            print(f"✓ {filename} already exists")
            continue
            
        print(f"Downloading {filename}...")
        url = BASE_URL + filename
        
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            return False
    
    print("\n✓ MNIST dataset download complete!")
    return True


if __name__ == "__main__":
    # Get project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / 'data' / 'mnist'
    
    download_mnist(str(data_dir))
