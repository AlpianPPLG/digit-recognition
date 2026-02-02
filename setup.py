#!/usr/bin/env python
"""
Setup script for Digit Recognition package.

This file is maintained for backward compatibility with older pip versions.
Modern installations should use pyproject.toml.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")
else:
    long_description = "Handwritten digit recognition using neural networks built from scratch"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#") and not line.startswith("-")
    ]
else:
    requirements = [
        "numpy>=1.24.0,<2.0.0",
        "Pillow>=10.0.0",
        "customtkinter>=5.2.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ]

setup(
    name="digit-recognition",
    version="1.0.0",
    author="AiDigit Team",
    description="Handwritten digit recognition using neural networks built from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlpianPPLG/digit-recognition",
    license="MIT",
    
    # Package discovery
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    # Python version requirement
    python_requires=">=3.10",
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "mypy>=1.5.0",
            "flake8>=6.1.0",
            "black>=23.9.0",
            "isort>=5.12.0",
        ],
        "webcam": [
            "opencv-python>=4.8.0",
        ],
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "digit-recognition=main:main",
            "digit-train=train:main",
            "digit-predict=predict:main",
        ],
    },
    
    # Package data
    include_package_data=True,
    package_data={
        "": ["*.json"],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    
    # Keywords
    keywords=[
        "machine-learning",
        "neural-network", 
        "digit-recognition",
        "mnist",
        "from-scratch",
        "educational",
    ],
)
