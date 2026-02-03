"""
GUI Module - User Interface Components

This module contains all GUI-related components:
- Main application window
- Drawing canvas for digit input
- Image upload interface
- Webcam capture module
- Training dashboard
- Prediction display
"""

from .main_window import MainWindow
from .drawing_canvas import DrawingCanvas, DigitPreview
from .app import DigitRecognitionApp, main

__all__ = [
    'MainWindow',
    'DrawingCanvas',
    'DigitPreview',
    'DigitRecognitionApp',
    'main'
]
