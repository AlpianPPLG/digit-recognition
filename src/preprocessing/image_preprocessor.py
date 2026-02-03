"""
Image Preprocessor

Preprocessing pipeline for converting images to neural network input format.
Handles canvas captures, file uploads, and various image transformations.
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union
from pathlib import Path


class ImagePreprocessor:
    """
    Image preprocessing pipeline for digit recognition.
    
    Converts input images from various sources into the format
    expected by the neural network: (784,) normalized array.
    
    The pipeline:
    1. Convert to grayscale
    2. Resize to 28x28
    3. Center the digit
    4. Normalize to [0, 1]
    5. Invert if needed (MNIST has white digits on black)
    6. Flatten to (784,)
    
    Example:
        >>> preprocessor = ImagePreprocessor()
        >>> image = preprocessor.load_image("digit.png")
        >>> processed = preprocessor.preprocess(image)
        >>> print(processed.shape)  # (784,)
    """
    
    TARGET_SIZE = (28, 28)
    
    def __init__(self, 
                 invert: bool = True,
                 center_digit: bool = True,
                 add_padding: bool = True):
        """
        Initialize preprocessor.
        
        Args:
            invert: Invert colors (for white-on-black MNIST format)
            center_digit: Center the digit in the image
            add_padding: Add padding around digit before resize
        """
        self.invert = invert
        self.center_digit = center_digit
        self.add_padding = add_padding
    
    def load_image(self, path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file.
        
        Args:
            path: Path to image file
            
        Returns:
            Grayscale image as numpy array
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        img = Image.open(path).convert('L')
        return np.array(img)
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.
        
        Args:
            image: Input image (H, W) or (H, W, C)
            
        Returns:
            Grayscale image (H, W)
        """
        if len(image.shape) == 2:
            return image
        
        if image.shape[2] == 4:
            # RGBA -> RGB
            image = image[:, :, :3]
        
        # Luminosity method
        if image.shape[2] == 3:
            weights = np.array([0.299, 0.587, 0.114])
            gray = np.dot(image, weights)
            return gray.astype(np.uint8)
        
        return image
    
    def resize(self, image: np.ndarray, 
               target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target (height, width), defaults to (28, 28)
            
        Returns:
            Resized image
        """
        if target_size is None:
            target_size = self.TARGET_SIZE
        
        pil_img = Image.fromarray(image.astype(np.uint8))
        resized = pil_img.resize((target_size[1], target_size[0]), Image.LANCZOS)
        return np.array(resized)
    
    def resize_with_aspect_ratio(self, image: np.ndarray,
                                  target_size: Tuple[int, int] = None,
                                  padding_value: int = 0) -> np.ndarray:
        """
        Resize maintaining aspect ratio with padding.
        
        Args:
            image: Input image
            target_size: Target (height, width)
            padding_value: Value for padding (0 = black)
            
        Returns:
            Resized and padded image
        """
        if target_size is None:
            target_size = self.TARGET_SIZE
        
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scaling factor
        scale = min(target_h / h, target_w / w) * 0.9  # Leave some margin
        
        # New dimensions
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        
        # Resize
        pil_img = Image.fromarray(image.astype(np.uint8))
        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        resized = np.array(resized)
        
        # Create output with padding
        output = np.full(target_size, padding_value, dtype=np.uint8)
        
        # Center the digit
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        
        output[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        return output
    
    def find_bounding_box(self, image: np.ndarray, 
                          threshold: int = 10) -> Tuple[int, int, int, int]:
        """
        Find bounding box of digit content.
        
        Args:
            image: Grayscale image
            threshold: Pixel value threshold for content detection
            
        Returns:
            (y_min, y_max, x_min, x_max) bounding box
        """
        # Find non-background pixels
        if self.invert:
            # For white on black, find pixels > threshold
            mask = image > threshold
        else:
            # For black on white, find pixels < (255 - threshold)
            mask = image < (255 - threshold)
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            # No content found, return full image
            return 0, image.shape[0], 0, image.shape[1]
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return y_min, y_max + 1, x_min, x_max + 1
    
    def crop_to_content(self, image: np.ndarray,
                        padding: int = 4) -> np.ndarray:
        """
        Crop image to digit content with padding.
        
        Args:
            image: Input image
            padding: Pixels of padding around content
            
        Returns:
            Cropped image
        """
        y_min, y_max, x_min, x_max = self.find_bounding_box(image)
        
        # Add padding
        y_min = max(0, y_min - padding)
        y_max = min(image.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        
        return image[y_min:y_max, x_min:x_max]
    
    def center_of_mass(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Calculate center of mass of digit.
        
        Args:
            image: Grayscale image
            
        Returns:
            (cy, cx) center of mass coordinates
        """
        h, w = image.shape
        
        # Normalize image
        img_norm = image.astype(np.float32)
        if img_norm.max() > 0:
            img_norm = img_norm / img_norm.max()
        
        total = img_norm.sum()
        if total == 0:
            return h / 2, w / 2
        
        # Calculate center of mass
        y_indices, x_indices = np.ogrid[:h, :w]
        cy = np.sum(y_indices * img_norm) / total
        cx = np.sum(x_indices * img_norm) / total
        
        return cy, cx
    
    def center_image(self, image: np.ndarray) -> np.ndarray:
        """
        Center digit based on center of mass.
        
        Args:
            image: Input image (28x28)
            
        Returns:
            Centered image
        """
        h, w = image.shape
        cy, cx = self.center_of_mass(image)
        
        # Calculate shift to center
        shift_y = int(h / 2 - cy)
        shift_x = int(w / 2 - cx)
        
        # Apply shift
        output = np.zeros_like(image)
        
        # Calculate valid source and destination ranges
        src_y_start = max(0, -shift_y)
        src_y_end = min(h, h - shift_y)
        src_x_start = max(0, -shift_x)
        src_x_end = min(w, w - shift_x)
        
        dst_y_start = max(0, shift_y)
        dst_y_end = min(h, h + shift_y)
        dst_x_start = max(0, shift_x)
        dst_x_end = min(w, w + shift_x)
        
        output[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            image[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return output
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [0, 1].
        
        Args:
            image: Input image
            
        Returns:
            Normalized image as float32
        """
        img_float = image.astype(np.float32)
        
        if img_float.max() > 1:
            img_float = img_float / 255.0
        
        return img_float
    
    def invert_colors(self, image: np.ndarray) -> np.ndarray:
        """
        Invert image colors.
        
        Args:
            image: Input image
            
        Returns:
            Inverted image
        """
        if image.max() <= 1:
            return 1.0 - image
        return 255 - image
    
    def flatten(self, image: np.ndarray) -> np.ndarray:
        """
        Flatten image to 1D array.
        
        Args:
            image: 2D image
            
        Returns:
            1D array of shape (784,)
        """
        return image.flatten()
    
    def preprocess(self, image: np.ndarray,
                   return_2d: bool = False) -> np.ndarray:
        """
        Full preprocessing pipeline.
        
        Args:
            image: Input image (any size, RGB or grayscale)
            return_2d: If True, return (28, 28) instead of (784,)
            
        Returns:
            Preprocessed image ready for neural network
        """
        # 1. Convert to grayscale
        img = self.to_grayscale(image)
        
        # 2. Crop to content if centering enabled
        if self.center_digit and self.add_padding:
            img = self.crop_to_content(img)
        
        # 3. Resize to 28x28
        if self.add_padding:
            img = self.resize_with_aspect_ratio(img)
        else:
            img = self.resize(img)
        
        # 4. Invert if needed (MNIST format)
        if self.invert:
            # Check if image has dark background
            if img.mean() > 127:
                # Light background - invert
                img = 255 - img
        
        # 5. Center based on center of mass
        if self.center_digit:
            img = self.center_image(img)
        
        # 6. Normalize to [0, 1]
        img = self.normalize(img)
        
        # 7. Flatten
        if not return_2d:
            img = self.flatten(img)
        
        return img
    
    def preprocess_batch(self, images: list) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of images
            
        Returns:
            Array of shape (N, 784)
        """
        processed = [self.preprocess(img) for img in images]
        return np.array(processed)
    
    def from_canvas(self, canvas_data: np.ndarray) -> np.ndarray:
        """
        Preprocess image from GUI canvas.
        
        Canvas typically has white background and black strokes.
        
        Args:
            canvas_data: Canvas image data
            
        Returns:
            Preprocessed array (784,)
        """
        # Canvas usually has light background
        return self.preprocess(canvas_data)


class DataAugmenter:
    """
    Data augmentation for training.
    
    Applies random transformations to increase dataset variety.
    """
    
    def __init__(self, 
                 rotation_range: float = 15,
                 shift_range: float = 0.1,
                 zoom_range: float = 0.1,
                 noise_factor: float = 0.1):
        """
        Initialize augmenter.
        
        Args:
            rotation_range: Max rotation in degrees
            shift_range: Max shift as fraction of image size
            zoom_range: Max zoom factor variation
            noise_factor: Max noise standard deviation
        """
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range
        self.noise_factor = noise_factor
    
    def rotate(self, image: np.ndarray, angle: float = None) -> np.ndarray:
        """Apply random rotation."""
        if angle is None:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        
        # Reshape to 28x28 if flattened
        if image.ndim == 1:
            image = image.reshape(28, 28)
            flatten_output = True
        else:
            flatten_output = False
        
        # Rotate using PIL
        pil_img = Image.fromarray((image * 255).astype(np.uint8))
        rotated = pil_img.rotate(angle, fillcolor=0)
        result = np.array(rotated).astype(np.float32) / 255.0
        
        if flatten_output:
            result = result.flatten()
        
        return result
    
    def shift(self, image: np.ndarray, 
              shift_x: float = None, 
              shift_y: float = None) -> np.ndarray:
        """Apply random shift."""
        if image.ndim == 1:
            image = image.reshape(28, 28)
            flatten_output = True
        else:
            flatten_output = False
        
        h, w = image.shape
        
        if shift_x is None:
            shift_x = np.random.uniform(-self.shift_range, self.shift_range) * w
        if shift_y is None:
            shift_y = np.random.uniform(-self.shift_range, self.shift_range) * h
        
        shift_x, shift_y = int(shift_x), int(shift_y)
        
        result = np.zeros_like(image)
        
        # Calculate source and dest ranges
        src_y = slice(max(0, -shift_y), min(h, h - shift_y))
        src_x = slice(max(0, -shift_x), min(w, w - shift_x))
        dst_y = slice(max(0, shift_y), min(h, h + shift_y))
        dst_x = slice(max(0, shift_x), min(w, w + shift_x))
        
        result[dst_y, dst_x] = image[src_y, src_x]
        
        if flatten_output:
            result = result.flatten()
        
        return result
    
    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add random Gaussian noise."""
        noise = np.random.normal(0, self.noise_factor, image.shape)
        noisy = image + noise
        return np.clip(noisy, 0, 1).astype(np.float32)
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations."""
        result = image.copy()
        
        # Random rotation
        if np.random.random() < 0.5:
            result = self.rotate(result)
        
        # Random shift
        if np.random.random() < 0.5:
            result = self.shift(result)
        
        # Random noise
        if np.random.random() < 0.3:
            result = self.add_noise(result)
        
        return result
    
    def augment_batch(self, images: np.ndarray, 
                      augment_factor: int = 2) -> np.ndarray:
        """
        Augment a batch of images.
        
        Args:
            images: Array of shape (N, 784)
            augment_factor: How many augmented versions per image
            
        Returns:
            Augmented array of shape (N * augment_factor, 784)
        """
        augmented = [images]  # Include originals
        
        for _ in range(augment_factor - 1):
            aug_batch = np.array([self.augment(img) for img in images])
            augmented.append(aug_batch)
        
        return np.vstack(augmented)


# Convenience exports
__all__ = ['ImagePreprocessor', 'DataAugmenter']
