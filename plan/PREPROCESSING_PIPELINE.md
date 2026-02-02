# 🖼️ Preprocessing Pipeline - Digit Recognition

**Version**: 1.0  
**Date**: 1 Feb 2026  
**Status**: Planning

---

## 1. Overview

Pipeline preprocessing adalah tahap krusial yang mengubah input mentah (gambar dari canvas, file, atau webcam) menjadi format yang dapat diproses oleh neural network. Kualitas preprocessing sangat mempengaruhi akurasi prediksi.

### 1.1 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING PIPELINE                               │
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│  │  INPUT   │    │ CONVERT  │    │  RESIZE  │    │  CENTER  │               │
│  │  IMAGE   │───►│GRAYSCALE │───►│  28x28   │───►│  DIGIT   │               │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘               │
│                                                        │                    │
│                                                        ▼                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│  │  OUTPUT  │    │ FLATTEN  │    │  INVERT  │    │NORMALIZE │               │
│  │  (784,)  │◄───│  ARRAY   │◄───│ (if need)│◄───│  [0,1]   │               │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Input Sources

| Source | Format | Resolution | Color |
|--------|--------|------------|-------|
| Canvas GUI | PIL Image | 280x280 | RGB/Grayscale |
| Image Upload | PNG/JPG/BMP | Variable | RGB/Grayscale |
| Webcam | OpenCV Frame | Variable | BGR |
| MNIST Dataset | NumPy Array | 28x28 | Grayscale |

### 1.3 Output Format

| Property | Value | Description |
|----------|-------|-------------|
| Shape | (784,) atau (1, 784) | Flattened 28x28 |
| Data Type | float32 | Untuk komputasi |
| Value Range | [0, 1] | Normalized |
| Background | 0 (hitam) | MNIST convention |
| Foreground | 1 (putih) | Digit stroke |

---

## 2. Image Acquisition

### 2.1 Canvas Capture

```python
import numpy as np
from PIL import Image, ImageGrab
import tkinter as tk

class CanvasCapture:
    """Capture drawing from Tkinter canvas"""
    
    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
    
    def capture(self) -> np.ndarray:
        """
        Capture canvas content as numpy array
        
        Returns:
            image: Grayscale image array (H, W)
        """
        # Method 1: Using PostScript (cross-platform)
        ps = self.canvas.postscript(colormode='gray')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        
        # Method 2: Using ImageGrab (Windows)
        # x = self.canvas.winfo_rootx()
        # y = self.canvas.winfo_rooty()
        # w = x + self.canvas.winfo_width()
        # h = y + self.canvas.winfo_height()
        # img = ImageGrab.grab(bbox=(x, y, w, h))
        
        return np.array(img.convert('L'))
    
    def capture_region(self, bbox: tuple) -> np.ndarray:
        """Capture specific region of canvas"""
        x1, y1, x2, y2 = bbox
        # Implementation for region capture
        pass
```

### 2.2 Image File Loading

```python
from PIL import Image
import numpy as np
from pathlib import Path

class ImageLoader:
    """Load images from file system"""
    
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
    
    def __init__(self):
        pass
    
    def load(self, path: str) -> np.ndarray:
        """
        Load image from file path
        
        Args:
            path: Path to image file
            
        Returns:
            image: Grayscale numpy array
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format not supported
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        # Load and convert to grayscale
        img = Image.open(path).convert('L')
        return np.array(img)
    
    def load_from_bytes(self, data: bytes) -> np.ndarray:
        """Load image from bytes (for drag & drop)"""
        img = Image.open(io.BytesIO(data)).convert('L')
        return np.array(img)
    
    def load_from_url(self, url: str) -> np.ndarray:
        """Load image from URL"""
        import requests
        response = requests.get(url)
        return self.load_from_bytes(response.content)
```

### 2.3 Webcam Capture

```python
import cv2
import numpy as np

class WebcamCapture:
    """Capture frames from webcam"""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
    
    def start(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
    
    def capture_frame(self) -> np.ndarray:
        """
        Capture single frame
        
        Returns:
            frame: Grayscale frame as numpy array
        """
        if self.cap is None:
            self.start()
        
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        
        # Convert BGR to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray
    
    def capture_roi(self, roi: tuple) -> np.ndarray:
        """Capture region of interest from frame"""
        frame = self.capture_frame()
        x, y, w, h = roi
        return frame[y:y+h, x:x+w]
    
    def stop(self):
        """Release camera"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
```

---

## 3. Grayscale Conversion

### 3.1 Color to Grayscale

```python
def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale using luminosity method
    
    Formula: Gray = 0.299*R + 0.587*G + 0.114*B
    
    Args:
        image: RGB image (H, W, 3)
        
    Returns:
        gray: Grayscale image (H, W)
    """
    if len(image.shape) == 2:
        # Already grayscale
        return image
    
    if image.shape[2] == 4:
        # RGBA - remove alpha channel
        image = image[:, :, :3]
    
    # Luminosity method (human perception weighted)
    weights = np.array([0.299, 0.587, 0.114])
    gray = np.dot(image, weights)
    
    return gray.astype(np.uint8)


def bgr_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV format) to grayscale"""
    if len(image.shape) == 2:
        return image
    
    weights = np.array([0.114, 0.587, 0.299])  # BGR order
    gray = np.dot(image, weights)
    
    return gray.astype(np.uint8)
```

### 3.2 Conversion Methods Comparison

| Method | Formula | Use Case |
|--------|---------|----------|
| Average | (R + G + B) / 3 | Simple, fast |
| Luminosity | 0.299R + 0.587G + 0.114B | Human perception |
| Desaturation | (max(R,G,B) + min(R,G,B)) / 2 | Preserve contrast |
| Max Decomposition | max(R, G, B) | Brightest channel |
| Min Decomposition | min(R, G, B) | Darkest channel |

---

## 4. Resizing

### 4.1 Resize to 28x28

```python
from PIL import Image
import numpy as np

def resize_image(image: np.ndarray, 
                 target_size: tuple = (28, 28),
                 method: str = 'lanczos') -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image (H, W)
        target_size: Target (height, width)
        method: Resampling method
        
    Returns:
        resized: Resized image
    """
    # Convert to PIL for high-quality resize
    pil_img = Image.fromarray(image)
    
    # Choose resampling method
    resample_methods = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS,
        'box': Image.BOX,
        'hamming': Image.HAMMING
    }
    
    resample = resample_methods.get(method, Image.LANCZOS)
    
    # Resize maintaining aspect ratio with padding
    resized = pil_img.resize(target_size, resample)
    
    return np.array(resized)


def resize_with_aspect_ratio(image: np.ndarray, 
                             target_size: tuple = (28, 28),
                             padding_value: int = 0) -> np.ndarray:
    """
    Resize image maintaining aspect ratio, pad remaining area
    
    Args:
        image: Input image
        target_size: Target (height, width)
        padding_value: Value for padding (0 = black)
        
    Returns:
        resized: Resized and padded image
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor
    scale = min(target_h / h, target_w / w)
    
    # New size maintaining aspect ratio
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize
    pil_img = Image.fromarray(image)
    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
    resized = np.array(resized)
    
    # Create padded output
    output = np.full(target_size, padding_value, dtype=np.uint8)
    
    # Calculate padding offsets (center the digit)
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    
    # Place resized image in center
    output[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    return output
```

### 4.2 Resampling Methods

| Method | Quality | Speed | Best For |
|--------|---------|-------|----------|
| Nearest | Low | Fastest | Pixel art, exact values |
| Bilinear | Medium | Fast | General use |
| Bicubic | High | Medium | Photographs |
| Lanczos | Highest | Slow | High quality downscaling |
| Box | Good | Fast | Downscaling integers |

**Recommendation**: Gunakan **Lanczos** untuk kualitas terbaik saat downscaling ke 28x28.

---

## 5. Digit Centering

### 5.1 Bounding Box Detection

```python
def find_bounding_box(image: np.ndarray, 
                      threshold: int = 10) -> tuple:
    """
    Find bounding box of non-zero pixels (digit region)
    
    Args:
        image: Grayscale image
        threshold: Minimum pixel value to consider as digit
        
    Returns:
        bbox: (x_min, y_min, x_max, y_max) or None if empty
    """
    # Find non-zero pixels
    rows = np.any(image > threshold, axis=1)
    cols = np.any(image > threshold, axis=0)
    
    if not rows.any() or not cols.any():
        return None
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return (x_min, y_min, x_max, y_max)


def crop_to_bounding_box(image: np.ndarray, 
                         padding: int = 2) -> np.ndarray:
    """
    Crop image to digit bounding box with padding
    
    Args:
        image: Input image
        padding: Pixels to add around digit
        
    Returns:
        cropped: Cropped image containing only the digit
    """
    bbox = find_bounding_box(image)
    
    if bbox is None:
        return image
    
    x_min, y_min, x_max, y_max = bbox
    
    # Add padding
    h, w = image.shape
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    return image[y_min:y_max, x_min:x_max]
```

### 5.2 Center of Mass Centering

```python
def compute_center_of_mass(image: np.ndarray) -> tuple:
    """
    Compute center of mass of the digit
    
    Args:
        image: Grayscale image
        
    Returns:
        (cx, cy): Center of mass coordinates
    """
    # Normalize image
    img = image.astype(float)
    total_mass = np.sum(img)
    
    if total_mass == 0:
        return (image.shape[1] // 2, image.shape[0] // 2)
    
    # Create coordinate grids
    y_indices, x_indices = np.indices(image.shape)
    
    # Weighted average of coordinates
    cx = np.sum(x_indices * img) / total_mass
    cy = np.sum(y_indices * img) / total_mass
    
    return (int(cx), int(cy))


def center_digit(image: np.ndarray, 
                 target_size: tuple = (28, 28)) -> np.ndarray:
    """
    Center digit in image based on center of mass
    
    MNIST digits are centered using center of mass to ensure
    consistency and improve recognition accuracy.
    
    Args:
        image: Input image with digit
        target_size: Output size
        
    Returns:
        centered: Image with digit centered
    """
    # Compute center of mass
    cx, cy = compute_center_of_mass(image)
    
    # Target center
    target_cx = target_size[1] // 2
    target_cy = target_size[0] // 2
    
    # Shift needed
    shift_x = target_cx - cx
    shift_y = target_cy - cy
    
    # Create output
    output = np.zeros(target_size, dtype=image.dtype)
    
    # Source and destination slices
    src_h, src_w = image.shape
    
    # Calculate valid regions
    src_x_start = max(0, -shift_x)
    src_y_start = max(0, -shift_y)
    src_x_end = min(src_w, target_size[1] - shift_x)
    src_y_end = min(src_h, target_size[0] - shift_y)
    
    dst_x_start = max(0, shift_x)
    dst_y_start = max(0, shift_y)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    
    # Copy centered region
    output[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        image[src_y_start:src_y_end, src_x_start:src_x_end]
    
    return output
```

### 5.3 MNIST Centering Convention

MNIST dataset uses specific centering:
1. Compute bounding box of digit
2. Resize digit to fit in 20x20 box (preserving aspect ratio)
3. Place 20x20 in center of 28x28 image
4. Center using center of mass

```python
def mnist_style_center(image: np.ndarray) -> np.ndarray:
    """
    Center digit using MNIST convention
    
    1. Crop to bounding box
    2. Resize to fit 20x20
    3. Center in 28x28 using center of mass
    """
    # Crop to digit
    cropped = crop_to_bounding_box(image, padding=0)
    
    # Resize to fit in 20x20
    resized = resize_with_aspect_ratio(cropped, (20, 20))
    
    # Create 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)
    
    # Compute center of mass of resized digit
    cx, cy = compute_center_of_mass(resized)
    
    # Place so center of mass is at image center (14, 14)
    offset_x = 14 - cx
    offset_y = 14 - cy
    
    # Adjust to keep within bounds
    offset_x = max(0, min(8, offset_x))
    offset_y = max(0, min(8, offset_y))
    
    # Place digit
    h, w = resized.shape
    canvas[offset_y:offset_y+h, offset_x:offset_x+w] = resized
    
    return canvas
```

---

## 6. Normalization

### 6.1 Pixel Value Normalization

```python
def normalize_0_1(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 1] range
    
    Args:
        image: Input image (0-255)
        
    Returns:
        normalized: Image with values in [0, 1]
    """
    return image.astype(np.float32) / 255.0


def normalize_minus1_1(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [-1, 1] range
    
    Useful for some network architectures
    """
    return (image.astype(np.float32) / 127.5) - 1.0


def standardize(image: np.ndarray, 
                mean: float = None, 
                std: float = None) -> np.ndarray:
    """
    Standardize image (zero mean, unit variance)
    
    Args:
        image: Input image
        mean: Dataset mean (computed if None)
        std: Dataset std (computed if None)
        
    Returns:
        standardized: (image - mean) / std
    """
    img = image.astype(np.float32)
    
    if mean is None:
        mean = np.mean(img)
    if std is None:
        std = np.std(img)
    
    # Avoid division by zero
    std = max(std, 1e-8)
    
    return (img - mean) / std


# MNIST specific values
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

def normalize_mnist(image: np.ndarray) -> np.ndarray:
    """Normalize using MNIST dataset statistics"""
    img = image.astype(np.float32) / 255.0
    return (img - MNIST_MEAN) / MNIST_STD
```

### 6.2 Color Inversion

```python
def invert_if_needed(image: np.ndarray, 
                     threshold: float = 0.5) -> np.ndarray:
    """
    Invert image colors if background is bright
    
    MNIST convention: dark background, light digit
    Canvas input: often light background, dark digit
    
    Args:
        image: Normalized image [0, 1]
        threshold: If mean > threshold, invert
        
    Returns:
        image: Possibly inverted image
    """
    if np.mean(image) > threshold:
        return 1.0 - image
    return image


def smart_invert(image: np.ndarray) -> np.ndarray:
    """
    Intelligently determine if inversion is needed
    
    Uses border pixels to detect background color
    """
    # Get border pixels
    h, w = image.shape
    border = np.concatenate([
        image[0, :],       # top
        image[-1, :],      # bottom
        image[:, 0],       # left
        image[:, -1]       # right
    ])
    
    border_mean = np.mean(border)
    
    # If border is bright, background is bright, need to invert
    if border_mean > 0.5:
        return 1.0 - image
    
    return image
```

---

## 7. Flatten to Vector

### 7.1 Flatten Operation

```python
def flatten(image: np.ndarray) -> np.ndarray:
    """
    Flatten 2D image to 1D vector
    
    Args:
        image: 2D image (28, 28)
        
    Returns:
        vector: 1D array (784,)
    """
    return image.flatten()


def flatten_batch(images: np.ndarray) -> np.ndarray:
    """
    Flatten batch of images
    
    Args:
        images: 3D array (N, 28, 28)
        
    Returns:
        vectors: 2D array (N, 784)
    """
    n_samples = images.shape[0]
    return images.reshape(n_samples, -1)


def unflatten(vector: np.ndarray, 
              shape: tuple = (28, 28)) -> np.ndarray:
    """
    Unflatten 1D vector back to 2D image
    
    Useful for visualization
    """
    return vector.reshape(shape)
```

---

## 8. Complete Pipeline

### 8.1 PreprocessingPipeline Class

```python
import numpy as np
from PIL import Image
from typing import Union, Optional
from dataclasses import dataclass

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    target_size: tuple = (28, 28)
    normalize_method: str = 'mnist'  # '0_1', '-1_1', 'mnist', 'standardize'
    center_method: str = 'mass'  # 'mass', 'bbox', 'none'
    auto_invert: bool = True
    padding: int = 2


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for digit images
    
    Usage:
        pipeline = PreprocessingPipeline()
        vector = pipeline.process(image)
    """
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
    
    def process(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """
        Process image through complete pipeline
        
        Args:
            image: Input image (array, PIL Image, or file path)
            
        Returns:
            vector: Processed 1D vector (784,) ready for network
        """
        # Step 1: Load/Convert to numpy array
        img = self._to_array(image)
        
        # Step 2: Convert to grayscale
        img = self._to_grayscale(img)
        
        # Step 3: Crop to digit region
        img = crop_to_bounding_box(img, padding=self.config.padding)
        
        # Step 4: Resize maintaining aspect ratio
        img = resize_with_aspect_ratio(img, self.config.target_size)
        
        # Step 5: Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Step 6: Invert if needed
        if self.config.auto_invert:
            img = smart_invert(img)
        
        # Step 7: Center digit
        if self.config.center_method == 'mass':
            img = center_digit((img * 255).astype(np.uint8), self.config.target_size)
            img = img.astype(np.float32) / 255.0
        
        # Step 8: Final normalization
        img = self._normalize(img)
        
        # Step 9: Flatten
        return img.flatten()
    
    def process_batch(self, images: list) -> np.ndarray:
        """Process multiple images"""
        return np.array([self.process(img) for img in images])
    
    def _to_array(self, image) -> np.ndarray:
        """Convert various input types to numpy array"""
        if isinstance(image, str):
            # File path
            return np.array(Image.open(image))
        elif isinstance(image, Image.Image):
            # PIL Image
            return np.array(image)
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
    
    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale if needed"""
        if len(image.shape) == 2:
            return image
        elif len(image.shape) == 3:
            return rgb_to_grayscale(image)
        else:
            raise ValueError(f"Invalid image shape: {image.shape}")
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply normalization based on config"""
        method = self.config.normalize_method
        
        if method == '0_1':
            return image  # Already in [0, 1]
        elif method == '-1_1':
            return image * 2 - 1
        elif method == 'mnist':
            return (image - MNIST_MEAN) / MNIST_STD
        elif method == 'standardize':
            return standardize(image)
        else:
            return image


# Convenience function
def preprocess(image, **kwargs) -> np.ndarray:
    """Quick preprocessing with default settings"""
    config = PreprocessingConfig(**kwargs)
    pipeline = PreprocessingPipeline(config)
    return pipeline.process(image)
```

### 8.2 Pipeline Usage Examples

```python
# Basic usage
pipeline = PreprocessingPipeline()

# From canvas capture
canvas_image = capture_canvas(canvas_widget)
vector = pipeline.process(canvas_image)
prediction = network.predict(vector)

# From file
vector = pipeline.process("digit.png")

# From webcam
with WebcamCapture() as cam:
    frame = cam.capture_frame()
    vector = pipeline.process(frame)

# Batch processing
images = ["1.png", "2.png", "3.png"]
vectors = pipeline.process_batch(images)
predictions = network.predict(vectors)

# Custom configuration
config = PreprocessingConfig(
    normalize_method='0_1',
    center_method='bbox',
    auto_invert=False
)
pipeline = PreprocessingPipeline(config)
```

---

## 9. Data Augmentation

### 9.1 Augmentation Techniques

```python
import numpy as np
from scipy import ndimage

def rotate_image(image: np.ndarray, 
                 angle: float,
                 fill_value: float = 0) -> np.ndarray:
    """
    Rotate image by given angle
    
    Args:
        image: Input image (28, 28)
        angle: Rotation angle in degrees
        fill_value: Value for empty pixels
    """
    return ndimage.rotate(image, angle, reshape=False, 
                         mode='constant', cval=fill_value)


def shift_image(image: np.ndarray,
                shift_x: int,
                shift_y: int,
                fill_value: float = 0) -> np.ndarray:
    """
    Shift image by given pixels
    """
    return ndimage.shift(image, [shift_y, shift_x], 
                        mode='constant', cval=fill_value)


def scale_image(image: np.ndarray,
                scale_factor: float) -> np.ndarray:
    """
    Scale image (zoom in/out)
    """
    return ndimage.zoom(image, scale_factor, order=1)


def add_noise(image: np.ndarray,
              noise_level: float = 0.1) -> np.ndarray:
    """
    Add random Gaussian noise
    """
    noise = np.random.randn(*image.shape) * noise_level
    noisy = image + noise
    return np.clip(noisy, 0, 1)


def elastic_transform(image: np.ndarray,
                      alpha: float = 20,
                      sigma: float = 3) -> np.ndarray:
    """
    Apply elastic deformation
    
    Popular augmentation for MNIST
    """
    random_state = np.random.RandomState(None)
    
    shape = image.shape
    
    # Random displacement fields
    dx = ndimage.gaussian_filter(
        random_state.rand(*shape) * 2 - 1, sigma) * alpha
    dy = ndimage.gaussian_filter(
        random_state.rand(*shape) * 2 - 1, sigma) * alpha
    
    # Create coordinate grid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    
    # Apply displacement
    indices = (y + dy).flatten(), (x + dx).flatten()
    
    return ndimage.map_coordinates(image, indices, order=1).reshape(shape)
```

### 9.2 Augmentation Pipeline

```python
class DataAugmentor:
    """
    Data augmentation for training
    
    Increases dataset diversity to improve generalization
    """
    
    def __init__(self, 
                 rotation_range: float = 10,
                 shift_range: int = 2,
                 scale_range: tuple = (0.9, 1.1),
                 noise_level: float = 0.05,
                 elastic_alpha: float = 15):
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.noise_level = noise_level
        self.elastic_alpha = elastic_alpha
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations"""
        img = image.copy()
        
        # Random rotation
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            img = rotate_image(img, angle)
        
        # Random shift
        if self.shift_range > 0:
            shift_x = np.random.randint(-self.shift_range, self.shift_range + 1)
            shift_y = np.random.randint(-self.shift_range, self.shift_range + 1)
            img = shift_image(img, shift_x, shift_y)
        
        # Random scale
        if self.scale_range != (1, 1):
            scale = np.random.uniform(*self.scale_range)
            img = scale_image(img, scale)
            # Resize back to original
            img = resize_image((img * 255).astype(np.uint8), (28, 28)) / 255.0
        
        # Add noise
        if self.noise_level > 0:
            img = add_noise(img, self.noise_level)
        
        return img
    
    def augment_batch(self, images: np.ndarray, 
                      augment_factor: int = 5) -> np.ndarray:
        """
        Augment batch of images
        
        Args:
            images: Original images (N, 28, 28)
            augment_factor: Number of augmented copies per image
            
        Returns:
            augmented: (N * (1 + augment_factor), 28, 28)
        """
        all_images = [images]  # Include originals
        
        for _ in range(augment_factor):
            augmented = np.array([self.augment(img) for img in images])
            all_images.append(augmented)
        
        return np.concatenate(all_images, axis=0)
```

---

## 10. Visualization Tools

### 10.1 Debug Visualization

```python
import matplotlib.pyplot as plt

def visualize_pipeline_steps(image: np.ndarray,
                            pipeline: PreprocessingPipeline):
    """
    Visualize each step of preprocessing pipeline
    
    Useful for debugging and understanding
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    steps = []
    
    # Step 1: Original
    steps.append(("Original", image))
    
    # Step 2: Grayscale
    gray = pipeline._to_grayscale(image)
    steps.append(("Grayscale", gray))
    
    # Step 3: Cropped
    cropped = crop_to_bounding_box(gray, padding=2)
    steps.append(("Cropped", cropped))
    
    # Step 4: Resized
    resized = resize_with_aspect_ratio(cropped, (28, 28))
    steps.append(("Resized 28x28", resized))
    
    # Step 5: Normalized
    normalized = resized.astype(np.float32) / 255.0
    steps.append(("Normalized [0,1]", normalized))
    
    # Step 6: Inverted
    inverted = smart_invert(normalized)
    steps.append(("Inverted", inverted))
    
    # Step 7: Centered
    centered = center_digit((inverted * 255).astype(np.uint8), (28, 28))
    steps.append(("Centered", centered))
    
    # Step 8: Final
    final = pipeline._normalize(centered.astype(np.float32) / 255.0)
    steps.append(("Final (MNIST norm)", final))
    
    # Plot
    for idx, (title, img) in enumerate(steps):
        ax = axes[idx // 4, idx % 4]
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def compare_augmentations(image: np.ndarray,
                         augmentor: DataAugmentor,
                         n_samples: int = 9):
    """Visualize augmentation results"""
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')
    
    for i in range(1, n_samples):
        ax = axes[i // 3, i % 3]
        augmented = augmentor.augment(image)
        ax.imshow(augmented, cmap='gray')
        ax.set_title(f"Augmented {i}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
```

---

## 11. Performance Considerations

### 11.1 Optimization Tips

| Operation | Tip | Speedup |
|-----------|-----|---------|
| Resize | Use PIL instead of scipy | 2-3x |
| Grayscale | Use NumPy dot product | 5x |
| Batch | Process in batches, not loops | 10x |
| Caching | Cache preprocessed training data | Memory vs speed |
| Float32 | Use float32 not float64 | 2x memory |

### 11.2 Memory Management

```python
def preprocess_mnist_dataset(X: np.ndarray,
                            chunk_size: int = 1000) -> np.ndarray:
    """
    Process large dataset in chunks to manage memory
    """
    n_samples = X.shape[0]
    output = np.zeros((n_samples, 784), dtype=np.float32)
    
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = X[start:end]
        
        # Process chunk
        processed = pipeline.process_batch(chunk)
        output[start:end] = processed
        
        # Optional: print progress
        print(f"Processed {end}/{n_samples}")
    
    return output
```

---

**Document Status**: ✅ Complete  
**Related Documents**:
- [NEURAL_NETWORK_DESIGN.md](NEURAL_NETWORK_DESIGN.md)
- [TRAINING_STRATEGY.md](TRAINING_STRATEGY.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
