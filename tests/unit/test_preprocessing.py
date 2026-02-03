"""
Unit Tests for Preprocessing Module

Tests for MNIST loading and image preprocessing.
"""

import pytest
import numpy as np
import sys
import tempfile
import os
from pathlib import Path
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from preprocessing.image_preprocessor import ImagePreprocessor, DataAugmenter


class TestImagePreprocessor:
    """Tests for ImagePreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return ImagePreprocessor()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample digit-like image."""
        # Create 100x100 white image with black digit
        img = np.full((100, 100), 255, dtype=np.uint8)
        # Draw a simple "1" shape
        img[20:80, 45:55] = 0  # Vertical line
        img[20:30, 35:55] = 0  # Top serif
        return img
    
    @pytest.fixture
    def sample_image_inverted(self):
        """Create sample with MNIST-style (white on black)."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[20:80, 45:55] = 255  # White digit on black
        return img
    
    def test_to_grayscale_rgb(self, preprocessor):
        """Test RGB to grayscale conversion."""
        rgb = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        gray = preprocessor.to_grayscale(rgb)
        
        assert gray.shape == (50, 50)
        assert gray.dtype == np.uint8
    
    def test_to_grayscale_rgba(self, preprocessor):
        """Test RGBA to grayscale conversion."""
        rgba = np.random.randint(0, 256, (50, 50, 4), dtype=np.uint8)
        gray = preprocessor.to_grayscale(rgba)
        
        assert gray.shape == (50, 50)
    
    def test_to_grayscale_already_gray(self, preprocessor):
        """Test grayscale passthrough."""
        gray_input = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        gray = preprocessor.to_grayscale(gray_input)
        
        np.testing.assert_array_equal(gray, gray_input)
    
    def test_resize(self, preprocessor, sample_image):
        """Test image resizing."""
        resized = preprocessor.resize(sample_image, (28, 28))
        
        assert resized.shape == (28, 28)
    
    def test_resize_with_aspect_ratio(self, preprocessor, sample_image):
        """Test resize with aspect ratio preservation."""
        resized = preprocessor.resize_with_aspect_ratio(sample_image, (28, 28))
        
        assert resized.shape == (28, 28)
    
    def test_find_bounding_box(self, preprocessor, sample_image_inverted):
        """Test bounding box detection."""
        preprocessor.invert = True
        bbox = preprocessor.find_bounding_box(sample_image_inverted)
        y_min, y_max, x_min, x_max = bbox
        
        # Content should be found
        assert y_min < y_max
        assert x_min < x_max
        # Should be around our drawn digit
        assert y_min <= 20
        assert y_max >= 80
    
    def test_crop_to_content(self, preprocessor, sample_image_inverted):
        """Test cropping to content."""
        preprocessor.invert = True
        cropped = preprocessor.crop_to_content(sample_image_inverted)
        
        # Should be smaller than original
        assert cropped.shape[0] < 100 or cropped.shape[1] < 100
    
    def test_center_of_mass(self, preprocessor):
        """Test center of mass calculation."""
        # Create image with content at known position
        img = np.zeros((28, 28))
        img[10:18, 10:18] = 1.0  # Square at top-left quadrant
        
        cy, cx = preprocessor.center_of_mass(img)
        
        # Center should be around (14, 14) for the square
        assert 10 < cy < 18
        assert 10 < cx < 18
    
    def test_center_image(self, preprocessor):
        """Test image centering."""
        # Create off-center image
        img = np.zeros((28, 28), dtype=np.uint8)
        img[0:10, 0:10] = 255  # Content in top-left
        
        centered = preprocessor.center_image(img)
        
        # Content should be more centered now
        cy_before, cx_before = preprocessor.center_of_mass(img)
        cy_after, cx_after = preprocessor.center_of_mass(centered)
        
        # After centering, should be closer to (14, 14)
        dist_before = abs(cy_before - 14) + abs(cx_before - 14)
        dist_after = abs(cy_after - 14) + abs(cx_after - 14)
        
        assert dist_after <= dist_before
    
    def test_normalize(self, preprocessor):
        """Test normalization."""
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        normalized = preprocessor.normalize(img)
        
        assert normalized.dtype == np.float32
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        np.testing.assert_almost_equal(normalized[0, 0], 0)
        np.testing.assert_almost_equal(normalized[0, 2], 1)
    
    def test_invert_colors(self, preprocessor):
        """Test color inversion."""
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        inverted = preprocessor.invert_colors(img)
        
        np.testing.assert_array_equal(inverted, [[255, 127, 0]])
    
    def test_flatten(self, preprocessor):
        """Test image flattening."""
        img = np.random.rand(28, 28)
        flattened = preprocessor.flatten(img)
        
        assert flattened.shape == (784,)
    
    def test_preprocess_output_shape(self, preprocessor, sample_image):
        """Test full preprocessing pipeline output shape."""
        processed = preprocessor.preprocess(sample_image)
        
        assert processed.shape == (784,)
        assert processed.dtype == np.float32
    
    def test_preprocess_output_range(self, preprocessor, sample_image):
        """Test preprocessing output value range."""
        processed = preprocessor.preprocess(sample_image)
        
        assert processed.min() >= 0
        assert processed.max() <= 1
    
    def test_preprocess_return_2d(self, preprocessor, sample_image):
        """Test preprocessing with 2D output."""
        processed = preprocessor.preprocess(sample_image, return_2d=True)
        
        assert processed.shape == (28, 28)
    
    def test_preprocess_batch(self, preprocessor, sample_image):
        """Test batch preprocessing."""
        images = [sample_image, sample_image, sample_image]
        batch = preprocessor.preprocess_batch(images)
        
        assert batch.shape == (3, 784)
    
    def test_load_image(self, preprocessor):
        """Test image loading from file."""
        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            filepath = f.name
        
        try:
            # Create and save test image
            img = Image.new('L', (50, 50), color=128)
            img.save(filepath)
            
            # Load with preprocessor
            loaded = preprocessor.load_image(filepath)
            
            assert loaded.shape == (50, 50)
        finally:
            os.remove(filepath)
    
    def test_load_image_not_found(self, preprocessor):
        """Test loading non-existent image."""
        with pytest.raises(FileNotFoundError):
            preprocessor.load_image("nonexistent.png")


class TestDataAugmenter:
    """Tests for DataAugmenter class."""
    
    @pytest.fixture
    def augmenter(self):
        """Create augmenter instance."""
        return DataAugmenter(
            rotation_range=15,
            shift_range=0.1,
            zoom_range=0.1,
            noise_factor=0.1
        )
    
    @pytest.fixture
    def sample_digit(self):
        """Create sample digit image."""
        img = np.zeros(784, dtype=np.float32)
        # Simple pattern
        img[300:400] = 1.0
        return img
    
    def test_rotate(self, augmenter, sample_digit):
        """Test rotation augmentation."""
        rotated = augmenter.rotate(sample_digit)
        
        assert rotated.shape == sample_digit.shape
        assert rotated.dtype == np.float32
    
    def test_rotate_preserves_range(self, augmenter, sample_digit):
        """Test rotation keeps values in valid range."""
        rotated = augmenter.rotate(sample_digit)
        
        assert rotated.min() >= 0
        assert rotated.max() <= 1
    
    def test_shift(self, augmenter, sample_digit):
        """Test shift augmentation."""
        shifted = augmenter.shift(sample_digit)
        
        assert shifted.shape == sample_digit.shape
    
    def test_shift_changes_image(self, augmenter, sample_digit):
        """Test that shift actually moves content."""
        np.random.seed(42)
        shifted = augmenter.shift(sample_digit, shift_x=3, shift_y=3)
        
        # Should be different from original
        assert not np.array_equal(shifted, sample_digit)
    
    def test_add_noise(self, augmenter, sample_digit):
        """Test noise augmentation."""
        np.random.seed(42)
        noisy = augmenter.add_noise(sample_digit)
        
        assert noisy.shape == sample_digit.shape
        # Should be different due to noise
        assert not np.array_equal(noisy, sample_digit)
    
    def test_add_noise_range(self, augmenter, sample_digit):
        """Test noise keeps values in valid range."""
        noisy = augmenter.add_noise(sample_digit)
        
        assert noisy.min() >= 0
        assert noisy.max() <= 1
    
    def test_augment(self, augmenter, sample_digit):
        """Test combined augmentation."""
        np.random.seed(42)
        augmented = augmenter.augment(sample_digit)
        
        assert augmented.shape == sample_digit.shape
    
    def test_augment_batch(self, augmenter):
        """Test batch augmentation."""
        np.random.seed(42)
        batch = np.random.rand(10, 784).astype(np.float32)
        augmented = augmenter.augment_batch(batch, augment_factor=2)
        
        # Should have 2x the samples
        assert augmented.shape == (20, 784)
    
    def test_augment_batch_includes_originals(self, augmenter):
        """Test that batch augmentation includes original images."""
        np.random.seed(42)
        batch = np.random.rand(5, 784).astype(np.float32)
        augmented = augmenter.augment_batch(batch, augment_factor=3)
        
        # First N should be originals
        np.testing.assert_array_equal(augmented[:5], batch)


class TestPreprocessingIntegration:
    """Integration tests for preprocessing pipeline."""
    
    def test_full_pipeline_realistic(self):
        """Test preprocessing with realistic input."""
        # Simulate canvas input: white background, black digit
        canvas = np.full((280, 280), 255, dtype=np.uint8)
        # Draw a simple digit-like shape
        canvas[50:230, 100:180] = 0
        
        preprocessor = ImagePreprocessor()
        result = preprocessor.preprocess(canvas)
        
        assert result.shape == (784,)
        assert result.min() >= 0
        assert result.max() <= 1
    
    def test_pipeline_with_rgb_input(self):
        """Test preprocessing with RGB input."""
        rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        preprocessor = ImagePreprocessor()
        result = preprocessor.preprocess(rgb_image)
        
        assert result.shape == (784,)
    
    def test_augmentation_pipeline(self):
        """Test preprocessing + augmentation pipeline."""
        np.random.seed(42)
        
        # Create sample data
        images = [np.random.randint(0, 256, (50, 50), dtype=np.uint8) for _ in range(5)]
        
        preprocessor = ImagePreprocessor()
        augmenter = DataAugmenter()
        
        # Preprocess
        processed = preprocessor.preprocess_batch(images)
        
        # Augment
        augmented = augmenter.augment_batch(processed, augment_factor=2)
        
        assert augmented.shape == (10, 784)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
