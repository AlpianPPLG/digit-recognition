"""
Drawing Canvas for Digit Input

A custom drawing canvas using tkinter Canvas widget for capturing
handwritten digit input.
"""

import tkinter as tk
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageDraw
from typing import Optional, Callable, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class DrawingCanvas(ctk.CTkFrame):
    """
    Custom drawing canvas for digit input.
    
    Features:
    - Mouse/touch drawing
    - Adjustable brush size
    - Clear functionality
    - Export to 28x28 numpy array for model input
    """
    
    def __init__(self, 
                 parent,
                 canvas_size: int = 280,
                 brush_size: int = 15,
                 bg_color: str = "white",
                 draw_color: str = "black",
                 **kwargs):
        """
        Initialize drawing canvas.
        
        Args:
            parent: Parent widget
            canvas_size: Size of canvas in pixels (square)
            brush_size: Drawing brush diameter
            bg_color: Canvas background color
            draw_color: Drawing color
        """
        super().__init__(parent, **kwargs)
        
        self.canvas_size = canvas_size
        self.brush_size = brush_size
        self.bg_color = bg_color
        self.draw_color = draw_color
        
        # Internal state
        self.last_x: Optional[int] = None
        self.last_y: Optional[int] = None
        self.is_drawing = False
        
        # Create PIL Image for drawing (higher res for quality)
        self.image = Image.new("L", (canvas_size, canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)
        
        # Callbacks
        self.on_draw_callback: Optional[Callable] = None
        
        # Create canvas
        self._create_canvas()
        self._bind_events()
    
    def _create_canvas(self):
        """Create the drawing canvas widget."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Canvas widget
        self.canvas = tk.Canvas(
            self,
            width=self.canvas_size,
            height=self.canvas_size,
            bg=self.bg_color,
            highlightthickness=2,
            highlightbackground="gray50"
        )
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        
        # Brush size slider
        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=1, column=0, pady=(0, 10))
        
        self.brush_label = ctk.CTkLabel(
            self.controls_frame,
            text="Brush Size:",
            font=ctk.CTkFont(size=11)
        )
        self.brush_label.grid(row=0, column=0, padx=(10, 5))
        
        self.brush_slider = ctk.CTkSlider(
            self.controls_frame,
            from_=5,
            to=30,
            number_of_steps=25,
            command=self._on_brush_change,
            width=120
        )
        self.brush_slider.set(self.brush_size)
        self.brush_slider.grid(row=0, column=1, padx=5)
        
        self.brush_value_label = ctk.CTkLabel(
            self.controls_frame,
            text=f"{self.brush_size}",
            width=30,
            font=ctk.CTkFont(size=11)
        )
        self.brush_value_label.grid(row=0, column=2, padx=(5, 10))
    
    def _bind_events(self):
        """Bind mouse events to canvas."""
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        
        # Touch support (for touchscreens)
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
    
    def _on_mouse_down(self, event):
        """Handle mouse button press."""
        self.is_drawing = True
        self.last_x = event.x
        self.last_y = event.y
        
        # Draw single point
        self._draw_point(event.x, event.y)
    
    def _on_mouse_move(self, event):
        """Handle mouse movement while pressed."""
        if not self.is_drawing:
            return
        
        if self.last_x is not None and self.last_y is not None:
            # Draw line from last point to current
            self._draw_line(self.last_x, self.last_y, event.x, event.y)
        
        self.last_x = event.x
        self.last_y = event.y
    
    def _on_mouse_up(self, event):
        """Handle mouse button release."""
        self.is_drawing = False
        self.last_x = None
        self.last_y = None
        
        # Trigger callback
        if self.on_draw_callback:
            self.on_draw_callback()
    
    def _on_brush_change(self, value):
        """Handle brush size change."""
        self.brush_size = int(value)
        self.brush_value_label.configure(text=f"{self.brush_size}")
    
    def _draw_point(self, x: int, y: int):
        """Draw a single point."""
        r = self.brush_size // 2
        
        # Draw on canvas
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            fill=self.draw_color, 
            outline=self.draw_color
        )
        
        # Draw on PIL image
        self.draw.ellipse(
            [x - r, y - r, x + r, y + r],
            fill=0
        )
    
    def _draw_line(self, x1: int, y1: int, x2: int, y2: int):
        """Draw a line between two points."""
        r = self.brush_size // 2
        
        # Draw on canvas
        self.canvas.create_line(
            x1, y1, x2, y2,
            fill=self.draw_color,
            width=self.brush_size,
            capstyle=tk.ROUND,
            smooth=True
        )
        # Add endpoint circles for smooth appearance
        self.canvas.create_oval(
            x2 - r, y2 - r, x2 + r, y2 + r,
            fill=self.draw_color,
            outline=self.draw_color
        )
        
        # Draw on PIL image
        self.draw.line(
            [x1, y1, x2, y2],
            fill=0,
            width=self.brush_size
        )
        self.draw.ellipse(
            [x2 - r, y2 - r, x2 + r, y2 + r],
            fill=0
        )
    
    def clear(self):
        """Clear the canvas."""
        # Clear tk canvas
        self.canvas.delete("all")
        
        # Reset PIL image
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)
    
    def get_image(self) -> np.ndarray:
        """
        Get canvas content as numpy array.
        
        Returns:
            Grayscale image as numpy array with shape (canvas_size, canvas_size)
        """
        return np.array(self.image)
    
    def get_normalized_image(self, target_size: int = 28) -> np.ndarray:
        """
        Get normalized image ready for model input.
        
        Performs:
        1. Resize to target_size x target_size
        2. Invert colors (white background -> black background)
        3. Normalize to [0, 1]
        4. Center the digit
        
        Args:
            target_size: Output image size (default 28 for MNIST)
            
        Returns:
            Normalized image array with shape (target_size, target_size)
        """
        # Get current image
        img = self.image.copy()
        
        # Find bounding box of drawn content
        img_array = np.array(img)
        
        # Check if canvas is empty
        if np.all(img_array == 255):
            return np.zeros((target_size, target_size), dtype=np.float32)
        
        # Find non-white pixels
        rows = np.any(img_array < 255, axis=1)
        cols = np.any(img_array < 255, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return np.zeros((target_size, target_size), dtype=np.float32)
        
        # Get bounding box
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]
        
        # Crop to content with padding
        padding = 20
        row_min = max(0, row_min - padding)
        row_max = min(img_array.shape[0] - 1, row_max + padding)
        col_min = max(0, col_min - padding)
        col_max = min(img_array.shape[1] - 1, col_max + padding)
        
        # Crop
        cropped = img_array[row_min:row_max+1, col_min:col_max+1]
        
        # Make square by padding shorter dimension
        h, w = cropped.shape
        max_dim = max(h, w)
        square = np.full((max_dim, max_dim), 255, dtype=np.uint8)
        
        y_offset = (max_dim - h) // 2
        x_offset = (max_dim - w) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
        
        # Resize to target size with some margin
        margin = 4
        inner_size = target_size - 2 * margin
        
        from PIL import Image as PILImage
        resized = PILImage.fromarray(square).resize(
            (inner_size, inner_size),
            PILImage.Resampling.LANCZOS
        )
        
        # Create final image with margin
        final = np.full((target_size, target_size), 255, dtype=np.uint8)
        final[margin:margin+inner_size, margin:margin+inner_size] = np.array(resized)
        
        # Invert (white background -> black, black drawing -> white)
        final = 255 - final
        
        # Normalize to [0, 1]
        normalized = final.astype(np.float32) / 255.0
        
        return normalized
    
    def get_flattened_image(self, target_size: int = 28) -> np.ndarray:
        """
        Get flattened image for model input.
        
        Args:
            target_size: Output image size
            
        Returns:
            Flattened array with shape (target_size * target_size,)
        """
        img = self.get_normalized_image(target_size)
        return img.flatten()
    
    def is_empty(self) -> bool:
        """Check if canvas is empty."""
        img_array = np.array(self.image)
        return np.all(img_array == 255)
    
    def set_brush_size(self, size: int):
        """Set brush size programmatically."""
        self.brush_size = size
        self.brush_slider.set(size)
        self.brush_value_label.configure(text=f"{size}")
    
    def set_colors(self, bg_color: str, draw_color: str):
        """Change canvas colors."""
        self.bg_color = bg_color
        self.draw_color = draw_color
        self.canvas.configure(bg=bg_color)


class DigitPreview(ctk.CTkFrame):
    """
    Preview widget showing the preprocessed digit image.
    
    Shows what the model actually "sees" after preprocessing.
    """
    
    def __init__(self, parent, preview_size: int = 112, **kwargs):
        """
        Initialize preview widget.
        
        Args:
            parent: Parent widget
            preview_size: Display size for preview
        """
        super().__init__(parent, **kwargs)
        
        self.preview_size = preview_size
        
        # Create label
        self.title_label = ctk.CTkLabel(
            self,
            text="Model Input Preview",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.title_label.grid(row=0, column=0, padx=10, pady=(10, 5))
        
        # Canvas for preview
        self.preview_canvas = tk.Canvas(
            self,
            width=preview_size,
            height=preview_size,
            bg="black",
            highlightthickness=1,
            highlightbackground="gray50"
        )
        self.preview_canvas.grid(row=1, column=0, padx=10, pady=(0, 10))
        
        # PIL image for display
        self._photo_image = None
    
    def update_preview(self, image_array: np.ndarray):
        """
        Update preview with preprocessed image.
        
        Args:
            image_array: 2D numpy array (28x28) normalized to [0, 1]
        """
        from PIL import ImageTk
        
        # Scale to preview size
        h, w = image_array.shape
        
        # Convert to 0-255 uint8
        img_uint8 = (image_array * 255).astype(np.uint8)
        
        # Create PIL image and resize
        img = Image.fromarray(img_uint8, mode='L')
        img_resized = img.resize(
            (self.preview_size, self.preview_size),
            Image.Resampling.NEAREST
        )
        
        # Convert to PhotoImage
        self._photo_image = ImageTk.PhotoImage(img_resized)
        
        # Update canvas
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(
            self.preview_size // 2,
            self.preview_size // 2,
            image=self._photo_image
        )
    
    def clear_preview(self):
        """Clear the preview."""
        self.preview_canvas.delete("all")
        self._photo_image = None


# Testing
if __name__ == "__main__":
    import customtkinter as ctk
    
    # Test window
    root = ctk.CTk()
    root.title("Drawing Canvas Test")
    root.geometry("600x500")
    
    # Drawing canvas
    canvas = DrawingCanvas(root, canvas_size=280)
    canvas.grid(row=0, column=0, padx=20, pady=20)
    
    # Preview
    preview = DigitPreview(root)
    preview.grid(row=0, column=1, padx=20, pady=20)
    
    def update_preview():
        img = canvas.get_normalized_image()
        preview.update_preview(img)
        print(f"Image shape: {img.shape}, range: [{img.min():.3f}, {img.max():.3f}]")
    
    canvas.on_draw_callback = update_preview
    
    # Clear button
    clear_btn = ctk.CTkButton(root, text="Clear", command=lambda: [canvas.clear(), preview.clear_preview()])
    clear_btn.grid(row=1, column=0, pady=10)
    
    # Print button
    def print_array():
        arr = canvas.get_flattened_image()
        print(f"Flattened shape: {arr.shape}")
        print(f"Non-zero: {np.count_nonzero(arr)}")
    
    print_btn = ctk.CTkButton(root, text="Print", command=print_array)
    print_btn.grid(row=1, column=1, pady=10)
    
    root.mainloop()
