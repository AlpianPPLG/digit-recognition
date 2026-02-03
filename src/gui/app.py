"""
Digit Recognition Application

Main application integrating GUI, model, and prediction logic.
"""

import customtkinter as ctk
import numpy as np
from tkinter import filedialog
from pathlib import Path
import sys
import os

# Add src to path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from .main_window import MainWindow
from .drawing_canvas import DrawingCanvas, DigitPreview


class DigitRecognitionApp(MainWindow):
    """
    Complete digit recognition application.
    
    Integrates:
    - Main window with sidebar and controls
    - Drawing canvas for digit input
    - Model loading and prediction
    - Results display with confidence
    """
    
    def __init__(self):
        super().__init__()
        
        # Add drawing canvas to main window
        self._setup_canvas()
        
        # Setup callbacks
        self._setup_callbacks()
        
        # Model state
        self.network = None
        self.preprocessor = None
    
    def _setup_canvas(self):
        """Setup drawing canvas in the main window."""
        # Drawing canvas
        self.drawing_canvas = DrawingCanvas(
            self.canvas_container,
            canvas_size=280,
            brush_size=18,
            bg_color="white",
            draw_color="black"
        )
        self.drawing_canvas.pack(expand=True, fill="both", padx=5, pady=5)
        
        # Preview widget (shows what model sees)
        self.preview = DigitPreview(
            self.canvas_frame,
            preview_size=84
        )
        self.preview.grid(row=3, column=0, pady=(0, 15))
        
        # Update preview on draw
        self.drawing_canvas.on_draw_callback = self._on_canvas_draw
    
    def _setup_callbacks(self):
        """Setup button callbacks."""
        self.on_load_model_callback = self._load_model
        self.on_predict_callback = self._predict
        self.on_clear_callback = self._clear_canvas
    
    def _on_canvas_draw(self):
        """Handle canvas drawing - update preview."""
        if not self.drawing_canvas.is_empty():
            img = self.drawing_canvas.get_normalized_image()
            self.preview.update_preview(img)
    
    def _load_model(self):
        """Load a trained model."""
        # Open file dialog
        filepath = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[
                ("NumPy Archive", "*.npz"),
                ("JSON Model", "*.json"),
                ("All files", "*.*")
            ],
            initialdir=Path(__file__).parent.parent.parent.parent / "models"
        )
        
        if not filepath:
            return
        
        try:
            # Import serialization module
            from core.serialization import load_model, ModelExporter
            from core.network import NetworkBuilder
            
            # Load model based on extension
            ext = Path(filepath).suffix.lower()
            
            if ext == '.npz':
                # Load full model
                self.network, _, metadata = load_model(filepath)
                model_name = Path(filepath).stem
            elif ext == '.json':
                # Create network and load JSON
                # Note: Need to know architecture beforehand
                self.network = (NetworkBuilder()
                               .input(784)
                               .dense(128, activation='relu')
                               .dense(64, activation='relu')
                               .dense(10, activation='softmax')
                               .build())
                ModelExporter.import_from_json(filepath, self.network)
                model_name = Path(filepath).stem
            else:
                self.show_error("Error", "Unsupported file format")
                return
            
            # Update status
            self.set_model_status(True, model_name)
            self.show_info("Success", f"Model '{model_name}' loaded successfully!")
            
        except Exception as e:
            self.show_error("Load Error", f"Failed to load model:\n{str(e)}")
            self.set_model_status(False)
    
    def _predict(self):
        """Run prediction on current drawing."""
        if self.network is None:
            self.show_error("Error", "Please load a model first!")
            return
        
        if self.drawing_canvas.is_empty():
            self.show_error("Error", "Please draw a digit first!")
            return
        
        try:
            # Get preprocessed image
            img = self.drawing_canvas.get_normalized_image()
            
            # Flatten for network input
            input_data = img.flatten().reshape(1, -1)
            
            # Run prediction
            output = self.network.forward(input_data)
            
            # Get probabilities
            probabilities = output[0]  # Shape: (10,)
            
            # Get predicted digit
            predicted_digit = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_digit] * 100)
            
            # Update display
            self.update_prediction(
                digit=predicted_digit,
                confidence=confidence,
                probabilities=probabilities.tolist()
            )
            
        except Exception as e:
            self.show_error("Prediction Error", f"Failed to predict:\n{str(e)}")
    
    def _clear_canvas(self):
        """Clear the drawing canvas."""
        self.drawing_canvas.clear()
        self.preview.clear_preview()


def create_default_model():
    """Create and save a default untrained model for testing."""
    from core.network import NetworkBuilder
    from core.serialization import save_model
    
    np.random.seed(42)
    
    network = (NetworkBuilder()
               .input(784)
               .dense(128, activation='relu')
               .dense(64, activation='relu')
               .dense(10, activation='softmax')
               .build())
    
    # Save to models folder
    models_dir = Path(__file__).parent.parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    filepath = models_dir / "untrained_model.npz"
    save_model(network, str(filepath), metadata={
        'trained': False,
        'architecture': '784-128-64-10'
    })
    
    print(f"Default model saved to: {filepath}")
    return filepath


def main():
    """Main entry point."""
    # Set appearance
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    # Create and run app
    app = DigitRecognitionApp()
    
    # Check for pre-trained model
    models_dir = Path(__file__).parent.parent.parent.parent / "models"
    default_model = models_dir / "digit_recognition_model.npz"
    
    if default_model.exists():
        try:
            from core.serialization import load_model
            app.network, _, _ = load_model(str(default_model))
            app.set_model_status(True, "digit_recognition_model")
        except:
            pass
    
    app.mainloop()


if __name__ == "__main__":
    main()
