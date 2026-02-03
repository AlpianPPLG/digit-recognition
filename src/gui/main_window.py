"""
Main Application Window for Digit Recognition

Using CustomTkinter for modern UI with light/dark mode support.
"""

import customtkinter as ctk
from typing import Optional, Callable
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Configure appearance
ctk.set_appearance_mode("dark")  # Options: "dark", "light", "system"
ctk.set_default_color_theme("blue")  # Options: "blue", "dark-blue", "green"


class MainWindow(ctk.CTk):
    """
    Main application window for digit recognition.
    
    Features:
    - Drawing canvas for digit input
    - Real-time prediction display
    - Model status and controls
    - Settings panel
    """
    
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title("AI Digit Recognition")
        self.geometry("1000x700")
        self.minsize(800, 600)
        
        # Configure grid layout
        self.grid_columnconfigure(0, weight=0)  # Sidebar - fixed
        self.grid_columnconfigure(1, weight=1)  # Main content - expandable
        self.grid_rowconfigure(0, weight=1)
        
        # Create main layout
        self._create_sidebar()
        self._create_main_content()
        
        # Model state
        self.model = None
        self.model_loaded = False
        
        # Callbacks
        self.on_predict_callback: Optional[Callable] = None
        self.on_clear_callback: Optional[Callable] = None
        self.on_load_model_callback: Optional[Callable] = None
    
    def _create_sidebar(self):
        """Create left sidebar with controls."""
        # Sidebar frame
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(7, weight=1)  # Spacer row
        
        # Logo/Title
        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="🔢 AI Digit",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.subtitle_label = ctk.CTkLabel(
            self.sidebar,
            text="Neural Network Recognition",
            font=ctk.CTkFont(size=12)
        )
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 20))
        
        # Model status
        self.model_status_frame = ctk.CTkFrame(self.sidebar)
        self.model_status_frame.grid(row=2, column=0, padx=15, pady=10, sticky="ew")
        
        self.model_status_label = ctk.CTkLabel(
            self.model_status_frame,
            text="Model Status:",
            font=ctk.CTkFont(size=12)
        )
        self.model_status_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.model_status_value = ctk.CTkLabel(
            self.model_status_frame,
            text="Not Loaded",
            text_color="orange",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.model_status_value.grid(row=0, column=1, padx=10, pady=5, sticky="e")
        
        # Load model button
        self.load_model_btn = ctk.CTkButton(
            self.sidebar,
            text="📂 Load Model",
            command=self._on_load_model_click
        )
        self.load_model_btn.grid(row=3, column=0, padx=15, pady=10, sticky="ew")
        
        # Predict button
        self.predict_btn = ctk.CTkButton(
            self.sidebar,
            text="🔮 Predict",
            command=self._on_predict_click,
            state="disabled"
        )
        self.predict_btn.grid(row=4, column=0, padx=15, pady=10, sticky="ew")
        
        # Clear button
        self.clear_btn = ctk.CTkButton(
            self.sidebar,
            text="🗑️ Clear Canvas",
            command=self._on_clear_click,
            fg_color="gray40",
            hover_color="gray30"
        )
        self.clear_btn.grid(row=5, column=0, padx=15, pady=10, sticky="ew")
        
        # Separator
        self.separator = ctk.CTkFrame(self.sidebar, height=2, fg_color="gray50")
        self.separator.grid(row=6, column=0, padx=15, pady=20, sticky="ew")
        
        # Spacer
        self.spacer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.spacer.grid(row=7, column=0, sticky="nsew")
        
        # Appearance mode toggle
        self.appearance_label = ctk.CTkLabel(
            self.sidebar,
            text="Appearance Mode:",
            font=ctk.CTkFont(size=12)
        )
        self.appearance_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        
        self.appearance_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=["Dark", "Light", "System"],
            command=self._change_appearance
        )
        self.appearance_menu.grid(row=9, column=0, padx=15, pady=(5, 15), sticky="ew")
        
        # Version info
        self.version_label = ctk.CTkLabel(
            self.sidebar,
            text="v1.0.0",
            text_color="gray50",
            font=ctk.CTkFont(size=10)
        )
        self.version_label.grid(row=10, column=0, padx=20, pady=(0, 10))
    
    def _create_main_content(self):
        """Create main content area with canvas and results."""
        # Main content frame
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # Left side - Drawing canvas
        self._create_canvas_section()
        
        # Right side - Results
        self._create_results_section()
    
    def _create_canvas_section(self):
        """Create drawing canvas section."""
        self.canvas_frame = ctk.CTkFrame(self.main_frame)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_rowconfigure(1, weight=1)
        
        # Canvas header
        self.canvas_header = ctk.CTkLabel(
            self.canvas_frame,
            text="✏️ Draw a Digit (0-9)",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.canvas_header.grid(row=0, column=0, padx=20, pady=(15, 10))
        
        # Canvas container (will hold actual drawing canvas)
        self.canvas_container = ctk.CTkFrame(
            self.canvas_frame,
            fg_color="white" if ctk.get_appearance_mode() == "Light" else "gray20"
        )
        self.canvas_container.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        
        # Instructions
        self.canvas_instructions = ctk.CTkLabel(
            self.canvas_frame,
            text="Draw a digit using your mouse or touch",
            text_color="gray60",
            font=ctk.CTkFont(size=11)
        )
        self.canvas_instructions.grid(row=2, column=0, padx=20, pady=(0, 15))
    
    def _create_results_section(self):
        """Create prediction results section."""
        self.results_frame = ctk.CTkFrame(self.main_frame)
        self.results_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)
        self.results_frame.grid_columnconfigure(0, weight=1)
        self.results_frame.grid_rowconfigure(2, weight=1)
        
        # Results header
        self.results_header = ctk.CTkLabel(
            self.results_frame,
            text="📊 Prediction Results",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.results_header.grid(row=0, column=0, padx=20, pady=(15, 10))
        
        # Main prediction display
        self.prediction_frame = ctk.CTkFrame(self.results_frame)
        self.prediction_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.prediction_frame.grid_columnconfigure(0, weight=1)
        
        self.prediction_label = ctk.CTkLabel(
            self.prediction_frame,
            text="Predicted Digit:",
            font=ctk.CTkFont(size=14)
        )
        self.prediction_label.grid(row=0, column=0, padx=15, pady=(15, 5))
        
        self.prediction_value = ctk.CTkLabel(
            self.prediction_frame,
            text="?",
            font=ctk.CTkFont(size=72, weight="bold"),
            text_color="#00a8ff"
        )
        self.prediction_value.grid(row=1, column=0, padx=15, pady=10)
        
        self.confidence_label = ctk.CTkLabel(
            self.prediction_frame,
            text="Confidence: --",
            font=ctk.CTkFont(size=14)
        )
        self.confidence_label.grid(row=2, column=0, padx=15, pady=(5, 15))
        
        # Probability distribution
        self.prob_frame = ctk.CTkFrame(self.results_frame)
        self.prob_frame.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="nsew")
        self.prob_frame.grid_columnconfigure(0, weight=1)
        
        self.prob_header = ctk.CTkLabel(
            self.prob_frame,
            text="All Probabilities",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.prob_header.grid(row=0, column=0, padx=15, pady=(15, 10))
        
        # Probability bars container
        self.prob_bars_frame = ctk.CTkScrollableFrame(
            self.prob_frame,
            label_text="",
            height=250
        )
        self.prob_bars_frame.grid(row=1, column=0, padx=10, pady=(0, 15), sticky="nsew")
        self.prob_bars_frame.grid_columnconfigure(1, weight=1)
        
        # Create probability bars for each digit
        self.prob_bars = []
        self.prob_labels = []
        for i in range(10):
            digit_label = ctk.CTkLabel(
                self.prob_bars_frame,
                text=f"{i}:",
                width=30,
                font=ctk.CTkFont(size=12)
            )
            digit_label.grid(row=i, column=0, padx=(10, 5), pady=3)
            
            progress = ctk.CTkProgressBar(self.prob_bars_frame)
            progress.grid(row=i, column=1, padx=5, pady=3, sticky="ew")
            progress.set(0)
            
            value_label = ctk.CTkLabel(
                self.prob_bars_frame,
                text="0.0%",
                width=50,
                font=ctk.CTkFont(size=11)
            )
            value_label.grid(row=i, column=2, padx=(5, 10), pady=3)
            
            self.prob_bars.append(progress)
            self.prob_labels.append(value_label)
    
    def _on_load_model_click(self):
        """Handle load model button click."""
        if self.on_load_model_callback:
            self.on_load_model_callback()
    
    def _on_predict_click(self):
        """Handle predict button click."""
        if self.on_predict_callback:
            self.on_predict_callback()
    
    def _on_clear_click(self):
        """Handle clear button click."""
        if self.on_clear_callback:
            self.on_clear_callback()
        self.reset_prediction_display()
    
    def _change_appearance(self, mode: str):
        """Change appearance mode."""
        ctk.set_appearance_mode(mode.lower())
        # Update canvas background
        if hasattr(self, 'canvas_container'):
            self.canvas_container.configure(
                fg_color="white" if mode == "Light" else "gray20"
            )
    
    # Public methods for external control
    def set_model_status(self, loaded: bool, name: str = ""):
        """Update model status display."""
        self.model_loaded = loaded
        if loaded:
            self.model_status_value.configure(
                text=f"Loaded" + (f" ({name})" if name else ""),
                text_color="green"
            )
            self.predict_btn.configure(state="normal")
        else:
            self.model_status_value.configure(
                text="Not Loaded",
                text_color="orange"
            )
            self.predict_btn.configure(state="disabled")
    
    def update_prediction(self, digit: int, confidence: float, probabilities: list):
        """
        Update prediction display.
        
        Args:
            digit: Predicted digit (0-9)
            confidence: Confidence percentage (0-100)
            probabilities: List of 10 probability values
        """
        # Update main prediction
        self.prediction_value.configure(text=str(digit))
        self.confidence_label.configure(text=f"Confidence: {confidence:.1f}%")
        
        # Highlight color based on confidence
        if confidence >= 90:
            color = "#00ff00"  # Green
        elif confidence >= 70:
            color = "#00a8ff"  # Blue
        elif confidence >= 50:
            color = "#ffaa00"  # Orange
        else:
            color = "#ff5555"  # Red
        
        self.prediction_value.configure(text_color=color)
        
        # Update probability bars
        for i, (prob, bar, label) in enumerate(zip(
            probabilities, self.prob_bars, self.prob_labels
        )):
            bar.set(prob)
            label.configure(text=f"{prob*100:.1f}%")
            
            # Highlight the predicted digit
            if i == digit:
                bar.configure(progress_color="#00a8ff")
            else:
                bar.configure(progress_color=["#1f538d", "#2980b9"])
    
    def reset_prediction_display(self):
        """Reset prediction display to initial state."""
        self.prediction_value.configure(text="?", text_color="#00a8ff")
        self.confidence_label.configure(text="Confidence: --")
        
        for bar, label in zip(self.prob_bars, self.prob_labels):
            bar.set(0)
            label.configure(text="0.0%")
    
    def show_error(self, title: str, message: str):
        """Show error dialog."""
        dialog = ctk.CTkToplevel(self)
        dialog.title(title)
        dialog.geometry("400x150")
        dialog.resizable(False, False)
        dialog.grab_set()
        
        # Center on parent
        dialog.transient(self)
        
        label = ctk.CTkLabel(
            dialog,
            text=message,
            wraplength=350,
            font=ctk.CTkFont(size=13)
        )
        label.pack(pady=30)
        
        btn = ctk.CTkButton(dialog, text="OK", command=dialog.destroy)
        btn.pack(pady=10)
    
    def show_info(self, title: str, message: str):
        """Show info dialog."""
        dialog = ctk.CTkToplevel(self)
        dialog.title(title)
        dialog.geometry("400x150")
        dialog.resizable(False, False)
        dialog.grab_set()
        dialog.transient(self)
        
        label = ctk.CTkLabel(
            dialog,
            text=message,
            wraplength=350,
            font=ctk.CTkFont(size=13)
        )
        label.pack(pady=30)
        
        btn = ctk.CTkButton(dialog, text="OK", command=dialog.destroy)
        btn.pack(pady=10)


# Entry point for testing
if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
