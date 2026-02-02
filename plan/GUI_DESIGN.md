# 🎨 GUI Design - Digit Recognition

**Version**: 1.0  
**Date**: 1 Feb 2026  
**Status**: Planning

---

## 1. Overview

Dokumen ini menjelaskan desain lengkap antarmuka pengguna (GUI) untuk aplikasi Digit Recognition, termasuk layout, komponen, interaksi, dan styling.

### 1.1 Design Goals

| Goal            | Description                                          | Priority |
| --------------- | ---------------------------------------------------- | -------- |
| **Intuitive**   | User dapat langsung menggunakan tanpa membaca manual | P0       |
| **Responsive**  | UI tidak freeze saat processing                      | P0       |
| **Educational** | Menampilkan informasi yang membantu pemahaman        | P1       |
| **Modern**      | Tampilan clean dan professional                      | P1       |
| **Accessible**  | Support keyboard navigation                          | P2       |

### 1.2 Technology Stack

| Component     | Technology              | Rationale                  |
| ------------- | ----------------------- | -------------------------- |
| **Framework** | Tkinter + CustomTkinter | Cross-platform, built-in   |
| **Canvas**    | Tkinter Canvas          | Native drawing support     |
| **Charts**    | Matplotlib (embedded)   | Professional visualization |
| **Icons**     | Pillow (PIL)            | Image handling             |
| **Theme**     | Dark Mode (default)     | Modern, easy on eyes       |

---

## 2. Main Window Layout

### 2.1 Window Specifications

| Property         | Value                                 |
| ---------------- | ------------------------------------- |
| **Default Size** | 1200 × 800 pixels                     |
| **Minimum Size** | 1024 × 700 pixels                     |
| **Resizable**    | Yes (with constraints)                |
| **Title**        | "Digit Recognition - AI from Scratch" |

### 2.2 Layout Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TITLE BAR                                       │
│  🔢 Digit Recognition - AI from Scratch                          [─][□][×]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                              MENU BAR                                        │
│  [File] [Model] [View] [Tools] [Help]                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                              TOOLBAR                                         │
│  [📂 Load] [💾 Save] [🎯 Train] | [🎨 Theme] [⚙️ Settings]                   │
├───────────────────────────────────────┬─────────────────────────────────────┤
│                                       │                                      │
│           MAIN CONTENT AREA           │          SIDEBAR                     │
│                                       │                                      │
│  ┌─────────────────────────────────┐  │  ┌─────────────────────────────┐    │
│  │                                 │  │  │      PREDICTION RESULT      │    │
│  │                                 │  │  │                             │    │
│  │        DRAWING CANVAS           │  │  │     ┌─────────────────┐     │    │
│  │           (280×280)             │  │  │     │       7         │     │    │
│  │                                 │  │  │     │    (98.5%)      │     │    │
│  │                                 │  │  │     └─────────────────┘     │    │
│  │                                 │  │  │                             │    │
│  └─────────────────────────────────┘  │  ├─────────────────────────────┤    │
│                                       │  │    PROBABILITY BARS         │    │
│  ┌─────────────────────────────────┐  │  │                             │    │
│  │      CANVAS CONTROLS            │  │  │  0 ▓░░░░░░░░░░░░░ 1.2%     │    │
│  │  [🗑️ Clear] [↩️ Undo] [📤 Load] │  │  │  1 ▓░░░░░░░░░░░░░ 0.3%     │    │
│  │                                 │  │  │  2 ▓░░░░░░░░░░░░░ 0.1%     │    │
│  │  Brush: ○ ● ◉  Thickness: [━]  │  │  │  3 ▓░░░░░░░░░░░░░ 0.2%     │    │
│  └─────────────────────────────────┘  │  │  4 ▓░░░░░░░░░░░░░ 0.1%     │    │
│                                       │  │  5 ▓░░░░░░░░░░░░░ 0.1%     │    │
│  ┌─────────────────────────────────┐  │  │  6 ▓░░░░░░░░░░░░░ 0.1%     │    │
│  │        TAB PANEL                │  │  │  7 ████████████████ 98.5%  │    │
│  │  [Canvas] [Image] [Webcam]      │  │  │  8 ▓░░░░░░░░░░░░░ 0.2%     │    │
│  │  [Training] [Evaluation]        │  │  │  9 ▓░░░░░░░░░░░░░ 0.2%     │    │
│  └─────────────────────────────────┘  │  │                             │    │
│                                       │  └─────────────────────────────┘    │
│                                       │                                      │
│                                       │  ┌─────────────────────────────┐    │
│                                       │  │      HISTORY                │    │
│                                       │  │  [7] [2] [5] [1] [9] ...    │    │
│                                       │  └─────────────────────────────┘    │
│                                       │                                      │
├───────────────────────────────────────┴─────────────────────────────────────┤
│                              STATUS BAR                                      │
│  Model: Loaded ✓  |  Accuracy: 97.5%  |  Last prediction: 7  |  Ready      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Specifications

### 3.1 Drawing Canvas

#### Specifications

| Property                | Value            |
| ----------------------- | ---------------- |
| **Display Size**        | 280 × 280 pixels |
| **Internal Resolution** | 28 × 28 (scaled) |
| **Background**          | Black (#000000)  |
| **Stroke Color**        | White (#FFFFFF)  |
| **Default Brush Size**  | 15 pixels        |
| **Anti-aliasing**       | Enabled          |

#### Implementation

```python
import customtkinter as ctk
from PIL import Image, ImageDraw
import numpy as np

class DrawingCanvas(ctk.CTkFrame):
    """
    Interactive canvas for digit drawing

    Features:
    - Smooth drawing with mouse
    - Adjustable brush size
    - Undo functionality
    - Export to 28x28 array
    """

    def __init__(self, parent, size: int = 280,
                 on_draw_callback=None):
        super().__init__(parent)

        self.size = size
        self.brush_size = 15
        self.on_draw_callback = on_draw_callback

        # Create PIL image for high-quality drawing
        self.image = Image.new('L', (size, size), 0)
        self.draw = ImageDraw.Draw(self.image)

        # History for undo
        self.history = []
        self.max_history = 20

        # Previous point for smooth lines
        self.prev_x = None
        self.prev_y = None

        self._setup_ui()
        self._bind_events()

    def _setup_ui(self):
        """Setup canvas UI"""
        self.canvas = ctk.CTkCanvas(
            self,
            width=self.size,
            height=self.size,
            bg='black',
            highlightthickness=2,
            highlightbackground='#444444'
        )
        self.canvas.pack(padx=10, pady=10)

    def _bind_events(self):
        """Bind mouse events"""
        self.canvas.bind('<Button-1>', self._start_draw)
        self.canvas.bind('<B1-Motion>', self._draw)
        self.canvas.bind('<ButtonRelease-1>', self._end_draw)

    def _start_draw(self, event):
        """Start drawing - save state for undo"""
        self._save_state()
        self.prev_x = event.x
        self.prev_y = event.y
        self._draw_point(event.x, event.y)

    def _draw(self, event):
        """Draw line from previous to current point"""
        if self.prev_x and self.prev_y:
            # Draw on canvas
            self.canvas.create_line(
                self.prev_x, self.prev_y,
                event.x, event.y,
                fill='white',
                width=self.brush_size,
                capstyle='round',
                smooth=True
            )

            # Draw on PIL image
            self.draw.line(
                [self.prev_x, self.prev_y, event.x, event.y],
                fill=255,
                width=self.brush_size
            )

        self.prev_x = event.x
        self.prev_y = event.y

    def _end_draw(self, event):
        """End drawing - trigger prediction"""
        self.prev_x = None
        self.prev_y = None

        if self.on_draw_callback:
            self.on_draw_callback(self.get_image_array())

    def _draw_point(self, x, y):
        """Draw single point"""
        r = self.brush_size // 2
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            fill='white', outline='white'
        )
        self.draw.ellipse(
            [x - r, y - r, x + r, y + r],
            fill=255
        )

    def _save_state(self):
        """Save current state for undo"""
        self.history.append(self.image.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def clear(self):
        """Clear canvas"""
        self._save_state()
        self.canvas.delete('all')
        self.image = Image.new('L', (self.size, self.size), 0)
        self.draw = ImageDraw.Draw(self.image)

    def undo(self):
        """Undo last action"""
        if self.history:
            self.image = self.history.pop()
            self.draw = ImageDraw.Draw(self.image)
            self._refresh_canvas()

    def _refresh_canvas(self):
        """Refresh canvas from PIL image"""
        self.canvas.delete('all')
        # Convert PIL to PhotoImage and display
        from PIL import ImageTk
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)

    def get_image_array(self) -> np.ndarray:
        """
        Get canvas content as 28x28 normalized array

        Returns:
            numpy array of shape (784,) with values [0, 1]
        """
        # Resize to 28x28
        small = self.image.resize((28, 28), Image.LANCZOS)

        # Convert to numpy and normalize
        arr = np.array(small).astype(np.float32) / 255.0

        return arr.flatten()

    def set_brush_size(self, size: int):
        """Set brush size"""
        self.brush_size = max(5, min(30, size))

    def load_image(self, image: Image.Image):
        """Load image onto canvas"""
        self._save_state()

        # Resize to canvas size
        resized = image.resize((self.size, self.size), Image.LANCZOS)

        # Convert to grayscale
        if resized.mode != 'L':
            resized = resized.convert('L')

        self.image = resized
        self.draw = ImageDraw.Draw(self.image)
        self._refresh_canvas()
```

### 3.2 Probability Display

```python
class ProbabilityDisplay(ctk.CTkFrame):
    """
    Display prediction probabilities as horizontal bars

    Features:
    - Animated bar updates
    - Color-coded confidence levels
    - Highlight top prediction
    """

    COLORS = {
        'high': '#4CAF50',    # Green (>80%)
        'medium': '#FFC107',  # Yellow (50-80%)
        'low': '#9E9E9E',     # Gray (<50%)
        'background': '#2D2D2D'
    }

    def __init__(self, parent):
        super().__init__(parent)

        self.bars = []
        self.labels = []
        self._setup_ui()

    def _setup_ui(self):
        """Create probability bars for digits 0-9"""
        title = ctk.CTkLabel(
            self,
            text="Probabilities",
            font=('Helvetica', 14, 'bold')
        )
        title.pack(pady=(10, 5))

        for digit in range(10):
            frame = ctk.CTkFrame(self)
            frame.pack(fill='x', padx=10, pady=2)

            # Digit label
            label = ctk.CTkLabel(
                frame,
                text=str(digit),
                width=20,
                font=('Helvetica', 12, 'bold')
            )
            label.pack(side='left')

            # Progress bar
            bar = ctk.CTkProgressBar(
                frame,
                width=150,
                height=15,
                progress_color=self.COLORS['low']
            )
            bar.set(0)
            bar.pack(side='left', padx=5)

            # Percentage label
            pct_label = ctk.CTkLabel(
                frame,
                text="0.0%",
                width=50,
                font=('Helvetica', 10)
            )
            pct_label.pack(side='left')

            self.bars.append(bar)
            self.labels.append(pct_label)

    def update(self, probabilities: np.ndarray):
        """
        Update display with new probabilities

        Args:
            probabilities: Array of 10 probability values
        """
        max_idx = np.argmax(probabilities)

        for i, (bar, label) in enumerate(zip(self.bars, self.labels)):
            prob = probabilities[i]

            # Update bar
            bar.set(prob)

            # Update color based on value and if it's the prediction
            if i == max_idx:
                bar.configure(progress_color='#2196F3')  # Blue for prediction
            elif prob > 0.8:
                bar.configure(progress_color=self.COLORS['high'])
            elif prob > 0.5:
                bar.configure(progress_color=self.COLORS['medium'])
            else:
                bar.configure(progress_color=self.COLORS['low'])

            # Update label
            label.configure(text=f"{prob*100:.1f}%")

    def clear(self):
        """Reset all bars to zero"""
        for bar, label in zip(self.bars, self.labels):
            bar.set(0)
            bar.configure(progress_color=self.COLORS['low'])
            label.configure(text="0.0%")
```

### 3.3 Prediction Result Display

```python
class PredictionResult(ctk.CTkFrame):
    """
    Display main prediction result with large digit

    Features:
    - Large digit display
    - Confidence percentage
    - Animation on update
    """

    def __init__(self, parent):
        super().__init__(parent)

        self._setup_ui()

    def _setup_ui(self):
        """Setup result display"""
        title = ctk.CTkLabel(
            self,
            text="Prediction",
            font=('Helvetica', 14, 'bold')
        )
        title.pack(pady=(10, 5))

        # Large digit display
        self.digit_frame = ctk.CTkFrame(
            self,
            width=120,
            height=120,
            corner_radius=10
        )
        self.digit_frame.pack(pady=10)
        self.digit_frame.pack_propagate(False)

        self.digit_label = ctk.CTkLabel(
            self.digit_frame,
            text="-",
            font=('Helvetica', 72, 'bold'),
            text_color='#2196F3'
        )
        self.digit_label.place(relx=0.5, rely=0.5, anchor='center')

        # Confidence display
        self.confidence_label = ctk.CTkLabel(
            self,
            text="Confidence: --",
            font=('Helvetica', 12)
        )
        self.confidence_label.pack(pady=5)

    def update(self, digit: int, confidence: float):
        """
        Update display with prediction

        Args:
            digit: Predicted digit (0-9)
            confidence: Confidence level (0-1)
        """
        self.digit_label.configure(text=str(digit))
        self.confidence_label.configure(
            text=f"Confidence: {confidence*100:.1f}%"
        )

        # Color based on confidence
        if confidence > 0.9:
            color = '#4CAF50'  # Green
        elif confidence > 0.7:
            color = '#FFC107'  # Yellow
        else:
            color = '#F44336'  # Red

        self.digit_label.configure(text_color=color)

    def clear(self):
        """Clear display"""
        self.digit_label.configure(text="-", text_color='#888888')
        self.confidence_label.configure(text="Confidence: --")
```

---

## 4. Training Interface

### 4.1 Training Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING DASHBOARD                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────┐  ┌──────────────────────────────────┐  │
│  │      HYPERPARAMETERS            │  │        TRAINING PROGRESS         │  │
│  │                                 │  │                                  │  │
│  │  Learning Rate: [0.001    ▼]   │  │  Epoch: 12 / 20                  │  │
│  │                                 │  │  ████████████░░░░░░░░ 60%       │  │
│  │  Batch Size:    [32       ▼]   │  │                                  │  │
│  │                                 │  │  Current Loss: 0.0823            │  │
│  │  Epochs:        [20       ▼]   │  │  Current Acc:  96.42%            │  │
│  │                                 │  │                                  │  │
│  │  Hidden Layers: [128, 64  ▼]   │  │  Val Loss: 0.0912                │  │
│  │                                 │  │  Val Acc:  96.15%                │  │
│  │  ☑ Use Validation (10%)        │  │                                  │  │
│  │  ☑ Early Stopping (patience 5) │  │  Time Elapsed: 2m 34s            │  │
│  │  ☐ Data Augmentation           │  │  Time Remaining: ~1m 45s         │  │
│  │                                 │  │                                  │  │
│  └─────────────────────────────────┘  └──────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      TRAINING CHARTS                                   │  │
│  │                                                                        │  │
│  │   Loss                              Accuracy                           │  │
│  │   0.5│                              100%│                              │  │
│  │      │╲                                 │           ╭──────────────    │  │
│  │      │ ╲                                │       ╭───╯                  │  │
│  │      │  ╲                               │   ╭───╯                      │  │
│  │      │   ╲___                           │╭──╯                          │  │
│  │      │       ╲___                       ╯                              │  │
│  │   0.0└──────────────────             0%└────────────────────           │  │
│  │       0    5   10   15   20            0    5   10   15   20           │  │
│  │              Epoch                            Epoch                    │  │
│  │                                                                        │  │
│  │   ─── Train    ─ ─ Validation                                         │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │   [▶ Start Training]   [⏸ Pause]   [⏹ Stop]   [💾 Save Model]        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Training Panel Implementation

```python
class TrainingPanel(ctk.CTkFrame):
    """
    Training interface with controls and visualization

    Features:
    - Hyperparameter configuration
    - Real-time training progress
    - Live loss/accuracy charts
    - Training controls
    """

    def __init__(self, parent, trainer_callback):
        super().__init__(parent)

        self.trainer_callback = trainer_callback
        self.is_training = False
        self.is_paused = False

        self._setup_ui()

    def _setup_ui(self):
        """Setup training panel UI"""
        # Main container with two columns
        left_frame = ctk.CTkFrame(self)
        left_frame.pack(side='left', fill='both', padx=10, pady=10)

        right_frame = ctk.CTkFrame(self)
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)

        # Left: Hyperparameters
        self._setup_hyperparameters(left_frame)

        # Right: Progress and Charts
        self._setup_progress(right_frame)
        self._setup_charts(right_frame)

        # Bottom: Controls
        self._setup_controls()

    def _setup_hyperparameters(self, parent):
        """Setup hyperparameter controls"""
        title = ctk.CTkLabel(parent, text="Hyperparameters",
                            font=('Helvetica', 14, 'bold'))
        title.pack(pady=10)

        # Learning Rate
        lr_frame = ctk.CTkFrame(parent)
        lr_frame.pack(fill='x', padx=10, pady=5)
        ctk.CTkLabel(lr_frame, text="Learning Rate:").pack(side='left')
        self.lr_var = ctk.StringVar(value="0.001")
        self.lr_combo = ctk.CTkComboBox(
            lr_frame,
            values=["0.0001", "0.001", "0.01"],
            variable=self.lr_var,
            width=100
        )
        self.lr_combo.pack(side='right')

        # Batch Size
        batch_frame = ctk.CTkFrame(parent)
        batch_frame.pack(fill='x', padx=10, pady=5)
        ctk.CTkLabel(batch_frame, text="Batch Size:").pack(side='left')
        self.batch_var = ctk.StringVar(value="32")
        self.batch_combo = ctk.CTkComboBox(
            batch_frame,
            values=["16", "32", "64", "128"],
            variable=self.batch_var,
            width=100
        )
        self.batch_combo.pack(side='right')

        # Epochs
        epoch_frame = ctk.CTkFrame(parent)
        epoch_frame.pack(fill='x', padx=10, pady=5)
        ctk.CTkLabel(epoch_frame, text="Epochs:").pack(side='left')
        self.epoch_var = ctk.StringVar(value="20")
        self.epoch_combo = ctk.CTkComboBox(
            epoch_frame,
            values=["10", "20", "50", "100"],
            variable=self.epoch_var,
            width=100
        )
        self.epoch_combo.pack(side='right')

        # Checkboxes
        self.use_val_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            parent,
            text="Use Validation (10%)",
            variable=self.use_val_var
        ).pack(pady=5)

        self.early_stop_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            parent,
            text="Early Stopping",
            variable=self.early_stop_var
        ).pack(pady=5)

    def _setup_progress(self, parent):
        """Setup progress display"""
        progress_frame = ctk.CTkFrame(parent)
        progress_frame.pack(fill='x', padx=10, pady=10)

        ctk.CTkLabel(progress_frame, text="Training Progress",
                    font=('Helvetica', 14, 'bold')).pack(pady=5)

        # Epoch progress
        self.epoch_label = ctk.CTkLabel(progress_frame, text="Epoch: 0 / 0")
        self.epoch_label.pack()

        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=300)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=5)

        # Metrics
        metrics_frame = ctk.CTkFrame(progress_frame)
        metrics_frame.pack(fill='x', pady=10)

        self.loss_label = ctk.CTkLabel(metrics_frame, text="Loss: --")
        self.loss_label.pack(side='left', padx=10)

        self.acc_label = ctk.CTkLabel(metrics_frame, text="Accuracy: --")
        self.acc_label.pack(side='right', padx=10)

        # Time
        self.time_label = ctk.CTkLabel(progress_frame, text="Time: --")
        self.time_label.pack()

    def _setup_charts(self, parent):
        """Setup training charts with Matplotlib"""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        chart_frame = ctk.CTkFrame(parent)
        chart_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Create matplotlib figure
        self.fig, (self.ax_loss, self.ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.set_facecolor('#2D2D2D')

        for ax in [self.ax_loss, self.ax_acc]:
            ax.set_facecolor('#2D2D2D')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        self.ax_loss.set_title('Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_acc.set_title('Accuracy')
        self.ax_acc.set_xlabel('Epoch')

        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Initialize data
        self.train_loss_data = []
        self.val_loss_data = []
        self.train_acc_data = []
        self.val_acc_data = []

    def _setup_controls(self):
        """Setup control buttons"""
        control_frame = ctk.CTkFrame(self)
        control_frame.pack(fill='x', padx=10, pady=10)

        self.start_btn = ctk.CTkButton(
            control_frame,
            text="▶ Start Training",
            command=self._start_training
        )
        self.start_btn.pack(side='left', padx=5)

        self.pause_btn = ctk.CTkButton(
            control_frame,
            text="⏸ Pause",
            command=self._pause_training,
            state='disabled'
        )
        self.pause_btn.pack(side='left', padx=5)

        self.stop_btn = ctk.CTkButton(
            control_frame,
            text="⏹ Stop",
            command=self._stop_training,
            state='disabled'
        )
        self.stop_btn.pack(side='left', padx=5)

        self.save_btn = ctk.CTkButton(
            control_frame,
            text="💾 Save Model",
            command=self._save_model,
            state='disabled'
        )
        self.save_btn.pack(side='right', padx=5)

    def update_progress(self, epoch: int, total_epochs: int,
                       train_loss: float, train_acc: float,
                       val_loss: float = None, val_acc: float = None):
        """Update training progress display"""
        # Update labels
        self.epoch_label.configure(text=f"Epoch: {epoch} / {total_epochs}")
        self.progress_bar.set(epoch / total_epochs)
        self.loss_label.configure(text=f"Loss: {train_loss:.4f}")
        self.acc_label.configure(text=f"Accuracy: {train_acc*100:.2f}%")

        # Update chart data
        self.train_loss_data.append(train_loss)
        self.train_acc_data.append(train_acc)
        if val_loss is not None:
            self.val_loss_data.append(val_loss)
            self.val_acc_data.append(val_acc)

        self._update_charts()

    def _update_charts(self):
        """Refresh chart display"""
        epochs = range(1, len(self.train_loss_data) + 1)

        # Clear axes
        self.ax_loss.clear()
        self.ax_acc.clear()

        # Plot loss
        self.ax_loss.plot(epochs, self.train_loss_data, 'b-', label='Train')
        if self.val_loss_data:
            self.ax_loss.plot(epochs, self.val_loss_data, 'r--', label='Val')
        self.ax_loss.set_title('Loss', color='white')
        self.ax_loss.legend()

        # Plot accuracy
        self.ax_acc.plot(epochs, self.train_acc_data, 'b-', label='Train')
        if self.val_acc_data:
            self.ax_acc.plot(epochs, self.val_acc_data, 'r--', label='Val')
        self.ax_acc.set_title('Accuracy', color='white')
        self.ax_acc.legend()

        self.canvas.draw()

    def _start_training(self):
        """Start training"""
        self.is_training = True
        self.start_btn.configure(state='disabled')
        self.pause_btn.configure(state='normal')
        self.stop_btn.configure(state='normal')

        # Get hyperparameters
        config = {
            'learning_rate': float(self.lr_var.get()),
            'batch_size': int(self.batch_var.get()),
            'epochs': int(self.epoch_var.get()),
            'use_validation': self.use_val_var.get(),
            'early_stopping': self.early_stop_var.get()
        }

        # Clear previous data
        self.train_loss_data = []
        self.val_loss_data = []
        self.train_acc_data = []
        self.val_acc_data = []

        # Call training callback
        self.trainer_callback(config, self.update_progress)

    def _pause_training(self):
        """Pause/resume training"""
        self.is_paused = not self.is_paused
        self.pause_btn.configure(
            text="▶ Resume" if self.is_paused else "⏸ Pause"
        )

    def _stop_training(self):
        """Stop training"""
        self.is_training = False
        self.start_btn.configure(state='normal')
        self.pause_btn.configure(state='disabled')
        self.stop_btn.configure(state='disabled')
        self.save_btn.configure(state='normal')

    def _save_model(self):
        """Save trained model"""
        from tkinter import filedialog
        filepath = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("NumPy Archive", "*.npz")]
        )
        if filepath:
            # Save model callback
            pass
```

---

## 5. Color Scheme & Typography

### 5.1 Dark Theme (Default)

```python
DARK_THEME = {
    # Background colors
    'bg_primary': '#1E1E1E',      # Main background
    'bg_secondary': '#2D2D2D',    # Card background
    'bg_tertiary': '#383838',     # Input background

    # Text colors
    'text_primary': '#FFFFFF',    # Main text
    'text_secondary': '#B0B0B0',  # Secondary text
    'text_muted': '#707070',      # Muted text

    # Accent colors
    'accent_primary': '#2196F3',  # Primary accent (blue)
    'accent_success': '#4CAF50',  # Success (green)
    'accent_warning': '#FFC107',  # Warning (yellow)
    'accent_error': '#F44336',    # Error (red)

    # Border colors
    'border': '#444444',
    'border_focus': '#2196F3',

    # Canvas
    'canvas_bg': '#000000',
    'canvas_stroke': '#FFFFFF'
}
```

### 5.2 Light Theme (Alternative)

```python
LIGHT_THEME = {
    'bg_primary': '#FFFFFF',
    'bg_secondary': '#F5F5F5',
    'bg_tertiary': '#EEEEEE',

    'text_primary': '#212121',
    'text_secondary': '#757575',
    'text_muted': '#BDBDBD',

    'accent_primary': '#1976D2',
    'accent_success': '#388E3C',
    'accent_warning': '#F57C00',
    'accent_error': '#D32F2F',

    'border': '#E0E0E0',
    'border_focus': '#1976D2',

    'canvas_bg': '#FFFFFF',
    'canvas_stroke': '#000000'
}
```

### 5.3 Typography

| Element           | Font      | Size | Weight  |
| ----------------- | --------- | ---- | ------- |
| **Title**         | Helvetica | 24px | Bold    |
| **Heading**       | Helvetica | 18px | Bold    |
| **Subheading**    | Helvetica | 14px | Bold    |
| **Body**          | Helvetica | 12px | Regular |
| **Small**         | Helvetica | 10px | Regular |
| **Digit Display** | Helvetica | 72px | Bold    |
| **Button**        | Helvetica | 12px | Medium  |

---

## 6. Interaction Patterns

### 6.1 Drawing Interactions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DRAWING INTERACTION FLOW                                │
│                                                                              │
│   User Action          System Response              Visual Feedback          │
│   ───────────────────────────────────────────────────────────────────────   │
│                                                                              │
│   Mouse Down    ──►    Start stroke          ──►    Dot appears             │
│        │                Save undo state                                      │
│        │                                                                     │
│        ▼                                                                     │
│   Mouse Drag    ──►    Continue stroke       ──►    Line follows cursor     │
│        │                Update PIL image                                     │
│        │                                                                     │
│        ▼                                                                     │
│   Mouse Up      ──►    End stroke            ──►    Prediction triggered    │
│                        Get 28×28 array                                       │
│                        Run prediction                                        │
│                        Update display                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Keyboard Shortcuts

| Shortcut | Action             |
| -------- | ------------------ |
| `Ctrl+Z` | Undo last stroke   |
| `Ctrl+C` | Clear canvas       |
| `Ctrl+S` | Save model         |
| `Ctrl+O` | Open/load image    |
| `Space`  | Trigger prediction |
| `1-9,0`  | Quick test digit   |
| `F5`     | Start training     |
| `Esc`    | Stop training      |

### 6.3 Tooltips

```python
TOOLTIPS = {
    'canvas': "Draw a digit (0-9) here. Click and drag to draw.",
    'clear_btn': "Clear the canvas (Ctrl+C)",
    'undo_btn': "Undo last stroke (Ctrl+Z)",
    'load_btn': "Load an image file (Ctrl+O)",
    'brush_slider': "Adjust brush thickness",
    'train_btn': "Start training the neural network",
    'save_btn': "Save the trained model to file",
    'lr_combo': "Learning rate controls training speed",
    'batch_combo': "Number of samples per training step",
    'epoch_combo': "Number of complete passes through data"
}
```

---

## 7. Responsive Design

### 7.1 Window Resizing

```python
class ResponsiveLayout:
    """Handle responsive layout on window resize"""

    BREAKPOINTS = {
        'small': 1024,
        'medium': 1280,
        'large': 1440
    }

    def __init__(self, root):
        self.root = root
        self.root.bind('<Configure>', self._on_resize)

    def _on_resize(self, event):
        """Handle window resize"""
        width = event.width

        if width < self.BREAKPOINTS['small']:
            self._apply_small_layout()
        elif width < self.BREAKPOINTS['medium']:
            self._apply_medium_layout()
        else:
            self._apply_large_layout()

    def _apply_small_layout(self):
        """Compact layout for small windows"""
        # Stack sidebar below main content
        # Hide some non-essential elements
        pass

    def _apply_medium_layout(self):
        """Standard layout"""
        # Side-by-side layout
        pass

    def _apply_large_layout(self):
        """Expanded layout for large windows"""
        # Add extra panels
        pass
```

### 7.2 Minimum Size Constraints

```python
# Main window constraints
MIN_WIDTH = 1024
MIN_HEIGHT = 700
DEFAULT_WIDTH = 1200
DEFAULT_HEIGHT = 800

# Component constraints
CANVAS_SIZE = 280  # Fixed for consistent preprocessing
SIDEBAR_MIN_WIDTH = 250
SIDEBAR_MAX_WIDTH = 350
```

---

## 8. Accessibility

### 8.1 Keyboard Navigation

```python
class AccessibilityManager:
    """Manage keyboard navigation and accessibility"""

    def __init__(self, root):
        self.root = root
        self._setup_keyboard_nav()
        self._setup_focus_indicators()

    def _setup_keyboard_nav(self):
        """Setup Tab navigation order"""
        # Define tab order
        tab_order = [
            self.canvas,
            self.clear_btn,
            self.undo_btn,
            self.brush_slider,
            self.train_btn,
            # ...
        ]

        for i, widget in enumerate(tab_order):
            widget.lift()  # Ensure proper tab order

    def _setup_focus_indicators(self):
        """Visual focus indicators"""
        def on_focus_in(event):
            event.widget.configure(highlightthickness=2,
                                  highlightcolor='#2196F3')

        def on_focus_out(event):
            event.widget.configure(highlightthickness=1,
                                  highlightcolor='#444444')
```

### 8.2 Screen Reader Support

```python
# Accessible labels for screen readers
ARIA_LABELS = {
    'canvas': "Drawing canvas for digit input",
    'prediction': "Predicted digit is {digit} with {confidence}% confidence",
    'training_progress': "Training progress: epoch {epoch} of {total}, accuracy {acc}%",
    'probability_bar': "Probability of digit {digit}: {prob}%"
}
```

---

## 9. Main Application

### 9.1 Application Class

```python
import customtkinter as ctk

class DigitRecognitionApp(ctk.CTk):
    """
    Main application window for Digit Recognition
    """

    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Digit Recognition - AI from Scratch")
        self.geometry("1200x800")
        self.minsize(1024, 700)

        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Initialize components
        self.network = None
        self.preprocessor = None

        # Setup UI
        self._setup_menubar()
        self._setup_toolbar()
        self._setup_main_content()
        self._setup_statusbar()

        # Load default model
        self._load_default_model()

    def _setup_menubar(self):
        """Create menu bar"""
        menubar = ctk.CTkFrame(self, height=30)
        menubar.pack(fill='x')

        # File menu
        file_btn = ctk.CTkButton(menubar, text="File", width=60)
        file_btn.pack(side='left', padx=2)

        # Model menu
        model_btn = ctk.CTkButton(menubar, text="Model", width=60)
        model_btn.pack(side='left', padx=2)

        # View menu
        view_btn = ctk.CTkButton(menubar, text="View", width=60)
        view_btn.pack(side='left', padx=2)

        # Help menu
        help_btn = ctk.CTkButton(menubar, text="Help", width=60)
        help_btn.pack(side='left', padx=2)

    def _setup_toolbar(self):
        """Create toolbar"""
        toolbar = ctk.CTkFrame(self, height=40)
        toolbar.pack(fill='x', padx=5, pady=5)

        # Load button
        ctk.CTkButton(toolbar, text="📂 Load", width=80,
                     command=self._load_model).pack(side='left', padx=2)

        # Save button
        ctk.CTkButton(toolbar, text="💾 Save", width=80,
                     command=self._save_model).pack(side='left', padx=2)

        # Train button
        ctk.CTkButton(toolbar, text="🎯 Train", width=80,
                     command=self._show_training).pack(side='left', padx=2)

        # Separator
        ctk.CTkLabel(toolbar, text="|").pack(side='left', padx=10)

        # Theme toggle
        self.theme_var = ctk.StringVar(value="dark")
        ctk.CTkOptionMenu(toolbar, values=["Dark", "Light"],
                         variable=self.theme_var,
                         command=self._change_theme,
                         width=80).pack(side='right', padx=2)

    def _setup_main_content(self):
        """Setup main content area"""
        content = ctk.CTkFrame(self)
        content.pack(fill='both', expand=True, padx=5, pady=5)

        # Left side - Canvas and controls
        left_frame = ctk.CTkFrame(content)
        left_frame.pack(side='left', fill='both', expand=True, padx=5)

        # Drawing canvas
        self.canvas = DrawingCanvas(left_frame,
                                    on_draw_callback=self._on_draw)
        self.canvas.pack(pady=10)

        # Canvas controls
        controls = ctk.CTkFrame(left_frame)
        controls.pack(fill='x', padx=10)

        ctk.CTkButton(controls, text="🗑️ Clear",
                     command=self.canvas.clear).pack(side='left', padx=5)
        ctk.CTkButton(controls, text="↩️ Undo",
                     command=self.canvas.undo).pack(side='left', padx=5)

        # Brush size
        ctk.CTkLabel(controls, text="Brush:").pack(side='left', padx=10)
        self.brush_slider = ctk.CTkSlider(controls, from_=5, to=30,
                                         command=self._on_brush_change)
        self.brush_slider.set(15)
        self.brush_slider.pack(side='left')

        # Tab panel
        self.tab_view = ctk.CTkTabview(left_frame)
        self.tab_view.pack(fill='both', expand=True, pady=10)

        self.tab_view.add("Canvas")
        self.tab_view.add("Image")
        self.tab_view.add("Training")

        # Right side - Results
        right_frame = ctk.CTkFrame(content, width=300)
        right_frame.pack(side='right', fill='y', padx=5)
        right_frame.pack_propagate(False)

        # Prediction result
        self.prediction_display = PredictionResult(right_frame)
        self.prediction_display.pack(fill='x', pady=10)

        # Probability bars
        self.probability_display = ProbabilityDisplay(right_frame)
        self.probability_display.pack(fill='x', pady=10)

        # History
        history_label = ctk.CTkLabel(right_frame, text="History",
                                    font=('Helvetica', 14, 'bold'))
        history_label.pack(pady=10)

        self.history_frame = ctk.CTkScrollableFrame(right_frame, height=150)
        self.history_frame.pack(fill='x', padx=10)

    def _setup_statusbar(self):
        """Create status bar"""
        statusbar = ctk.CTkFrame(self, height=25)
        statusbar.pack(fill='x', side='bottom')

        self.status_model = ctk.CTkLabel(statusbar, text="Model: Not loaded")
        self.status_model.pack(side='left', padx=10)

        self.status_acc = ctk.CTkLabel(statusbar, text="Accuracy: --")
        self.status_acc.pack(side='left', padx=10)

        self.status_ready = ctk.CTkLabel(statusbar, text="Ready")
        self.status_ready.pack(side='right', padx=10)

    def _on_draw(self, image_array):
        """Handle drawing complete - run prediction"""
        if self.network:
            # Run prediction
            probabilities = self.network.forward(image_array.reshape(1, -1))
            prediction = np.argmax(probabilities)
            confidence = probabilities[0, prediction]

            # Update displays
            self.prediction_display.update(prediction, confidence)
            self.probability_display.update(probabilities[0])

            # Add to history
            self._add_to_history(prediction)

    def _on_brush_change(self, value):
        """Handle brush size change"""
        self.canvas.set_brush_size(int(value))

    def _change_theme(self, theme):
        """Change application theme"""
        ctk.set_appearance_mode(theme.lower())

    def _load_default_model(self):
        """Load pre-trained model"""
        try:
            self.network = create_network()
            ModelIO.load('models/pretrained.npz', self.network)
            self.status_model.configure(text="Model: Loaded ✓")
        except:
            self.status_model.configure(text="Model: Not found")

    def _load_model(self):
        """Load model from file"""
        from tkinter import filedialog
        filepath = filedialog.askopenfilename(
            filetypes=[("NumPy Archive", "*.npz")]
        )
        if filepath:
            self.network = create_network()
            ModelIO.load(filepath, self.network)
            self.status_model.configure(text=f"Model: {os.path.basename(filepath)}")

    def _save_model(self):
        """Save current model"""
        if self.network:
            from tkinter import filedialog
            filepath = filedialog.asksaveasfilename(
                defaultextension=".npz",
                filetypes=[("NumPy Archive", "*.npz")]
            )
            if filepath:
                ModelIO.save(self.network, filepath)

    def _show_training(self):
        """Show training panel"""
        self.tab_view.set("Training")

    def _add_to_history(self, digit):
        """Add prediction to history"""
        btn = ctk.CTkButton(
            self.history_frame,
            text=str(digit),
            width=40,
            height=40
        )
        btn.pack(side='left', padx=2, pady=2)


def main():
    """Application entry point"""
    app = DigitRecognitionApp()
    app.mainloop()


if __name__ == '__main__':
    main()
```

---

## 10. File Dialogs & Popups

### 10.1 Image Upload Dialog

```python
class ImageUploadDialog(ctk.CTkToplevel):
    """Dialog for uploading and preprocessing images"""

    def __init__(self, parent, callback):
        super().__init__(parent)

        self.callback = callback

        self.title("Upload Image")
        self.geometry("500x400")

        self._setup_ui()

    def _setup_ui(self):
        # Drop zone
        drop_zone = ctk.CTkFrame(self, width=400, height=200)
        drop_zone.pack(pady=20)

        ctk.CTkLabel(drop_zone,
                    text="📁 Drag & Drop Image Here\nor",
                    font=('Helvetica', 14)).pack(pady=30)

        ctk.CTkButton(drop_zone, text="Browse Files",
                     command=self._browse).pack()

        # Preview
        self.preview_label = ctk.CTkLabel(self, text="No image selected")
        self.preview_label.pack(pady=10)

        # Buttons
        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=20)

        ctk.CTkButton(btn_frame, text="Cancel",
                     command=self.destroy).pack(side='left', padx=10)

        self.predict_btn = ctk.CTkButton(btn_frame, text="Predict",
                                        command=self._predict,
                                        state='disabled')
        self.predict_btn.pack(side='left', padx=10)

    def _browse(self):
        filepath = ctk.filedialog.askopenfilename(
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        if filepath:
            self._load_image(filepath)

    def _load_image(self, filepath):
        self.image = Image.open(filepath)
        # Show preview
        # Enable predict button
        self.predict_btn.configure(state='normal')

    def _predict(self):
        if hasattr(self, 'image'):
            self.callback(self.image)
            self.destroy()
```

---

**Document Status**: ✅ Complete  
**Related Documents**:

- [USER_GUIDE.md](USER_GUIDE.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
