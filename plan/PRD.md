# Product Requirements Document (PRD) - Digit Recognition

**Version**: 1.0  
**Date**: 1 Feb 2026  
**Status**: Planning  
**Author**: Development Team

---

## 1. Executive Summary

### 1.1 Product Vision

Membangun sistem pengenalan angka tulisan tangan (digit 0-9) dengan implementasi neural network berbasis matematis murni menggunakan Python. Project ini berfokus pada pemahaman mendalam algoritma machine learning tanpa menggunakan high-level frameworks.

### 1.2 Problem Statement

Banyak programmer dan mahasiswa yang menggunakan ML frameworks seperti TensorFlow atau PyTorch tanpa memahami matematika fundamental di baliknya. Project ini menjembatani gap tersebut dengan:

- Implementasi neural network dari nol
- Dokumentasi matematika yang komprehensif
- Visualisasi proses pembelajaran
- Interface interaktif untuk eksperimen

### 1.3 Product Description

Aplikasi desktop berbasis Python dengan GUI interaktif yang memungkinkan user untuk:

- Menggambar angka pada canvas dan mendapatkan prediksi real-time
- Upload gambar angka untuk dikenali
- Melatih model dengan dataset MNIST
- Memvisualisasi proses training dan weights
- Memahami matematika melalui step-by-step mode

### 1.4 Target Release

**MVP Release**: Week 4 (akhir Februari 2026)  
**Full Release**: Week 6 (pertengahan Maret 2026)

---

## 2. Objectives & Goals

### 2.1 Primary Objectives

| ID  | Objective                             | Success Criteria                     |
| --- | ------------------------------------- | ------------------------------------ |
| O1  | Akurasi tinggi dalam pengenalan digit | ≥ 97% accuracy pada MNIST test set   |
| O2  | Implementasi matematis murni          | Tidak menggunakan TensorFlow/PyTorch |
| O3  | GUI intuitif dan responsif            | Response time < 100ms                |
| O4  | Educational value                     | Dokumentasi lengkap setiap komponen  |

### 2.2 Secondary Objectives

| ID  | Objective                 | Success Criteria                       |
| --- | ------------------------- | -------------------------------------- |
| O5  | Multi-input support       | Canvas, image, webcam                  |
| O6  | Model customization       | Adjustable hyperparameters             |
| O7  | Performance visualization | Real-time training charts              |
| O8  | Portability               | Cross-platform (Windows, macOS, Linux) |

### 2.3 Learning Goals

Project ini dirancang untuk memberikan pemahaman tentang:

1. **Linear Algebra** - Operasi matrix dalam neural network
2. **Calculus** - Derivative dan chain rule untuk backpropagation
3. **Probability** - Softmax dan cross-entropy loss
4. **Optimization** - Gradient descent dan variannya
5. **Deep Learning Theory** - Layer architecture dan activation functions

---

## 3. User Roles & Target Audience

### 3.1 Primary Users

#### 👨‍🎓 Student / Learner

- **Profile**: Mahasiswa CS/Data Science, self-taught programmer
- **Goals**: Memahami cara kerja neural network secara fundamental
- **Pain Points**: Kesulitan memahami matematika ML dari dokumentasi frameworks
- **Needs**: Step-by-step explanation, visualisasi, kode yang readable

#### 👨‍🔬 Researcher / Experimenter

- **Profile**: Researcher yang perlu custom implementation
- **Goals**: Memodifikasi dan eksperimen dengan arsitektur network
- **Pain Points**: Framework terlalu abstrak untuk customization
- **Needs**: Modular code, well-documented APIs

### 3.2 Secondary Users

#### 👨‍💻 Developer / Integrator

- **Profile**: Software developer yang ingin integrasi OCR sederhana
- **Goals**: Menggunakan digit recognition dalam aplikasi lain
- **Pain Points**: Setup framework besar untuk task sederhana
- **Needs**: Lightweight, easy-to-integrate library

#### 👨‍🏫 Educator / Teacher

- **Profile**: Dosen atau instructor ML/AI course
- **Goals**: Teaching tool untuk menjelaskan neural networks
- **Pain Points**: Sulit menunjukkan internal workings
- **Needs**: Visualization tools, step-by-step mode

---

## 4. Features & Requirements

### 4.1 Core Features (P0 - Must Have)

#### F1: Neural Network Engine

| Requirement | Description          | Acceptance Criteria                   |
| ----------- | -------------------- | ------------------------------------- |
| F1.1        | Forward Propagation  | Correct layer-by-layer computation    |
| F1.2        | Backward Propagation | Accurate gradient calculation         |
| F1.3        | Activation Functions | ReLU, Sigmoid, Softmax implemented    |
| F1.4        | Loss Functions       | Cross-entropy dengan correct gradient |
| F1.5        | Weight Update        | SGD optimizer working correctly       |
| F1.6        | Model Persistence    | Save/load weights to file             |

#### F2: Data Processing

| Requirement | Description         | Acceptance Criteria        |
| ----------- | ------------------- | -------------------------- |
| F2.1        | MNIST Loader        | Download dan parse dataset |
| F2.2        | Image Preprocessing | Resize to 28x28, normalize |
| F2.3        | Batch Processing    | Generate mini-batches      |
| F2.4        | Data Shuffling      | Random order each epoch    |

#### F3: Basic GUI

| Requirement | Description        | Acceptance Criteria        |
| ----------- | ------------------ | -------------------------- |
| F3.1        | Canvas Drawing     | 280x280 drawing area       |
| F3.2        | Prediction Display | Show predicted digit       |
| F3.3        | Probability Bars   | Visualize all 10 classes   |
| F3.4        | Clear/Undo         | Reset canvas functionality |

### 4.2 Important Features (P1 - Should Have)

#### F4: Training Interface

| Requirement | Description       | Acceptance Criteria               |
| ----------- | ----------------- | --------------------------------- |
| F4.1        | Training Controls | Start/stop/pause buttons          |
| F4.2        | Progress Display  | Epoch, loss, accuracy shown       |
| F4.3        | Live Charts       | Real-time loss/accuracy graph     |
| F4.4        | Hyperparameter UI | Sliders for learning rate, epochs |

#### F5: Advanced Input Methods

| Requirement | Description   | Acceptance Criteria    |
| ----------- | ------------- | ---------------------- |
| F5.1        | Image Upload  | Support PNG, JPG, BMP  |
| F5.2        | Image Preview | Show uploaded image    |
| F5.3        | Drag & Drop   | File drag-drop support |

#### F6: Results & Analytics

| Requirement | Description        | Acceptance Criteria         |
| ----------- | ------------------ | --------------------------- |
| F6.1        | Prediction History | List of recent predictions  |
| F6.2        | Confusion Matrix   | After training completion   |
| F6.3        | Metrics Display    | Precision, recall, F1-score |

### 4.3 Nice-to-Have Features (P2 - Could Have)

#### F7: Webcam Integration

| Requirement | Description    | Acceptance Criteria       |
| ----------- | -------------- | ------------------------- |
| F7.1        | Camera Capture | Capture digit from webcam |
| F7.2        | ROI Selection  | Select region of interest |
| F7.3        | Auto-detection | Detect digit boundaries   |

#### F8: Educational Features

| Requirement | Description          | Acceptance Criteria              |
| ----------- | -------------------- | -------------------------------- |
| F8.1        | Step-by-step Mode    | Visualize each forward pass step |
| F8.2        | Weight Visualization | Display learned filters          |
| F8.3        | Gradient Flow        | Show backprop visually           |
| F8.4        | Interactive Tutorial | Built-in learning modules        |

#### F9: Advanced Training

| Requirement | Description         | Acceptance Criteria      |
| ----------- | ------------------- | ------------------------ |
| F9.1        | Adam Optimizer      | Implement Adam algorithm |
| F9.2        | Learning Rate Decay | Scheduled learning rate  |
| F9.3        | Early Stopping      | Stop when no improvement |
| F9.4        | Data Augmentation   | Rotation, scaling, noise |

### 4.4 Future Features (P3 - Won't Have This Time)

| Feature                 | Description          | Rationale for Deferring     |
| ----------------------- | -------------------- | --------------------------- |
| CNN Implementation      | Convolutional layers | Requires more complexity    |
| GPU Acceleration        | CUDA/OpenCL support  | Out of scope for pure NumPy |
| Web Interface           | Browser-based GUI    | Different tech stack        |
| Mobile App              | iOS/Android version  | Different platform          |
| Multi-digit Recognition | Recognize sequences  | More complex preprocessing  |

---

## 5. Non-Functional Requirements

### 5.1 Performance Requirements

| Requirement        | Specification                           |
| ------------------ | --------------------------------------- |
| **Accuracy**       | ≥ 97% pada MNIST test set               |
| **Training Time**  | < 5 menit untuk full MNIST (60k images) |
| **Inference Time** | < 50ms per image                        |
| **GUI Response**   | < 100ms dari input ke prediction        |
| **Startup Time**   | < 3 detik application launch            |
| **Memory Usage**   | < 500MB peak selama training            |

### 5.2 Reliability Requirements

| Requirement         | Specification                       |
| ------------------- | ----------------------------------- |
| **Error Handling**  | Graceful handling semua exceptions  |
| **Data Validation** | Validate all user inputs            |
| **Crash Recovery**  | Save state before crash             |
| **Logging**         | Comprehensive logging for debugging |

### 5.3 Usability Requirements

| Requirement       | Specification                         |
| ----------------- | ------------------------------------- |
| **Intuitive UI**  | No manual needed for basic usage      |
| **Feedback**      | Visual feedback for all actions       |
| **Accessibility** | Keyboard navigation support           |
| **Documentation** | Tooltip dan help untuk semua features |

### 5.4 Compatibility Requirements

| Requirement           | Specification                         |
| --------------------- | ------------------------------------- |
| **Python Version**    | 3.10+                                 |
| **Operating Systems** | Windows 10+, macOS 11+, Ubuntu 20.04+ |
| **Screen Resolution** | Minimum 1280x720                      |
| **Dependencies**      | Minimal external packages             |

### 5.5 Maintainability Requirements

| Requirement       | Specification                           |
| ----------------- | --------------------------------------- |
| **Code Coverage** | ≥ 80% test coverage                     |
| **Documentation** | Docstrings untuk semua public functions |
| **Type Hints**    | 100% type annotation coverage           |
| **Code Style**    | PEP 8 compliant (black formatter)       |
| **Modularity**    | Loose coupling, high cohesion           |

---

## 6. Technical Constraints

### 6.1 Technology Constraints

- **No ML Frameworks**: Tidak menggunakan TensorFlow, PyTorch, Keras, etc.
- **Pure Python + NumPy**: Hanya menggunakan NumPy untuk matrix operations
- **Standard Library**: Minimize external dependencies
- **Cross-platform**: Harus berjalan di Windows, macOS, Linux

### 6.2 Design Constraints

- **Single Executable**: Dapat di-package sebagai single application
- **Offline Capable**: Tidak membutuhkan internet setelah setup
- **Portable**: Dapat di-run dari folder tanpa installation

### 6.3 Resource Constraints

- **Development Time**: 6 minggu
- **Team Size**: Solo developer / small team
- **Budget**: Open source, no paid services

---

## 7. Assumptions & Dependencies

### 7.1 Assumptions

1. User memiliki basic Python knowledge
2. User memiliki komputer dengan spesifikasi minimal
3. MNIST dataset tersedia untuk download
4. User familiar dengan basic ML concepts (optional)

### 7.2 Dependencies

| Dependency    | Version  | Purpose                 |
| ------------- | -------- | ----------------------- |
| Python        | ≥ 3.10   | Runtime environment     |
| NumPy         | ≥ 1.24   | Mathematical operations |
| Pillow        | ≥ 9.0    | Image processing        |
| Tkinter       | Built-in | GUI framework           |
| CustomTkinter | ≥ 5.0    | Modern UI components    |
| Matplotlib    | ≥ 3.7    | Visualization           |

### 7.3 External Dependencies

- MNIST Dataset (auto-download saat pertama run)
- Internet connection (hanya untuk initial setup)

---

## 8. Risks & Mitigation

| Risk                               | Impact | Likelihood | Mitigation                                  |
| ---------------------------------- | ------ | ---------- | ------------------------------------------- |
| Accuracy tidak mencapai target     | High   | Medium     | Tune hyperparameters, add regularization    |
| Training terlalu lambat            | Medium | Low        | Optimize NumPy operations, batch processing |
| GUI tidak responsive               | Medium | Medium     | Async processing, loading indicators        |
| Cross-platform issues              | Medium | Medium     | Test pada semua OS, use standard libraries  |
| Memory issues dengan large dataset | Low    | Low        | Implement data streaming, memory management |

---

## 9. Success Metrics & KPIs

### 9.1 Technical KPIs

| Metric         | Target       | Measurement     |
| -------------- | ------------ | --------------- |
| Model Accuracy | ≥ 97%        | MNIST test set  |
| Code Coverage  | ≥ 80%        | pytest-cov      |
| Documentation  | 100%         | All public APIs |
| Bug Count      | < 5 critical | Issue tracking  |

### 9.2 User Experience KPIs

| Metric                  | Target                 | Measurement                   |
| ----------------------- | ---------------------- | ----------------------------- |
| Prediction Success Rate | ≥ 95%                  | Real-world handwritten digits |
| User Satisfaction       | ≥ 4/5 stars            | User feedback                 |
| Learning Effectiveness  | Improved understanding | Survey                        |

### 9.3 Project KPIs

| Metric             | Target      | Measurement        |
| ------------------ | ----------- | ------------------ |
| On-time Delivery   | 100%        | Timeline adherence |
| Feature Completion | ≥ 90% P0/P1 | Feature checklist  |
| Code Quality       | A grade     | pylint score       |

---

## 10. Glossary

| Term                    | Definition                                       |
| ----------------------- | ------------------------------------------------ |
| **Neural Network**      | Sistem komputasi terinspirasi dari otak biologis |
| **Backpropagation**     | Algoritma untuk menghitung gradient              |
| **Gradient Descent**    | Metode optimisasi untuk minimize loss            |
| **MNIST**               | Dataset standar untuk digit recognition          |
| **Epoch**               | Satu iterasi lengkap melalui training data       |
| **Batch**               | Subset data yang diproses bersamaan              |
| **Learning Rate**       | Parameter yang mengontrol step size update       |
| **Activation Function** | Fungsi non-linear pada setiap neuron             |
| **Loss Function**       | Fungsi yang mengukur error prediksi              |
| **Forward Propagation** | Proses menghitung output dari input              |
| **Weight**              | Parameter yang dipelajari oleh network           |
| **Bias**                | Parameter tambahan pada setiap neuron            |

---

## 11. Appendices

### Appendix A: Related Documents

- [PLANNING_SUMMARY.md](PLANNING_SUMMARY.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [MATHEMATICAL_FOUNDATION.md](MATHEMATICAL_FOUNDATION.md)
- [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md)

### Appendix B: References

- MNIST Database: http://yann.lecun.com/exdb/mnist/
- Neural Networks and Deep Learning (Michael Nielsen)
- CS231n: Convolutional Neural Networks for Visual Recognition (Stanford)

### Appendix C: Revision History

| Version | Date       | Author   | Changes         |
| ------- | ---------- | -------- | --------------- |
| 1.0     | 1 Feb 2026 | Dev Team | Initial version |

---

**Document Status**: ✅ Complete  
**Next Review**: After Week 3 milestone
