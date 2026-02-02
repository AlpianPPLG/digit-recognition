# 📖 User Guide - Digit Recognition

**Version**: 1.0
**Date**: 1 Feb 2026
**Status**: Planning

---

## 1. Introduction

### 1.1 Welcome

Selamat datang di **Digit Recognition**, aplikasi pengenalan angka tulisan tangan berbasis AI yang dibangun dari fondasi matematis murni. Panduan ini akan membantu Anda memahami dan menggunakan semua fitur aplikasi.

### 1.2 What You Can Do

- ✏️ **Gambar angka** di canvas dan dapatkan prediksi real-time
- 🖼️ **Upload gambar** angka untuk dikenali
- 📷 **Tangkap dari webcam** angka di dunia nyata
- 🎓 **Latih model** dengan dataset MNIST
- 📊 **Visualisasi** proses training dan hasil
- 💾 **Simpan & muat** model yang sudah dilatih

### 1.3 System Requirements

| Requirement | Minimum                              | Recommended    |
| ----------- | ------------------------------------ | -------------- |
| **OS**      | Windows 10 / macOS 11 / Ubuntu 20.04 | Latest version |
| **Python**  | 3.10                                 | 3.11+          |
| **RAM**     | 4 GB                                 | 8 GB           |
| **Storage** | 500 MB                               | 1 GB           |
| **Display** | 1280 × 720                           | 1920 × 1080    |

---

## 2. Getting Started

### 2.1 Installation

#### Step 1: Clone Repository

```bash
git clone https://github.com/username/digit-recognition.git
cd digit-recognition
```

#### Step 2: Create Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Run Application

```bash
python main.py
```

### 2.2 First Launch

Saat pertama kali menjalankan aplikasi:

1. **MNIST dataset** akan didownload otomatis (~15 MB)
2. **Pre-trained model** akan dimuat jika tersedia
3. Jika tidak ada model, Anda perlu **train model** terlebih dahulu

### 2.3 Interface Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ① Menu Bar - File, Model, View, Help options                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ② Toolbar - Quick actions: Load, Save, Train                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                       │                                      │
│  ③ Drawing Canvas                     │  ⑤ Prediction Result                │
│     Draw digits here                  │     Shows predicted digit            │
│                                       │                                      │
│                                       │  ⑥ Probability Bars                  │
│  ④ Canvas Controls                    │     Confidence for each digit        │
│     Clear, Undo, Brush size           │                                      │
│                                       │  ⑦ History                           │
│                                       │     Recent predictions               │
│                                       │                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  ⑧ Status Bar - Model status, accuracy, ready state                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Drawing Digits

### 3.1 Basic Drawing

1. **Position cursor** di atas canvas (area hitam)
2. **Klik dan tahan** mouse button
3. **Gerakkan mouse** untuk menggambar
4. **Lepaskan mouse** - prediksi akan muncul otomatis

### 3.2 Drawing Tips for Best Results

✅ **DO:**

- Gambar angka di **tengah canvas**
- Gunakan **stroke tebal** (brush size 15-20)
- Gambar dengan **ukuran besar** (hampir memenuhi canvas)
- Gambar dengan **gerakan smooth**

❌ **DON'T:**

- Gambar terlalu kecil
- Gambar di pinggir canvas
- Gambar dengan stroke terlalu tipis
- Membuat angka miring ekstrem

### 3.3 Canvas Controls

| Control          | Function                 | Shortcut |
| ---------------- | ------------------------ | -------- |
| **🗑️ Clear**     | Hapus semua gambar       | `Ctrl+C` |
| **↩️ Undo**      | Batalkan stroke terakhir | `Ctrl+Z` |
| **Brush Slider** | Ubah ketebalan brush     | -        |
| **📤 Load**      | Muat gambar dari file    | `Ctrl+O` |

### 3.4 Brush Size Guide

```
Brush Size Examples:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   Size 5        Size 10       Size 15       Size 20       Size 25           │
│     ·             ●             ●             ●             ●               │
│   (thin)       (medium)    (recommended)   (thick)      (very thick)        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Rekomendasi:** Size 15-20 untuk hasil terbaik

---

## 4. Understanding Results

### 4.1 Prediction Display

Setelah menggambar, hasil prediksi ditampilkan:

```
┌─────────────────────────────────────┐
│          PREDICTION                 │
│                                     │
│         ┌───────────┐               │
│         │     7     │  ← Predicted  │
│         │  (98.5%)  │    digit      │
│         └───────────┘               │
│                                     │
│    Confidence: 98.5%  ← Confidence  │
│                          level      │
└─────────────────────────────────────┘
```

### 4.2 Confidence Levels

| Confidence | Color     | Meaning                        |
| ---------- | --------- | ------------------------------ |
| **> 90%**  | 🟢 Green  | Very confident, likely correct |
| **70-90%** | 🟡 Yellow | Moderately confident           |
| **< 70%**  | 🔴 Red    | Low confidence, may be wrong   |

### 4.3 Probability Bars

Probability bars menunjukkan kemungkinan untuk setiap digit:

```
┌─────────────────────────────────────┐
│         PROBABILITIES               │
│                                     │
│  0 ▓░░░░░░░░░░░░░░░░░░░░  1.2%     │
│  1 ▓░░░░░░░░░░░░░░░░░░░░  0.3%     │
│  2 ▓░░░░░░░░░░░░░░░░░░░░  0.1%     │
│  3 ▓░░░░░░░░░░░░░░░░░░░░  0.2%     │
│  4 ▓░░░░░░░░░░░░░░░░░░░░  0.1%     │
│  5 ▓░░░░░░░░░░░░░░░░░░░░  0.1%     │
│  6 ▓░░░░░░░░░░░░░░░░░░░░  0.1%     │
│  7 ████████████████████  98.5%  ← │
│  8 ▓░░░░░░░░░░░░░░░░░░░░  0.2%     │
│  9 ▓░░░░░░░░░░░░░░░░░░░░  0.2%     │
│                                     │
└─────────────────────────────────────┘
```

### 4.4 When Predictions are Wrong

Jika prediksi salah:

1. **Coba gambar ulang** dengan lebih jelas
2. **Pastikan angka di tengah** canvas
3. **Gunakan brush lebih tebal**
4. **Gambar lebih besar**

Beberapa angka sering tertukar:

- **1** dan **7** - pastikan 7 punya garis horizontal
- **4** dan **9** - pastikan 4 punya sudut tajam
- **5** dan **6** - pastikan kurva jelas
- **3** dan **8** - pastikan celah terlihat

---

## 5. Uploading Images

### 5.1 Supported Formats

| Format | Extension       | Notes                               |
| ------ | --------------- | ----------------------------------- |
| PNG    | `.png`          | Best quality, supports transparency |
| JPEG   | `.jpg`, `.jpeg` | Good for photos                     |
| BMP    | `.bmp`          | Uncompressed                        |
| GIF    | `.gif`          | Single frame only                   |

### 5.2 How to Upload

**Method 1: File Dialog**

1. Klik tombol **📤 Load Image**
2. Pilih file gambar
3. Klik **Open**
4. Prediksi akan muncul otomatis

**Method 2: Drag & Drop**

1. Drag file gambar dari file explorer
2. Drop ke area canvas
3. Gambar akan dimuat dan diprediksi

### 5.3 Image Requirements

✅ **Ideal Image:**

- Single digit di tengah
- Background kontras (putih/hitam)
- Resolusi minimal 28×28 pixels
- Digit jelas dan mudah dibaca

❌ **Avoid:**

- Multiple digits dalam satu gambar
- Background ramai/kompleks
- Gambar blur atau noisy
- Digit terlalu kecil dalam frame

### 5.4 Preprocessing Applied

Gambar yang diupload akan diproses otomatis:

```
Original Image → Grayscale → Resize 28×28 → Normalize → Predict
```

1. **Grayscale** - Konversi ke hitam putih
2. **Resize** - Ubah ukuran ke 28×28 pixels
3. **Normalize** - Normalisasi nilai pixel ke 0-1
4. **Center** - Posisikan digit di tengah

---

## 6. Training the Model

### 6.1 When to Train

- Pertama kali menggunakan aplikasi
- Ingin meningkatkan akurasi
- Eksperimen dengan hyperparameters
- Membuat model custom

### 6.2 Accessing Training Panel

1. Klik tab **Training** di panel bawah
2. Atau klik tombol **🎯 Train** di toolbar

### 6.3 Training Options

```
┌─────────────────────────────────────┐
│         HYPERPARAMETERS             │
├─────────────────────────────────────┤
│                                     │
│  Learning Rate: [0.001    ▼]        │
│  → Kecepatan pembelajaran           │
│  → Lebih kecil = lebih stabil       │
│  → Lebih besar = lebih cepat        │
│                                     │
│  Batch Size:    [32       ▼]        │
│  → Jumlah sampel per step           │
│  → 32 adalah nilai umum             │
│                                     │
│  Epochs:        [20       ▼]        │
│  → Berapa kali lewati semua data    │
│  → Lebih banyak = lebih akurat      │
│                                     │
│  ☑ Use Validation                   │
│  → Evaluasi di data terpisah        │
│                                     │
│  ☑ Early Stopping                   │
│  → Berhenti jika tidak membaik      │
│                                     │
└─────────────────────────────────────┘
```

### 6.4 Starting Training

1. **Konfigurasi hyperparameters** sesuai kebutuhan
2. Klik tombol **▶ Start Training**
3. **Tunggu** hingga training selesai
4. Progress akan ditampilkan real-time

### 6.5 Understanding Training Progress

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PROGRESS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Epoch: 12 / 20        ← Current epoch / total                              │
│  ████████████░░░░░░░░  60%  ← Progress bar                                  │
│                                                                              │
│  Current Loss: 0.0823  ← Error (semakin kecil semakin baik)                 │
│  Current Acc:  96.42%  ← Akurasi (semakin tinggi semakin baik)              │
│                                                                              │
│  Val Loss: 0.0912      ← Validation loss                                    │
│  Val Acc:  96.15%      ← Validation accuracy                                │
│                                                                              │
│  Time Elapsed: 2m 34s  ← Waktu yang sudah berjalan                          │
│  Time Remaining: ~1m   ← Estimasi waktu tersisa                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.6 Training Controls

| Button      | Function                                 |
| ----------- | ---------------------------------------- |
| **▶ Start** | Mulai training                           |
| **⏸ Pause** | Pause training (lanjutkan dengan Resume) |
| **⏹ Stop**  | Hentikan training                        |
| **💾 Save** | Simpan model yang sudah dilatih          |

### 6.7 Expected Training Results

| Epochs | Expected Accuracy | Time (approx) |
| ------ | ----------------- | ------------- |
| 5      | ~90-93%           | ~1 min        |
| 10     | ~95-96%           | ~2 min        |
| 20     | ~97-98%           | ~4 min        |
| 50     | ~98%+             | ~10 min       |

### 6.8 Saving Trained Model

Setelah training selesai:

1. Klik tombol **💾 Save Model**
2. Pilih lokasi dan nama file
3. Klik **Save**
4. Model akan disimpan dalam format `.npz`

---

## 7. Loading & Saving Models

### 7.1 Model Files

Model disimpan dalam format **NumPy Archive (.npz)** yang berisi:

- Network weights (bobot)
- Network architecture
- Training history (optional)

### 7.2 Loading a Model

1. Klik **File** → **Load Model** (atau `Ctrl+O`)
2. Pilih file `.npz`
3. Klik **Open**
4. Status bar akan menunjukkan model loaded

### 7.3 Saving a Model

1. Pastikan model sudah di-train
2. Klik **File** → **Save Model** (atau `Ctrl+S`)
3. Pilih lokasi dan nama
4. Klik **Save**

### 7.4 Pre-trained Models

Beberapa pre-trained models tersedia:

| Model             | Accuracy | Size    | Description              |
| ----------------- | -------- | ------- | ------------------------ |
| `default.npz`     | 97.5%    | ~1 MB   | Model standar, balance   |
| `lightweight.npz` | 95%      | ~200 KB | Untuk perangkat terbatas |
| `accurate.npz`    | 98.2%    | ~2 MB   | Akurasi tinggi           |

---

## 8. Keyboard Shortcuts

### 8.1 General Shortcuts

| Shortcut | Action                   |
| -------- | ------------------------ |
| `Ctrl+O` | Open/Load model or image |
| `Ctrl+S` | Save model               |
| `Ctrl+Z` | Undo last stroke         |
| `Ctrl+C` | Clear canvas             |
| `F5`     | Start training           |
| `Esc`    | Stop training            |
| `F1`     | Open help                |

### 8.2 Canvas Shortcuts

| Shortcut  | Action                       |
| --------- | ---------------------------- |
| `Space`   | Trigger prediction           |
| `Delete`  | Clear canvas                 |
| `+` / `-` | Increase/decrease brush size |

### 8.3 Navigation

| Shortcut    | Action           |
| ----------- | ---------------- |
| `Tab`       | Next control     |
| `Shift+Tab` | Previous control |
| `Enter`     | Activate button  |

---

## 9. Troubleshooting

### 9.1 Common Issues

#### ❌ "Model not loaded"

**Problem:** Aplikasi tidak bisa memuat model

**Solutions:**

1. Pastikan file model ada di folder `models/`
2. Coba train model baru
3. Download pre-trained model dari repository

#### ❌ Wrong predictions

**Problem:** Prediksi sering salah

**Solutions:**

1. Gambar lebih jelas dan besar
2. Posisikan digit di tengah
3. Train model lebih lama
4. Coba re-train dengan data augmentation

#### ❌ Slow performance

**Problem:** Aplikasi lambat

**Solutions:**

1. Kurangi batch size saat training
2. Tutup aplikasi lain
3. Gunakan model yang lebih kecil

#### ❌ Training stuck

**Problem:** Training tidak progres

**Solutions:**

1. Coba learning rate lebih kecil
2. Re-initialize weights (train ulang)
3. Periksa apakah data loaded dengan benar

### 9.2 Error Messages

| Error                   | Meaning                | Solution                            |
| ----------------------- | ---------------------- | ----------------------------------- |
| "MNIST download failed" | Gagal download dataset | Check internet connection           |
| "Out of memory"         | RAM tidak cukup        | Kurangi batch size                  |
| "Invalid model file"    | File model corrupt     | Download ulang atau train baru      |
| "No GPU available"      | GPU tidak terdeteksi   | Normal, aplikasi tetap jalan di CPU |

### 9.3 Getting Help

Jika masih ada masalah:

1. **Check documentation** di folder `docs/`
2. **Open an issue** di GitHub repository
3. **Join Discord** community (link di README)

---

## 10. Tips & Best Practices

### 10.1 For Best Accuracy

1. **Train dengan epochs cukup** (minimal 10-20)
2. **Gunakan validation** untuk monitor overfitting
3. **Gambar dengan konsisten** - ukuran dan posisi sama
4. **Gunakan brush size optimal** (15-20)

### 10.2 For Faster Training

1. **Gunakan batch size lebih besar** (64-128) jika RAM cukup
2. **Matikan data augmentation** untuk speed
3. **Gunakan early stopping** untuk efisiensi

### 10.3 For Learning

1. **Baca kode sumber** untuk memahami implementasi
2. **Eksperimen dengan hyperparameters**
3. **Coba arsitektur network berbeda**
4. **Lihat visualisasi weights** untuk insight

---

## 11. Glossary

| Term              | Definition                                             |
| ----------------- | ------------------------------------------------------ |
| **Accuracy**      | Persentase prediksi yang benar                         |
| **Batch**         | Kelompok sampel yang diproses bersamaan                |
| **Canvas**        | Area untuk menggambar digit                            |
| **Confidence**    | Tingkat keyakinan prediksi                             |
| **Epoch**         | Satu kali lewat seluruh dataset                        |
| **Learning Rate** | Kecepatan model belajar                                |
| **Loss**          | Ukuran error prediksi                                  |
| **Model**         | Neural network yang sudah dilatih                      |
| **MNIST**         | Dataset standar untuk digit recognition                |
| **Prediction**    | Hasil tebakan model                                    |
| **Training**      | Proses melatih model dengan data                       |
| **Validation**    | Evaluasi model pada data yang tidak digunakan training |
| **Weights**       | Parameter yang dipelajari oleh network                 |

---

## 12. FAQ

### Q: Berapa lama training biasanya?

**A:** Dengan default settings (20 epochs), sekitar 3-5 menit pada laptop standar.

### Q: Akurasi berapa yang bisa dicapai?

**A:** Target 97%+. Dengan tuning yang baik bisa mencapai 98%+.

### Q: Bisakah digunakan untuk huruf?

**A:** Saat ini hanya untuk digit (0-9). Untuk huruf perlu training dengan dataset berbeda (seperti EMNIST).

### Q: Apakah perlu GPU?

**A:** Tidak. Aplikasi ini menggunakan NumPy dan berjalan efisien di CPU. GPU tidak diperlukan.

### Q: Bisa offline?

**A:** Ya, setelah MNIST didownload dan model di-train, semuanya berjalan offline.

### Q: Apakah data gambar saya disimpan?

**A:** Tidak. Gambar yang dibuat hanya diproses di memory dan tidak disimpan ke disk.

---

**Document Status**: ✅ Complete  
**Need More Help?** Check [SETUP_GUIDE.md](SETUP_GUIDE.md) atau open an issue di GitHub.
