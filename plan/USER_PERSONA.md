# 👥 User Persona - Digit Recognition

**Version**: 1.0  
**Date**: 1 Feb 2026  
**Status**: Planning

---

## 1. Overview

Dokumen ini mendefinisikan user persona yang menjadi target utama aplikasi Digit Recognition. Pemahaman mendalam tentang user membantu membuat keputusan desain yang tepat.

### 1.1 Persona Summary

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           USER PERSONAS                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   STUDENT    │  │  EDUCATOR    │  │  DEVELOPER   │  │  RESEARCHER  │     │
│  │              │  │              │  │              │  │              │     │
│  │   "Budi"     │  │   "Dewi"     │  │   "Andi"     │  │   "Rina"     │     │
│  │   Primary    │  │   Secondary  │  │   Secondary  │  │   Tertiary   │     │
│  │              │  │              │  │              │  │              │     │
│  │  Learning    │  │  Teaching    │  │  Prototyping │  │  Baseline    │     │
│  │  ML basics   │  │  ML concepts │  │  & testing   │  │  comparison  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Primary Persona: Student

### 2.1 Profile: Budi

```
╔═══════════════════════════════════════════════════════════════════╗
║                         PERSONA: STUDENT                          ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Name: Budi Santoso                                               ║
║  Age: 21 tahun                                                    ║
║  Location: Bandung, Indonesia                                     ║
║  Education: Mahasiswa Teknik Informatika semester 5               ║
║  Tech Level: Intermediate                                         ║
║                                                                   ║
║  ┌─────────────────────────────────────────────────────────────┐  ║
║  │                                                             │  ║
║  │  "Saya ingin memahami bagaimana neural network bekerja     │  ║
║  │   dari dasar, bukan hanya menggunakan library yang ada."   │  ║
║  │                                                             │  ║
║  └─────────────────────────────────────────────────────────────┘  ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

### 2.2 Demographics

| Attribute     | Value                        |
| ------------- | ---------------------------- |
| **Age Range** | 18-25 tahun                  |
| **Education** | S1 Informatika/Ilmu Komputer |
| **Location**  | Urban Indonesia              |
| **Income**    | Mahasiswa (terbatas)         |
| **Device**    | Laptop mid-range             |
| **OS**        | Windows 10/11                |

### 2.3 Technical Background

```
Technical Skills Assessment:

Python       ████████████████████░░░░░░░░░░ 65%
Mathematics  ██████████████░░░░░░░░░░░░░░░░ 45%
ML/AI        ████████░░░░░░░░░░░░░░░░░░░░░░ 25%
NumPy        ████████████░░░░░░░░░░░░░░░░░░ 40%
Deep Learning ████░░░░░░░░░░░░░░░░░░░░░░░░░░ 15%
```

### 2.4 Goals & Motivations

**Primary Goals:**

- ✅ Memahami konsep neural network dari scratch
- ✅ Menyelesaikan tugas/proyek kuliah
- ✅ Membangun portfolio untuk magang/kerja
- ✅ Persiapan untuk kursus ML lanjutan

**Secondary Goals:**

- Mendapat nilai bagus di mata kuliah
- Bisa menjelaskan konsep ke teman
- Membuat variasi project sendiri

### 2.5 Pain Points & Frustrations

| Pain Point                    | Impact                 | Severity |
| ----------------------------- | ---------------------- | -------- |
| Library ML terlalu abstrak    | Tidak paham cara kerja | High     |
| Matematika ML kompleks        | Sulit mengikuti        | High     |
| Error message tidak jelas     | Stuck debugging        | Medium   |
| Resource komputer terbatas    | Training lambat        | Medium   |
| Dokumentasi berbahasa Inggris | Kurang paham           | Low      |

### 2.6 Behavior Patterns

**Typical Usage:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    BUDI'S TYPICAL DAY                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  09:00  📖 Baca tutorial/dokumentasi                            │
│  10:00  💻 Coba jalankan code contoh                            │
│  11:00  🔧 Modifikasi code untuk eksperimen                     │
│  12:00  🍽️ Istirahat                                             │
│  14:00  📝 Analisis hasil, catat pembelajaran                   │
│  15:00  💬 Diskusi dengan teman jika stuck                      │
│  16:00  📊 Presentasi/submit hasil                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.7 User Journey

```
AWARENESS  →  CONSIDERATION  →  TRIAL  →  ADOPTION  →  ADVOCACY
    │              │             │           │            │
    ▼              ▼             ▼           ▼            ▼
Dapat tugas    Search       Download    Paham        Recommend
ML project   "NN scratch"   & install  konsep NN    ke teman
```

### 2.8 Feature Preferences

| Feature               | Importance | Notes              |
| --------------------- | ---------- | ------------------ |
| Kode yang readable    | ⭐⭐⭐⭐⭐ | Bisa dipelajari    |
| Dokumentasi lengkap   | ⭐⭐⭐⭐⭐ | Bahasa Indonesia + |
| GUI untuk eksperimen  | ⭐⭐⭐⭐   | Visual learner     |
| Step-by-step tutorial | ⭐⭐⭐⭐⭐ | Panduan belajar    |
| Visualisasi training  | ⭐⭐⭐⭐   | Lihat progress     |
| Fast training         | ⭐⭐⭐     | Resource terbatas  |

---

## 3. Secondary Persona: Educator

### 3.1 Profile: Dewi

```
╔═══════════════════════════════════════════════════════════════════╗
║                        PERSONA: EDUCATOR                          ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Name: Dr. Dewi Kusuma                                            ║
║  Age: 38 tahun                                                    ║
║  Location: Jakarta, Indonesia                                     ║
║  Occupation: Dosen Ilmu Komputer                                  ║
║  Tech Level: Advanced                                             ║
║                                                                   ║
║  ┌─────────────────────────────────────────────────────────────┐  ║
║  │                                                             │  ║
║  │  "Saya butuh tool yang bisa mendemonstrasikan konsep       │  ║
║  │   neural network dengan jelas kepada mahasiswa."            │  ║
║  │                                                             │  ║
║  └─────────────────────────────────────────────────────────────┘  ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

### 3.2 Demographics

| Attribute      | Value                |
| -------------- | -------------------- |
| **Age Range**  | 30-50 tahun          |
| **Education**  | S2/S3 Informatika    |
| **Occupation** | Dosen/Trainer        |
| **Class Size** | 30-60 mahasiswa      |
| **Device**     | Laptop/PC good specs |
| **OS**         | Windows/macOS        |

### 3.3 Goals & Motivations

**Primary Goals:**

- ✅ Menjelaskan konsep NN dengan visual yang menarik
- ✅ Memberikan hands-on experience ke mahasiswa
- ✅ Menggunakan contoh yang relatable (digit recognition)
- ✅ Material yang bisa di-reuse setiap semester

**Secondary Goals:**

- Membuat mahasiswa excited tentang ML
- Mengurangi waktu persiapan materi
- Diferensiasi dengan dosen lain

### 3.4 Pain Points

| Pain Point                            | Impact               | Severity |
| ------------------------------------- | -------------------- | -------- |
| Framework terlalu kompleks untuk demo | Mahasiswa bingung    | High     |
| Butuh waktu lama setup environment    | Waktu kelas terbatas | High     |
| Sulit menjelaskan backpropagation     | Konsep abstrak       | High     |
| Mahasiswa dengan skill berbeda        | Sulit sinkronisasi   | Medium   |

### 3.5 Feature Preferences

| Feature                 | Importance | Notes              |
| ----------------------- | ---------- | ------------------ |
| Real-time visualization | ⭐⭐⭐⭐⭐ | Demo di kelas      |
| Simple installation     | ⭐⭐⭐⭐⭐ | Lab setup cepat    |
| Adjustable parameters   | ⭐⭐⭐⭐   | Eksperimen live    |
| Export charts           | ⭐⭐⭐⭐   | Untuk slide        |
| Clean codebase          | ⭐⭐⭐⭐⭐ | Teaching reference |

---

## 4. Secondary Persona: Developer

### 4.1 Profile: Andi

```
╔═══════════════════════════════════════════════════════════════════╗
║                        PERSONA: DEVELOPER                         ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Name: Andi Wijaya                                                ║
║  Age: 28 tahun                                                    ║
║  Location: Surabaya, Indonesia                                    ║
║  Occupation: Software Developer di startup                        ║
║  Tech Level: Advanced                                             ║
║                                                                   ║
║  ┌─────────────────────────────────────────────────────────────┐  ║
║  │                                                             │  ║
║  │  "Saya perlu prototype cepat untuk validasi ide sebelum    │  ║
║  │   invest waktu di framework yang lebih kompleks."           │  ║
║  │                                                             │  ║
║  └─────────────────────────────────────────────────────────────┘  ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

### 4.2 Demographics

| Attribute      | Value                   |
| -------------- | ----------------------- |
| **Age Range**  | 25-35 tahun             |
| **Education**  | S1 Informatika          |
| **Occupation** | Software Developer      |
| **Experience** | 3-7 tahun               |
| **Device**     | MacBook Pro / PC Gaming |
| **OS**         | macOS / Linux           |

### 4.3 Goals & Motivations

**Primary Goals:**

- ✅ Quick prototyping untuk proof of concept
- ✅ Memahami NN internals untuk debugging
- ✅ Baseline comparison dengan framework besar
- ✅ Customizable untuk specific use case

**Secondary Goals:**

- Mengevaluasi apakah NN cocok untuk problem
- Estimasi resource requirements
- Learning untuk career growth

### 4.4 Pain Points

| Pain Point                         | Impact              | Severity |
| ---------------------------------- | ------------------- | -------- |
| Framework besar overkill untuk POC | Wasted time         | High     |
| Black box debugging                | Sulit debug         | High     |
| Dependency hell                    | Installation issues | Medium   |
| Performance untuk production       | Scaling concern     | Medium   |

### 4.5 Feature Preferences

| Feature                 | Importance | Notes                |
| ----------------------- | ---------- | -------------------- |
| Clean API               | ⭐⭐⭐⭐⭐ | Easy to integrate    |
| Extensible architecture | ⭐⭐⭐⭐⭐ | Custom layers        |
| Good performance        | ⭐⭐⭐⭐   | Fast iteration       |
| Model export            | ⭐⭐⭐     | Production migration |
| Minimal dependencies    | ⭐⭐⭐⭐⭐ | Easy deployment      |

---

## 5. Tertiary Persona: Researcher

### 5.1 Profile: Rina

```
╔═══════════════════════════════════════════════════════════════════╗
║                        PERSONA: RESEARCHER                        ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Name: Rina Putri, M.Sc                                           ║
║  Age: 26 tahun                                                    ║
║  Location: Yogyakarta, Indonesia                                  ║
║  Occupation: Mahasiswa S2 / Research Assistant                    ║
║  Tech Level: Advanced                                             ║
║                                                                   ║
║  ┌─────────────────────────────────────────────────────────────┐  ║
║  │                                                             │  ║
║  │  "Saya butuh baseline yang bersih dan reproducible untuk   │  ║
║  │   membandingkan dengan metode baru yang saya propose."      │  ║
║  │                                                             │  ║
║  └─────────────────────────────────────────────────────────────┘  ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

### 5.2 Goals & Motivations

**Primary Goals:**

- ✅ Reproducible baseline untuk paper
- ✅ Memahami setiap komponen untuk variasi
- ✅ Fair comparison tanpa framework overhead
- ✅ Easy to modify untuk eksperimen

### 5.3 Feature Preferences

| Feature           | Importance | Notes                 |
| ----------------- | ---------- | --------------------- |
| Reproducibility   | ⭐⭐⭐⭐⭐ | Fixed seeds           |
| Clear mathematics | ⭐⭐⭐⭐⭐ | Paper citation        |
| Modifiable code   | ⭐⭐⭐⭐⭐ | Experiment variations |
| Benchmark results | ⭐⭐⭐⭐   | Comparison            |
| Logging/export    | ⭐⭐⭐⭐   | Result analysis       |

---

## 6. Anti-Personas

### 6.1 Who is NOT our target user

```
┌─────────────────────────────────────────────────────────────────┐
│                      ANTI-PERSONAS                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ❌ Production ML Engineer                                       │
│     → Butuh scalability dan GPU support                         │
│     → Lebih cocok: PyTorch, TensorFlow                          │
│                                                                 │
│  ❌ Data Scientist Enterprise                                    │
│     → Butuh banyak model types dan integrations                 │
│     → Lebih cocok: scikit-learn, XGBoost                        │
│                                                                 │
│  ❌ Computer Vision Expert                                       │
│     → Butuh ConvNets dan transfer learning                      │
│     → Lebih cocok: torchvision, keras                           │
│                                                                 │
│  ❌ Non-technical User                                           │
│     → Butuh zero-code solution                                  │
│     → Lebih cocok: AutoML tools                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Persona-Feature Matrix

### 7.1 Feature Prioritization by Persona

| Feature             | Student    | Educator   | Developer  | Researcher |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| **GUI Canvas**      | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐     | ⭐⭐       |
| **Training Viz**    | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | ⭐⭐⭐     | ⭐⭐⭐⭐   |
| **Clean Code**      | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ |
| **Documentation**   | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   | ⭐⭐⭐     | ⭐⭐⭐     |
| **Fast Training**   | ⭐⭐⭐     | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   |
| **Extensibility**   | ⭐⭐       | ⭐⭐       | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Reproducibility** | ⭐⭐⭐     | ⭐⭐⭐     | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ |
| **Tutorials**       | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   | ⭐⭐       | ⭐⭐       |

### 7.2 Priority Matrix

```
                        HIGH IMPACT
                            ▲
                            │
           ┌────────────────┼────────────────┐
           │                │                │
           │  GUI Canvas    │  Clean Code    │
           │  Training Viz  │  Documentation │
           │                │                │
     LOW   ├────────────────┼────────────────┤  HIGH
    EFFORT │                │                │  EFFORT
           │  Fast Train    │  Extensibility │
           │  Tutorials     │  Reproducible  │
           │                │                │
           └────────────────┼────────────────┘
                            │
                            ▼
                        LOW IMPACT
```

---

## 8. User Scenarios

### 8.1 Scenario: Budi's ML Assignment

```
SCENARIO: Budi mengerjakan tugas mata kuliah ML

CONTEXT:
- Deadline: 2 minggu
- Task: Implementasi NN untuk digit recognition
- Requirements: Kode sendiri (no framework), dokumentasi

STEPS:
1. Budi search "neural network from scratch python"
2. Menemukan project ini, baca README
3. Clone repo, ikuti setup guide
4. Jalankan demo, lihat prediksi bekerja
5. Baca MATHEMATICAL_FOUNDATION.md
6. Baca NEURAL_NETWORK_DESIGN.md
7. Coba modifikasi arsitektur network
8. Eksperimen dengan hyperparameter
9. Dokumentasi hasil eksperimen
10. Submit tugas dengan pemahaman lengkap

OUTCOME:
✅ Tugas selesai dengan nilai A
✅ Paham konsep NN dari dasar
✅ Bisa menjelaskan ke dosen saat presentasi
```

### 8.2 Scenario: Dewi's Classroom Demo

```
SCENARIO: Dewi demo NN di kelas Machine Learning

CONTEXT:
- Class: 40 mahasiswa
- Time: 2 jam lab session
- Goal: Hands-on experience dengan NN

STEPS:
1. Sebelum kelas: Install di semua PC lab
2. Mulai kelas: Jelaskan teori NN (15 min)
3. Demo: Gambar digit, tunjukkan prediksi
4. Demo: Jalankan training, tunjukkan loss turun
5. Mahasiswa: Coba gambar sendiri
6. Mahasiswa: Eksperimen learning rate
7. Diskusi: Apa yang terjadi jika LR terlalu besar?
8. Demo: Tunjukkan kode, jelaskan forward pass
9. Assignment: Modifikasi arsitektur di rumah

OUTCOME:
✅ Mahasiswa engaged dan excited
✅ Konsep abstrak jadi concrete
✅ Hands-on experience membekas
```

---

## 9. Design Implications

### 9.1 Based on Persona Analysis

| Insight                          | Design Implication                      |
| -------------------------------- | --------------------------------------- |
| Students need readable code      | Comment extensively, consistent style   |
| Educators need visual demos      | Prominent GUI, real-time charts         |
| Developers need extensibility    | Clean architecture, abstract interfaces |
| Researchers need reproducibility | Fixed seeds, deterministic behavior     |

### 9.2 UI/UX Priorities

```
PRIORITY 1 (All personas):
├── Intuitive drawing canvas
├── Clear prediction display
└── Visible training progress

PRIORITY 2 (Students & Educators):
├── Step-by-step tutorials
├── Comprehensive documentation
└── Error messages yang jelas

PRIORITY 3 (Developers & Researchers):
├── Clean API
├── Extensible architecture
└── Benchmark scripts
```

---

## 10. Success Metrics by Persona

### 10.1 Measuring Persona Satisfaction

| Persona        | Success Metric                    | Target   |
| -------------- | --------------------------------- | -------- |
| **Student**    | Can explain NN concepts after use | 80%      |
| **Student**    | Complete project successfully     | 90%      |
| **Educator**   | Use in class again next semester  | 70%      |
| **Educator**   | Recommend to colleagues           | 60%      |
| **Developer**  | Time to working prototype         | < 30 min |
| **Developer**  | Successfully customize            | 80%      |
| **Researcher** | Reproducible results              | 100%     |
| **Researcher** | Use as paper baseline             | 50%      |

---

**Document Status**: ✅ Complete  
**Related Documents**:

- [PRD.md](PRD.md)
- [GUI_DESIGN.md](GUI_DESIGN.md)
- [USER_GUIDE.md](USER_GUIDE.md)
