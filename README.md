# Eksperimen_SML_Sonyalfauzan

Repository eksperimen machine learning — Kelas **Membangun Sistem Machine Learning (MSML)** Dicoding.

**Author:** Sonyalfauzan  
**GitHub:** https://github.com/Sonyalfauzan  
**DagsHub:** https://dagshub.com/Sonyalfauzan

---

## Dataset

**Wine Quality Dataset — Red Wine**  
Sumber: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)

| Atribut | Detail |
|---------|--------|
| Sumber | UCI ML Repository |
| Jumlah sampel | 1.599 |
| Fitur | 11 (fisikokimia) |
| Target | Binary: Good (quality ≥ 7) / Bad (quality < 7) |
| Tipe masalah | Binary Classification |

---

## Struktur Repository

```
Eksperimen_SML_Sonyalfauzan/
├── .github/
│   └── workflows/
│       └── preprocessing.yml          ← GitHub Actions (Advanced)
├── winequality_raw/
│   └── winequality-red.csv            ← Dataset mentah
├── preprocessing/
│   ├── Eksperimen_Sonyalfauzan.ipynb  ← Notebook eksperimen lengkap
│   ├── automate_Sonyalfauzan.py       ← Script otomatisasi
│   └── winequality_preprocessing/
│       ├── winequality_preprocessed.csv
│       ├── train.csv
│       └── test.csv
└── README.md
```

---

## Struktur Notebook (mengikuti Template MSML)

| Section | Konten |
|---------|--------|
| **1. Perkenalan Dataset** | Deskripsi dataset, business understanding, problem statement |
| **2. Import Library** | NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn |
| **3. Memuat Dataset** | Load dari UCI URL, simpan lokal |
| **4. EDA** | Missing values, duplikat, distribusi target, histogram fitur, correlation matrix, boxplot, outlier IQR |
| **5. Data Preprocessing** | Hapus duplikat, encode target, IQR capping, StandardScaler, train-test split, simpan artefak |

---

## Cara Menjalankan

### Notebook Eksperimen (Basic)
```bash
pip install jupyter scikit-learn pandas matplotlib seaborn
jupyter notebook preprocessing/Eksperimen_Sonyalfauzan.ipynb
```

### Script Otomatis (Skilled)
```bash
pip install scikit-learn pandas numpy scipy

# Default — download dari UCI
python preprocessing/automate_Sonyalfauzan.py

# Custom parameter
python preprocessing/automate_Sonyalfauzan.py \
    --input winequality_raw/winequality-red.csv \
    --output_dir preprocessing/winequality_preprocessing \
    --test_size 0.2 \
    --quality_threshold 7
```

### GitHub Actions (Advanced)
Workflow otomatis terpantik saat ada push ke `main`/`master` pada path `preprocessing/**`.

---

## Tahapan Preprocessing

| # | Langkah | Metode | Alasan |
|---|---------|--------|--------|
| 1 | Hapus duplikat | `drop_duplicates()` | Cegah overfitting |
| 2 | Encode target | quality ≥ 7 → Good(1) | Ubah ke binary classification |
| 3 | Outlier handling | IQR Capping (Winsorization) | Pertahankan data, batasi nilai ekstrem |
| 4 | Feature scaling | `StandardScaler` | Samakan skala fitur |
| 5 | Train-test split | 80:20, stratified | Jaga distribusi kelas |

---

*Dibuat untuk submission Kelas Membangun Sistem Machine Learning — Dicoding*
