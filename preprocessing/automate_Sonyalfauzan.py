"""
automate_Sonyalfauzan.py
-------------------------
Script otomatisasi preprocessing untuk Wine Quality Dataset.
Melakukan seluruh tahapan preprocessing yang sama dengan notebook
eksperimen, namun dalam bentuk fungsi modular yang dapat dijalankan
secara otomatis maupun dipanggil sebagai modul Python.

Pipeline:
    1. Load dataset (dari URL atau file lokal)
    2. Hapus baris duplikat
    3. Encode target variable (quality >= 7 → Good=1, Bad=0)
    4. Penanganan outlier — IQR Capping (Winsorization)
    5. Standarisasi fitur — StandardScaler
    6. Train-Test Split (stratified, 80:20)
    7. Simpan semua artefak preprocessing

Penggunaan:
    # Default (download dari UCI)
    python automate_Sonyalfauzan.py

    # Custom
    python automate_Sonyalfauzan.py \\
        --input winequality_raw/winequality-red.csv \\
        --output_dir winequality_preprocessing \\
        --test_size 0.2 \\
        --quality_threshold 7

Author : Sonyalfauzan
Date   : 2025
"""

import os
import sys
import pickle
import logging
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("preprocessing_Sonyalfauzan.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
DEFAULT_URL       = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
TARGET_COL        = "quality"
QUALITY_THRESHOLD = 7
RANDOM_STATE      = 42
TEST_SIZE_DEFAULT = 0.2


# ---------------------------------------------------------------------------
# Fungsi-Fungsi Pipeline
# ---------------------------------------------------------------------------

def load_data(source: str) -> pd.DataFrame:
    """
    Memuat dataset dari file lokal (CSV) atau URL.

    Args:
        source (str): Path file lokal atau URL HTTP/HTTPS.

    Returns:
        pd.DataFrame: Dataset mentah yang sudah dimuat.

    Raises:
        FileNotFoundError: Jika file lokal tidak ditemukan.
        ValueError: Jika dataframe kosong setelah load.
    """
    logger.info("LANGKAH 1: Memuat dataset dari → %s", source)

    if source.startswith("http://") or source.startswith("https://"):
        df = pd.read_csv(source, sep=";")
    elif os.path.isfile(source):
        sep = ";" if source.endswith(".csv") else ","
        df  = pd.read_csv(source, sep=sep)
    else:
        raise FileNotFoundError(f"Sumber data tidak ditemukan: {source}")

    if df.empty:
        raise ValueError("Dataset kosong! Periksa kembali sumber data.")

    logger.info(
        "  Dataset dimuat → %d baris × %d kolom | %.2f KB",
        df.shape[0], df.shape[1],
        df.memory_usage(deep=True).sum() / 1024,
    )
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat dari dataset.

    Args:
        df (pd.DataFrame): Dataset input.

    Returns:
        pd.DataFrame: Dataset tanpa duplikat.
    """
    logger.info("LANGKAH 2: Menghapus baris duplikat ...")
    before     = len(df)
    df_clean   = df.drop_duplicates().reset_index(drop=True)
    removed    = before - len(df_clean)
    logger.info(
        "  Dihapus: %d baris (%.2f%%) | Tersisa: %d baris",
        removed, removed / before * 100, len(df_clean),
    )
    return df_clean


def encode_target(df: pd.DataFrame, threshold: int = QUALITY_THRESHOLD) -> pd.DataFrame:
    """
    Mengubah kolom 'quality' menjadi label binary.
        quality >= threshold → 1 (Good)
        quality <  threshold → 0 (Bad)

    Args:
        df (pd.DataFrame): Dataset dengan kolom 'quality'.
        threshold (int): Batas ambang kualitas baik (default: 7).

    Returns:
        pd.DataFrame: Dataset dengan kolom 'quality_binary', tanpa 'quality'.
    """
    logger.info("LANGKAH 3: Encoding target variable (threshold ≥ %d → Good=1) ...", threshold)
    df = df.copy()
    df["quality_binary"] = (df[TARGET_COL] >= threshold).astype(int)
    df = df.drop(columns=[TARGET_COL])

    vc = df["quality_binary"].value_counts()
    logger.info(
        "  Bad  (0): %d sampel | Good (1): %d sampel | Rasio: %.2f:1",
        vc.get(0, 0), vc.get(1, 0), vc.get(0, 0) / max(vc.get(1, 1), 1),
    )
    return df


def handle_outliers_iqr(df: pd.DataFrame, target_col: str = "quality_binary") -> pd.DataFrame:
    """
    Menangani outlier menggunakan metode IQR Capping (Winsorization).
    Nilai di bawah Q1 - 1.5×IQR akan dikap ke batas bawah.
    Nilai di atas Q3 + 1.5×IQR akan dikap ke batas atas.

    Args:
        df (pd.DataFrame): Dataset input.
        target_col (str): Kolom target yang dikecualikan dari proses.

    Returns:
        pd.DataFrame: Dataset dengan outlier yang sudah dikap.
    """
    logger.info("LANGKAH 4: Penanganan outlier — IQR Capping ...")
    df_capped    = df.copy()
    feature_cols = [c for c in df.columns if c != target_col]
    total_capped = 0

    for col in feature_cols:
        Q1  = df_capped[col].quantile(0.25)
        Q3  = df_capped[col].quantile(0.75)
        IQR = Q3 - Q1
        lo  = Q1 - 1.5 * IQR
        hi  = Q3 + 1.5 * IQR

        mask_out   = (df_capped[col] < lo) | (df_capped[col] > hi)
        n_cap      = int(mask_out.sum())
        total_capped += n_cap

        df_capped[col] = df_capped[col].clip(lower=lo, upper=hi)
        if n_cap:
            logger.debug("    [%s] dikap %d nilai — batas: [%.4f, %.4f]", col, n_cap, lo, hi)

    logger.info("  Total nilai dikap: %d pada %d fitur", total_capped, len(feature_cols))
    return df_capped


def scale_features(
    df: pd.DataFrame,
    target_col: str = "quality_binary",
) -> tuple:
    """
    Menstandarisasi semua fitur menggunakan StandardScaler
    (mean ≈ 0, std ≈ 1).

    Args:
        df (pd.DataFrame): Dataset setelah penanganan outlier.
        target_col (str): Kolom target yang dikecualikan.

    Returns:
        tuple: (df_scaled, scaler_fitted)
    """
    logger.info("LANGKAH 5: Standarisasi fitur — StandardScaler ...")
    feature_cols = [c for c in df.columns if c != target_col]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    df_scaled[target_col] = df[target_col].values

    logger.info(
        "  Mean range: [%.4f, %.4f] | Std range: [%.4f, %.4f]",
        df_scaled[feature_cols].mean().min(),
        df_scaled[feature_cols].mean().max(),
        df_scaled[feature_cols].std().min(),
        df_scaled[feature_cols].std().max(),
    )
    return df_scaled, scaler


def split_dataset(
    df: pd.DataFrame,
    target_col: str = "quality_binary",
    test_size: float = TEST_SIZE_DEFAULT,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Membagi dataset menjadi training dan testing set (stratified).

    Args:
        df (pd.DataFrame): Dataset lengkap yang sudah dipreproses.
        target_col (str): Kolom target untuk stratifikasi.
        test_size (float): Proporsi data test (0.0–1.0).
        random_state (int): Seed untuk reprodusibilitas.

    Returns:
        tuple: (df_train, df_test)
    """
    logger.info(
        "LANGKAH 6: Train-Test Split → train: %.0f%% | test: %.0f%% (stratified) ...",
        (1 - test_size) * 100, test_size * 100,
    )
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    df_train = pd.concat(
        [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
    )
    df_test  = pd.concat(
        [X_test.reset_index(drop=True),  y_test.reset_index(drop=True)],  axis=1
    )

    logger.info(
        "  Train: %d baris | Test: %d baris | Fitur: %d",
        len(df_train), len(df_test), X.shape[1],
    )
    return df_train, df_test


def save_artifacts(
    df_full: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    scaler: StandardScaler,
    output_dir: str,
) -> None:
    """
    Menyimpan semua artefak preprocessing ke direktori output.

    File yang disimpan:
        - winequality_preprocessed.csv  (dataset penuh)
        - train.csv                     (data latih)
        - test.csv                      (data uji)
        - scaler.pkl                    (StandardScaler fitted)

    Args:
        df_full (pd.DataFrame): Dataset penuh hasil preprocessing.
        df_train (pd.DataFrame): Data latih.
        df_test (pd.DataFrame): Data uji.
        scaler (StandardScaler): Scaler yang sudah di-fit.
        output_dir (str): Direktori tujuan penyimpanan.
    """
    logger.info("LANGKAH 7: Menyimpan artefak ke '%s' ...", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "full":   os.path.join(output_dir, "winequality_preprocessed.csv"),
        "train":  os.path.join(output_dir, "train.csv"),
        "test":   os.path.join(output_dir, "test.csv"),
        "scaler": os.path.join(output_dir, "scaler.pkl"),
    }

    df_full.to_csv(paths["full"],  index=False)
    df_train.to_csv(paths["train"], index=False)
    df_test.to_csv(paths["test"],  index=False)

    with open(paths["scaler"], "wb") as fh:
        pickle.dump(scaler, fh)

    logger.info("  ✅ %s  (%d baris)", paths["full"],  len(df_full))
    logger.info("  ✅ %s (%d baris)", paths["train"], len(df_train))
    logger.info("  ✅ %s  (%d baris)", paths["test"],  len(df_test))
    logger.info("  ✅ %s",  paths["scaler"])


# ---------------------------------------------------------------------------
# Pipeline Utama
# ---------------------------------------------------------------------------

def run_pipeline(
    source: str,
    output_dir: str,
    test_size: float = TEST_SIZE_DEFAULT,
    quality_threshold: int = QUALITY_THRESHOLD,
) -> dict:
    """
    Menjalankan pipeline preprocessing end-to-end.

    Args:
        source (str): Sumber dataset (path lokal atau URL).
        output_dir (str): Direktori penyimpanan output.
        test_size (float): Proporsi data test.
        quality_threshold (int): Ambang batas kelas Good.

    Returns:
        dict: Ringkasan metadata pipeline.
    """
    logger.info("=" * 60)
    logger.info("PIPELINE PREPROCESSING — Wine Quality (Sonyalfauzan)")
    logger.info("=" * 60)

    df_raw                = load_data(source)
    df_dedup              = remove_duplicates(df_raw)
    df_encoded            = encode_target(df_dedup, threshold=quality_threshold)
    df_capped             = handle_outliers_iqr(df_encoded)
    df_scaled, scaler     = scale_features(df_capped)
    df_train, df_test     = split_dataset(df_scaled, test_size=test_size)
    save_artifacts(df_scaled, df_train, df_test, scaler, output_dir)

    summary = {
        "author":            "Sonyalfauzan",
        "raw_rows":          len(df_raw),
        "preprocessed_rows": len(df_scaled),
        "removed_duplicates": len(df_raw) - len(df_dedup),
        "n_features":        df_scaled.shape[1] - 1,
        "train_rows":        len(df_train),
        "test_rows":         len(df_test),
        "class_0_bad":       int((df_scaled["quality_binary"] == 0).sum()),
        "class_1_good":      int((df_scaled["quality_binary"] == 1).sum()),
        "output_dir":        output_dir,
    }

    logger.info("=" * 60)
    logger.info("PIPELINE SELESAI — Ringkasan:")
    for key, val in summary.items():
        logger.info("  %-25s : %s", key, val)
    logger.info("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated preprocessing pipeline — Wine Quality Dataset (Sonyalfauzan)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, default=DEFAULT_URL,
        help="Path lokal atau URL ke dataset CSV mentah."
    )
    parser.add_argument(
        "--output_dir", type=str, default="winequality_preprocessing",
        help="Direktori output untuk artefak preprocessing."
    )
    parser.add_argument(
        "--test_size", type=float, default=TEST_SIZE_DEFAULT,
        help="Proporsi data testing (0.0–1.0)."
    )
    parser.add_argument(
        "--quality_threshold", type=int, default=QUALITY_THRESHOLD,
        help="Ambang batas kualitas untuk kelas Good (quality >= threshold)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    summary = run_pipeline(
        source            = args.input,
        output_dir        = args.output_dir,
        test_size         = args.test_size,
        quality_threshold = args.quality_threshold,
    )
    sys.exit(0)
