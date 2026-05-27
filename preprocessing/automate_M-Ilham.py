# automate_Nama-siswa.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(input_path='heart_disease_uci_raw.csv', output_path='preprocessing/heart_disease_uci_processing.csv'):
    """
    Melakukan preprocessing otomatis pada dataset Heart Disease UCI.
    Mengembalikan DataFrame yang sudah siap dilatih dan menyimpannya ke CSV.
    """
    # Load data
    df = pd.read_csv(input_path)
    print(f"Data loaded: {df.shape}")

    # Drop kolom tidak relevan
    df_clean = df.drop(columns=['id', 'dataset'], errors='ignore')
    print(f"Kolom setelah drop: {df_clean.columns.tolist()}")

    # Penanganan Missing Values (jika ada)
    missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()
    if missing_cols:
        for col in missing_cols:
            if df_clean[col].dtype in ['float64', 'int64']:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"Missing {col}: diisi median={median_val}")
            else:
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                print(f"Missing {col}: diisi modus={mode_val}")
    else:
        print("Tidak ada missing values.")

    # Encoding Data Kategorikal
    cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        le = LabelEncoder()
        for col in cat_cols:
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        print(f"Encoding selesai: {cat_cols}")
    else:
        print("Tidak ada kolom kategorikal.")

    # Penanganan Outlier (Winsorization)
    outlier_cols = ['trestbps', 'chol', 'thalch', 'oldpeak']
    for col in outlier_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean[col] = np.where(df_clean[col] < lower, lower, df_clean[col])
        df_clean[col] = np.where(df_clean[col] > upper, upper, df_clean[col])
        print(f"Outlier {col}: winsorized (lower={lower:.2f}, upper={upper:.2f})")

    # Pisahkan Fitur dan Target
    X = df_clean.drop(columns=['num'])
    y = df_clean['num']

    # Standardisasi Fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    print("Standardisasi selesai.")

    # Gabung Kembali
    df_preprocessed = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)

    # Simpan
    os.makedirs('preprocessing', exist_ok=True)  # jika dipanggil dari root
    df_preprocessed.to_csv(output_path, index=False)
    print(f"Dataset preprocessed disimpan ke {output_path}")
    return df_preprocessed

if __name__ == "__main__":
    # Panggil langsung saat script dieksekusi
    preprocess()