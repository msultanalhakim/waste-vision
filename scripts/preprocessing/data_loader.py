import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from .config import DATASET_DIR, OUTPUT_DIR, SEED


def load_image_paths_and_labels():
    """
    Mengambil seluruh path gambar dan label (nama folder) dari direktori dataset.
    Diasumsikan struktur: dataset_dir/class_name/image.jpg
    """
    image_paths = []
    labels = []

    for class_name in os.listdir(DATASET_DIR): # Iterasi setiap folder kelas
        class_dir = os.path.join(DATASET_DIR, class_name) # Path ke folder kelas
        if os.path.isdir(class_dir): # Pastikan ini adalah direktori
            for file in os.listdir(class_dir): # Iterasi setiap file dalam folder kelas
                if file.lower().endswith((".jpg", ".jpeg", ".png")): # Cek ekstensi file gambar
                    image_paths.append(os.path.join(class_dir, file)) # Path lengkap ke gambar
                    labels.append(class_name)
    return image_paths, labels


def encode_labels(labels):
    """
    Encode label string ke integer dan simpan LabelEncoder.
    """
    le = LabelEncoder() # Inisialisasi LabelEncoder
    encoded = le.fit_transform(labels) # Encode label menjadi integer

    os.makedirs(OUTPUT_DIR, exist_ok=True) # Buat direktori output jika belum ada
    with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f) # Simpan LabelEncoder ke file .pkl

    return encoded


def split_dataset(image_paths, encoded_labels):
    """
    Split data: train 80%, val 10%, test 10%.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, encoded_labels, test_size=0.1, # 10% untuk test set
        stratify=encoded_labels, random_state=SEED # Stratified split untuk menjaga proporsi label
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1111, # 10% dari 90% (yaitu 10% dari total)
        stratify=y_temp, random_state=SEED # Stratified split untuk val set juga
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_and_save_class_weights(y_train):
    """
    Hitung class weight dan simpan ke file .pkl.
    """
    weights = compute_class_weight( # Menghitung class weight
        class_weight='balanced', # Menggunakan metode balanced
        classes=np.unique(y_train), # Kelas unik dari label training
        y=y_train # Label training
    )
    class_weights = dict(enumerate(weights)) # Convert ke dictionary {class_id: weight}

    os.makedirs(OUTPUT_DIR, exist_ok=True) # Buat direktori output jika belum ada
    with open(os.path.join(OUTPUT_DIR, "class_weights.pkl"), "wb") as f: 
        pickle.dump(class_weights, f) # Simpan class weights ke file .pkl

    return class_weights
