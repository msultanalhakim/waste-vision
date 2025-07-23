"""
Pipeline untuk preprocessing dataset klasifikasi gambar:
- Load path dan label dari direktori dataset
- Encode label menjadi integer
- Split menjadi train, val, test (80/10/10)
- Hitung class weight untuk mengatasi ketidakseimbangan kelas
- Bangun pipeline tf.data untuk training, validasi, dan pengujian
"""

from preprocessing.data_loader import (
    load_image_paths_and_labels,
    encode_labels,
    split_dataset,
    compute_and_save_class_weights,
)

from preprocessing.dataset_builder import build_dataset


# Langkah 1: Load paths dan label
# Membaca semua path gambar dan nama label (berdasarkan folder)
image_paths, labels = load_image_paths_and_labels()

# Langkah 2: Encode label string jadi integer
# Misal: ['organic', 'inorganic', 'organic'] -> [1, 0, 1]
encoded_labels = encode_labels(labels)

# Langkah 3: Split data menjadi train (80%), val (10%), test (10%)
# Stratified split agar proporsi label tetap konsisten di semua set
X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(image_paths, encoded_labels)

# Langkah 4: Hitung class weights
# Untuk digunakan saat training agar model lebih adil terhadap kelas minoritas
class_weights = compute_and_save_class_weights(y_train)

# Langkah 5: Buat pipeline tf.data
# Training set diberi augmentasi, val dan test hanya preprocessing
train_ds = build_dataset(X_train, y_train, training=True)
val_ds   = build_dataset(X_val, y_val, training=False)
test_ds  = build_dataset(X_test, y_test, training=False)
