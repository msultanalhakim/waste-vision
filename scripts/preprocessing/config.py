import os

# Konfigurasi
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Path
DATASET_DIR = "dataset"
OUTPUT_DIR = "output_prepro"

# Buat output_dir jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)
