import os

# Konfigurasi
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATASET_DIR = os.path.join(BASE_DIR, "data/raw/garbage_dataset")
OUTPUT_DIR = "output/preprocessing"

# Buat output_dir jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)
