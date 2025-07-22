# image_eda.py
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

# Konstanta: Lokasi dataset. Ubah sesuai kebutuhanmu.
DATASET_DIR = "../data/raw/garbage_dataset"  # Contoh struktur folder: /class_name/image.jpg

# Fungsi untuk memeriksa apakah dataset memiliki struktur folder yang sesuai
def validate_dataset_structure(dataset_dir):
    print("[INFO] Validating dataset structure...")
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")
    
    class_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    if not class_dirs:
        raise ValueError("No class subdirectories found inside the dataset directory.")
    
    print(f"[INFO] Found {len(class_dirs)} class(es): {class_dirs}")
    return class_dirs

# Fungsi untuk menghitung jumlah gambar per kelas
def count_images_per_class(dataset_dir, class_dirs):
    print("[INFO] Counting images per class...")
    class_counts = {}
    for class_name in class_dirs:
        class_path = os.path.join(dataset_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[class_name] = len(images)
        print(f"  - {class_name}: {len(images)} image(s)")
    return class_counts

# Fungsi untuk menampilkan grafik distribusi kelas
def plot_class_distribution(class_counts):
    print("[INFO] Plotting class distribution...")
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.xticks(rotation=45)
    plt.title("Distribusi Jumlah Gambar per Kelas")
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah Gambar")
    plt.tight_layout()
    plt.show()

# Fungsi untuk menampilkan contoh gambar dari setiap kelas
def show_sample_images(dataset_dir, class_dirs, img_size=(128, 128)):
    print("[INFO] Showing sample image from each class...")
    num_classes = len(class_dirs)
    plt.figure(figsize=(num_classes * 2.5, 3))
    
    for idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(dataset_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            continue
        img_path = os.path.join(class_path, images[0])  # Ambil gambar pertama sebagai contoh
        try:
            img = Image.open(img_path).resize(img_size)
            plt.subplot(1, num_classes, idx + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(class_name, fontsize=10)
        except Exception as e:
            print(f"[WARNING] Could not load image '{img_path}': {e}")
    plt.tight_layout()
    plt.show()

# Fungsi untuk menghitung statistik ukuran gambar
def analyze_image_dimensions(dataset_dir, class_dirs):
    print("[INFO] Analyzing image dimensions...")
    widths, heights = [], []

    for class_name in class_dirs:
        class_path = os.path.join(dataset_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_file in images:
            img_path = os.path.join(class_path, img_file)
            try:
                img = Image.open(img_path)
                w, h = img.size
                widths.append(w)
                heights.append(h)
            except Exception as e:
                print(f"[WARNING] Could not read image '{img_path}': {e}")
    
    print(f"[INFO] Total analyzed images: {len(widths)}")

    # Plot distribusi lebar dan tinggi gambar
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(widths, kde=True)
    plt.title("Distribusi Lebar Gambar")
    plt.xlabel("Lebar (px)")

    plt.subplot(1, 2, 2)
    sns.histplot(heights, kde=True)
    plt.title("Distribusi Tinggi Gambar")
    plt.xlabel("Tinggi (px)")

    plt.tight_layout()
    plt.show()

    # Menampilkan rata-rata dan modus ukuran
    avg_width = np.mean(widths)
    avg_height = np.mean(heights)
    print(f"Rata-rata ukuran gambar: {avg_width:.1f} x {avg_height:.1f} px")


# Fungsi utama untuk menjalankan semua proses EDA
def run_eda(dataset_dir):
    class_dirs = validate_dataset_structure(dataset_dir)
    class_counts = count_images_per_class(dataset_dir, class_dirs)
    plot_class_distribution(class_counts)
    show_sample_images(dataset_dir, class_dirs)
    analyze_image_dimensions(dataset_dir, class_dirs)


if __name__ == "__main__":
    run_eda(DATASET_DIR)
