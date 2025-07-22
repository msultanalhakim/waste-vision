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

    for class_name in os.listdir(DATASET_DIR):
        class_dir = os.path.join(DATASET_DIR, class_name)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(class_dir, file))
                    labels.append(class_name)
    return image_paths, labels


def encode_labels(labels):
    """
    Encode label string ke integer dan simpan LabelEncoder.
    """
    le = LabelEncoder()
    encoded = le.fit_transform(labels)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    return encoded


def split_dataset(image_paths, encoded_labels):
    """
    Split data: train 80%, val 10%, test 10%.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, encoded_labels, test_size=0.1,
        stratify=encoded_labels, random_state=SEED
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1111,
        stratify=y_temp, random_state=SEED
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_and_save_class_weights(y_train):
    """
    Hitung class weight dan simpan ke file .pkl.
    """
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(weights))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "class_weights.pkl"), "wb") as f:
        pickle.dump(class_weights, f)

    return class_weights
