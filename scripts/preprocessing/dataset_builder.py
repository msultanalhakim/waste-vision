import tensorflow as tf
from .config import IMAGE_SIZE, BATCH_SIZE

def preprocess_image(path, label):
    """
    Membaca dan memproses gambar dari path:
    
    Proses meliputi:
    - Membaca file gambar dari path sebagai byte string.
    - Mendekode gambar JPEG menjadi tensor format RGB.
    - Meresize gambar ke ukuran standar IMAGE_SIZE.
    - Normalisasi piksel ke rentang [0, 1].

    Args:
        path (str): Lokasi file gambar.
        label (int/str): Label yang terkait dengan gambar.

    Returns:
        tuple: (image_tensor, label)
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # Normalisasi ke [0,1]
    return image, label

def data_augmentation(image, label):
    """
    Menerapkan augmentasi data ringan untuk memperkaya variasi gambar saat training:
    
    Teknik augmentasi meliputi:
    - Flip horizontal acak
    - Penyesuaian brightness dan contrast secara acak

    Args:
        image (Tensor): Gambar hasil preprocessing.
        label (int/str): Label yang sesuai.

    Returns:
        tuple: (augmented_image_tensor, label)
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label

def build_dataset(paths, labels, training=True):
    """
    Membangun pipeline `tf.data.Dataset` yang efisien untuk training/validasi.

    Pipeline ini mencakup:
    - Membaca dan memproses gambar dari file path
    - Augmentasi data jika mode training
    - Shuffling dan batching
    - Prefetch untuk performa optimal

    Args:
        paths (list): Daftar path gambar.
        labels (list): Daftar label yang sesuai.
        training (bool): Apakah dataset digunakan untuk training.

    Returns:
        tf.data.Dataset: Dataset siap dilatih/divalidasi.
    """
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if training:
        ds = ds.shuffle(buffer_size=1024)  # Agar batch tidak overfit urutan
        ds = ds.map(data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)  # Pipeline non-blocking
    
    return ds
