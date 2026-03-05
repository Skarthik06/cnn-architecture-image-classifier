import tensorflow as tf
import os

IMG_SIZE = 128
BATCH_SIZE = 32

AUTOTUNE = tf.data.AUTOTUNE


# 🔹 Internal helper to create dataset safely
def _create_dataset(directory):
    if directory is None or not os.path.exists(directory):
        raise ValueError(f"❌ Dataset directory not found: {directory}")

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    # Ignore corrupted images
    dataset = dataset.apply(tf.data.experimental.ignore_errors())

    # Performance optimization
    dataset = dataset.cache().prefetch(AUTOTUNE)

    return dataset


# 🔹 Used during training (V1, V2, V3)
def load_train_val_data(train_dir, val_dir):
    print("📂 Loading training data from:", train_dir)
    train_data = _create_dataset(train_dir)

    print("📂 Loading validation data from:", val_dir)
    val_data = _create_dataset(val_dir)

    return train_data, val_data


# 🔹 Used during evaluation only
def load_only_validation_data(val_dir):
    print("📂 Loading validation data from:", val_dir)
    val_data = _create_dataset(val_dir)
    return val_data
