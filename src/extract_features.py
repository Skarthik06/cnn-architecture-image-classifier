import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# ==============================
# CONFIGURATION
# ==============================
IMG_SIZE = 128
BATCH_SIZE = 32

DATA_DIR = r"D:\SK\dog_cat_cnn_project\dataset\validation"
SAVE_DIR = r"D:\SK\dog_cat_cnn_project\datasets"
os.makedirs(SAVE_DIR, exist_ok=True)

MODELS = {
    "V1_Advanced_CNN": r"D:\SK\dog_cat_cnn_project\experiments\v1_baseline\model\v1_model.keras",
    "V2_Augmented_CNN": r"D:\SK\dog_cat_cnn_project\experiments\v2_augmentation\model\v2_model.keras",
    "V3_Transfer_Learning": r"D:\SK\dog_cat_cnn_project\experiments\v3_transfer_learning\model\v3_model.keras"
}

# ==============================
# LOAD DATASET (SAFE)
# ==============================
def load_dataset():
    print("\n📂 Loading validation dataset...")
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    # ✅ Correct way (TensorFlow ≥ 2.12)
    dataset = dataset.ignore_errors()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    print("✅ Dataset loaded successfully")
    return dataset


dataset = load_dataset()

# ==============================
# FEATURE EXTRACTION
# ==============================
for model_name, model_path in MODELS.items():
    print(f"\n🚀 Extracting features from {model_name}")

    model = tf.keras.models.load_model(model_path)

    # 🔍 Find best Dense feature layer (not softmax)
    feature_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Dense) and layer.units != 2:
            feature_layer = layer
            break

    if feature_layer is None:
        raise RuntimeError(f"❌ Feature layer not found in {model_name}")

    feature_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=feature_layer.output
    )

    features = []
    labels = []

    for images, lbls in dataset:
        feats = feature_model.predict(images, verbose=0)
        features.append(feats)
        labels.extend(lbls.numpy())

    features = np.vstack(features)

    # ==============================
    # NORMALIZATION (DENSE MATRIX)
    # ==============================
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # ==============================
    # CREATE DATAFRAME
    # ==============================
    feature_cols = [f"feature_{i+1}" for i in range(features.shape[1])]

    df = pd.DataFrame(features, columns=feature_cols)
    df["label"] = labels
    df["model_version"] = model_name

    csv_path = os.path.join(SAVE_DIR, f"{model_name}_features.csv")
    df.to_csv(csv_path, index=False)

    zero_ratio = (df == 0).sum().sum() / df.size

    print(f"✅ Saved: {csv_path}")
    print(f"   Shape       : {df.shape}")
    print(f"   Zero ratio  : {zero_ratio:.4f}")

print("\n🎉 Feature extraction completed successfully for V1, V2 & V3!")
