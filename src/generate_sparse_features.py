import pandas as pd
import numpy as np
import os

# ==========================
# CONFIGURATION
# ==========================
INPUT_DIR = r"D:\SK\dog_cat_cnn_project\datasets"
OUTPUT_DIR = r"D:\SK\dog_cat_cnn_project\datasets\sparse"

THRESHOLD = 0.15  # controls sparsity (increase = more zeros)

os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = [
    "V1_Advanced_CNN_features.csv",
    "V2_Augmented_CNN_features.csv",
    "V3_Transfer_Learning_features.csv"
]

print("🚀 Generating Sparse Feature Matrices...\n")

for file in FILES:
    print(f"📂 Processing {file}")

    df = pd.read_csv(os.path.join(INPUT_DIR, file))

    # Separate metadata
    labels = df["label"]
    model_version = df["model_version"]

    feature_df = df.drop(columns=["label", "model_version"])

    # 🔥 Convert to sparse: small values -> 0
    sparse_features = feature_df.where(
        np.abs(feature_df) >= THRESHOLD, 0
    )

    # Reattach metadata
    sparse_features["label"] = labels
    sparse_features["model_version"] = model_version

    output_path = os.path.join(
        OUTPUT_DIR, file.replace(".csv", "_SPARSE.csv")
    )

    sparse_features.to_csv(output_path, index=False)

    zero_ratio = (sparse_features == 0).sum().sum() / sparse_features.size
    print(f"✅ Saved: {output_path}")
    print(f"📊 Sparsity level: {zero_ratio:.2%}\n")

print("🎉 Sparse matrix generation completed!")

