import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

DENSE_DIR = r"D:\SK\dog_cat_cnn_project\datasets"
SPARSE_DIR = r"D:\SK\dog_cat_cnn_project\datasets\sparse"
SAVE_DIR = r"D:\SK\dog_cat_cnn_project\experiments\final_comparisons"

os.makedirs(SAVE_DIR, exist_ok=True)

models = {
    "V1": ("V1_Advanced_CNN_features.csv", "V1_Advanced_CNN_features_SPARSE.csv"),
    "V2": ("V2_Augmented_CNN_features.csv", "V2_Augmented_CNN_features_SPARSE.csv"),
    "V3": ("V3_Transfer_Learning_features.csv", "V3_Transfer_Learning_features_SPARSE.csv")
}

dense_nonzero = []
sparse_nonzero = []
labels = []

for model, (dense_file, sparse_file) in models.items():
    dense = pd.read_csv(os.path.join(DENSE_DIR, dense_file))
    sparse = pd.read_csv(os.path.join(SPARSE_DIR, sparse_file))

    dense_values = dense.drop(columns=["label", "model_version"]).values
    sparse_values = sparse.drop(columns=["label", "model_version"]).values

    dense_nonzero.append(np.count_nonzero(dense_values))
    sparse_nonzero.append(np.count_nonzero(sparse_values))
    labels.append(model)

plt.figure()
plt.bar(labels, dense_nonzero)
plt.bar(labels, sparse_nonzero, bottom=dense_nonzero)
plt.xlabel("Model Version")
plt.ylabel("Non-zero Feature Values")
plt.title("Dense vs Sparse Feature Representation")
plt.legend(["Dense Features", "Sparse Features"])
plt.savefig(os.path.join(SAVE_DIR, "dense_vs_sparse.png"))
plt.close()

print("✅ Dense vs Sparse comparison graph saved")
