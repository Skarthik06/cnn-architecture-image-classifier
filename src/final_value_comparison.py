import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import entropy

BASE_DIR = r"D:\SK\dog_cat_cnn_project\datasets"
SPARSE_DIR = os.path.join(BASE_DIR, "sparse")
PLOT_DIR = os.path.join(BASE_DIR, "final_plots")

os.makedirs(PLOT_DIR, exist_ok=True)

MODELS = {
    "V1": "V1_Advanced_CNN",
    "V2": "V2_Augmented_CNN",
    "V3": "V3_Transfer_Learning"
}

def load_features(path):
    df = pd.read_csv(path)
    feature_cols = df.columns[:-2]  # drop label & model_version
    values = df[feature_cols].values.flatten()
    values = values[np.isfinite(values)]  # remove NaN / inf
    return values

def signed_log_transform(x):
    return np.sign(x) * np.log1p(np.abs(x))

# ============================
# 1️⃣ Log-Scaled Distribution
# ============================
plt.figure(figsize=(10,6))
for v, name in MODELS.items():
    dense = signed_log_transform(load_features(
        os.path.join(BASE_DIR, f"{name}_features.csv")
    ))
    sparse = signed_log_transform(load_features(
        os.path.join(SPARSE_DIR, f"{name}_features_SPARSE.csv")
    ))

    plt.hist(dense, bins=200, alpha=0.4, label=f"{v} Dense", density=True)
    plt.hist(sparse, bins=200, alpha=0.4, label=f"{v} Sparse", density=True)

plt.legend()
plt.title("Signed Log Feature Distribution (Dense vs Sparse)")
plt.xlabel("Signed log(|activation| + 1)")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "log_distribution.png"), dpi=300)
plt.close()

# ============================
# 2️⃣ Cumulative Energy Curve
# ============================
plt.figure(figsize=(10,6))
for v, name in MODELS.items():
    dense = np.abs(load_features(
        os.path.join(BASE_DIR, f"{name}_features.csv")
    ))
    dense = np.sort(dense)[::-1]
    energy = np.cumsum(dense) / np.sum(dense)
    plt.plot(energy, label=v)

plt.legend()
plt.title("Cumulative Feature Energy Curve")
plt.xlabel("Feature Index")
plt.ylabel("Cumulative Energy")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "cumulative_energy.png"), dpi=300)
plt.close()

# ============================
# 3️⃣ Mean Activation
# ============================
means = []
for v, name in MODELS.items():
    dense = load_features(os.path.join(BASE_DIR, f"{name}_features.csv"))
    means.append(np.mean(np.abs(dense)))

plt.figure()
plt.bar(MODELS.keys(), means)
plt.title("Mean Absolute Feature Activation")
plt.ylabel("Mean |activation|")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "mean_activation.png"), dpi=300)
plt.close()

# ============================
# 4️⃣ Std Deviation
# ============================
stds = []
for v, name in MODELS.items():
    dense = load_features(os.path.join(BASE_DIR, f"{name}_features.csv"))
    stds.append(np.std(dense))

plt.figure()
plt.bar(MODELS.keys(), stds)
plt.title("Feature Activation Standard Deviation")
plt.ylabel("Std Deviation")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "std_activation.png"), dpi=300)
plt.close()

# ============================
# 5️⃣ Entropy
# ============================
entropies = []
for v, name in MODELS.items():
    dense = load_features(os.path.join(BASE_DIR, f"{name}_features.csv"))
    hist, _ = np.histogram(dense, bins=256, density=True)
    entropies.append(entropy(hist + 1e-10))

plt.figure()
plt.bar(MODELS.keys(), entropies)
plt.title("Feature Entropy Comparison")
plt.ylabel("Entropy")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "entropy_comparison.png"), dpi=300)
plt.close()

print("✅ FINAL VALUE-BASED COMPARISON GRAPHS CREATED SUCCESSFULLY")
print(f"📂 Saved in: {PLOT_DIR}")
