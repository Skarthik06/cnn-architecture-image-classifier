import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from data_loader import load_only_validation_data
import os

MODEL_PATH = r"D:\SK\dog_cat_cnn_project\experiments\v3_transfer_learning\model\v3_model.keras"
VAL_DIR    = r"D:\SK\dog_cat_cnn_project\dataset\validation"
SAVE_BASE  = r"D:\SK\dog_cat_cnn_project\experiments\v3_transfer_learning"

print("📦 Loading V3 Transfer Learning model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("📂 Loading validation data...")
val_data = load_only_validation_data(VAL_DIR)

y_true, y_pred = [], []

print("🔍 Running predictions on validation set...")
for images, labels in val_data:
    preds = model.predict(images)
    preds = np.argmax(preds, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(preds)

# =============================
# 🧩 Confusion Matrix
# =============================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Cat", "Dog"],
            yticklabels=["Cat", "Dog"])
plt.title("Confusion Matrix – V3 Transfer Learning")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(SAVE_BASE, "plots", "confusion_matrix.png"))
plt.close()
print("🧩 Confusion matrix saved!")

# =============================
# 📄 Classification Report
# =============================
report = classification_report(y_true, y_pred, target_names=["Cat", "Dog"])

with open(os.path.join(SAVE_BASE, "metrics", "classification_report.txt"), "w") as f:
    f.write("V3 TRANSFER LEARNING CLASSIFICATION REPORT\n\n")
    f.write(report)

print("📄 Classification report saved!")

# =============================
# 📈 Accuracy Metric
# =============================
accuracy = np.mean(np.array(y_true) == np.array(y_pred))

with open(os.path.join(SAVE_BASE, "metrics", "metrics.txt"), "w") as f:
    f.write("V3 TRANSFER LEARNING METRICS\n\n")
    f.write(f"Validation Accuracy: {accuracy:.4f}\n")

print("📈 Metrics saved!")

print("\n✅ V3 Evaluation Complete!")
