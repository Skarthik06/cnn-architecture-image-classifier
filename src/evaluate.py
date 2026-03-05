import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import load_only_validation_data

# Experiment paths
BASE_PATH = r"D:\SK\dog_cat_cnn_project\experiments\v1_baseline"
MODEL_PATH = os.path.join(BASE_PATH, "model", "v1_model.keras")
CONF_MATRIX_PATH = os.path.join(BASE_PATH, "plots", "confusion_matrix.png")
REPORT_PATH = os.path.join(BASE_PATH, "metrics", "classification_report.txt")
METRICS_PATH = os.path.join(BASE_PATH, "metrics", "metrics.txt")

VAL_DIR = r"D:\SK\dog_cat_cnn_project\dataset\validation"

def main():
    os.makedirs(os.path.join(BASE_PATH, "plots"), exist_ok=True)
    os.makedirs(os.path.join(BASE_PATH, "metrics"), exist_ok=True)

    print("📦 Loading trained V1 model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("📂 Loading validation dataset...")
    val_data = load_only_validation_data(VAL_DIR)

    y_true = []
    y_pred = []

    print("🔍 Running predictions...")
    for images, labels in val_data:
        preds = model.predict(images)
        preds = np.argmax(preds, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Cat", "Dog"],
                yticklabels=["Cat", "Dog"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix – V1 Baseline CNN")
    plt.savefig(CONF_MATRIX_PATH)
    plt.close()
    print(f"🧩 Confusion matrix saved at {CONF_MATRIX_PATH}")

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=["Cat", "Dog"])
    print("\nClassification Report:\n")
    print(report)

    with open(REPORT_PATH, "w") as f:
        f.write(report)

    # Accuracy metric
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))

    with open(METRICS_PATH, "w") as f:
        f.write(f"Validation Accuracy: {accuracy:.4f}\n")

    print(f"📄 Classification report saved at {REPORT_PATH}")
    print(f"📈 Accuracy metric saved at {METRICS_PATH}")

if __name__ == "__main__":
    main()
