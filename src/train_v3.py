import os
import matplotlib.pyplot as plt
import tensorflow as tf
from data_loader import load_train_val_data
from models import build_transfer_model

TRAIN_DIR = r"D:\SK\dog_cat_cnn_project\dataset\train"
VAL_DIR   = r"D:\SK\dog_cat_cnn_project\dataset\validation"

SAVE_BASE = r"D:\SK\dog_cat_cnn_project\experiments\v3_transfer_learning"

def main():

    print("📂 Loading datasets...")
    train_data, val_data = load_train_val_data(TRAIN_DIR, VAL_DIR)

    print("🧠 Building V3 Transfer Learning Model (MobileNetV2)...")
    model = build_transfer_model()
    model.summary()

    print("🚀 Training V3 model (Feature Extraction Phase)...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10
    )

    # ✅ Save trained model
    model.save(os.path.join(SAVE_BASE, "model", "v3_model.keras"))
    print("✅ V3 model saved!")

    # 📊 Save accuracy graph
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("V3 Training vs Validation Accuracy")
    plt.savefig(os.path.join(SAVE_BASE, "plots", "training_accuracy.png"))
    plt.close()
    print("📊 Accuracy graph saved!")

    # 📊 Save loss graph
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("V3 Training vs Validation Loss")
    plt.savefig(os.path.join(SAVE_BASE, "plots", "training_loss.png"))
    plt.close()
    print("📉 Loss graph saved!")

    # 📝 Save experiment notes (IMPORTANT FOR VIVA)
    with open(os.path.join(SAVE_BASE, "notes", "notes.txt"), "w") as f:
        f.write("EXPERIMENT VERSION: V3 Transfer Learning (MobileNetV2)\n")
        f.write("Base Model: MobileNetV2 pretrained on ImageNet\n")
        f.write("Trainable Layers: Top classifier only (feature extraction phase)\n")
        f.write("Image Size: 128x128\n")
        f.write("Epochs: 10\n")
        f.write("Optimizer: Adam (lr=0.0001)\n")
        f.write("Purpose: Improve accuracy using pretrained deep features\n")

    print("📝 Experiment notes saved!")

    print("\n🎉 V3 Training Complete!")

if __name__ == "__main__":
    main()
