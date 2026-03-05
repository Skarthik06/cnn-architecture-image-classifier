import os
import matplotlib.pyplot as plt
from data_loader import load_train_val_data
from models import build_augmented_cnn

TRAIN_DIR = r"D:\SK\dog_cat_cnn_project\dataset\train"
VAL_DIR   = r"D:\SK\dog_cat_cnn_project\dataset\validation"

SAVE_BASE = r"D:\SK\dog_cat_cnn_project\experiments\v2_augmentation"

def main():
    print("📂 Loading datasets...")
    train_data, val_data = load_train_val_data(TRAIN_DIR, VAL_DIR)

    print("🧠 Building V2 Augmented CNN model...")
    model = build_augmented_cnn()
    model.summary()

    print("🚀 Training V2 model with Data Augmentation...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10
    )

    # ✅ Save model
    model.save(os.path.join(SAVE_BASE, "model", "v2_model.keras"))
    print("✅ V2 model saved!")

    # 📊 Save training graphs
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("V2 Training vs Validation Accuracy")
    plt.savefig(os.path.join(SAVE_BASE, "plots", "training_accuracy.png"))
    plt.close()
    print("📊 Training graph saved!")

    # 📝 Save experiment notes
    with open(os.path.join(SAVE_BASE, "notes", "notes.txt"), "w") as f:
        f.write("EXPERIMENT VERSION: V2 CNN with Data Augmentation\n")
        f.write("Added Layers: RandomFlip, RandomRotation, RandomZoom, RandomContrast\n")
        f.write("Image Size: 128x128\n")
        f.write("Epochs: 10\n")
        f.write("Optimizer: Adam\n")
        f.write("Purpose: Improve generalization and reduce overfitting\n")

    print("📝 V2 experiment notes saved!")

if __name__ == "__main__":
    main()
