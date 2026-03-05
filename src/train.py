import os
import matplotlib.pyplot as plt
from data_loader import load_train_val_data
from models import build_advanced_cnn

TRAIN_DIR = r"D:\SK\dog_cat_cnn_project\dataset\train"
VAL_DIR   = r"D:\SK\dog_cat_cnn_project\dataset\validation"

# Experiment paths
BASE_PATH = r"D:\SK\dog_cat_cnn_project\experiments\v1_baseline"
MODEL_PATH = os.path.join(BASE_PATH, "model", "v1_model.keras")
PLOT_PATH  = os.path.join(BASE_PATH, "plots", "training_accuracy.png")
NOTES_PATH = os.path.join(BASE_PATH, "notes.txt")

def main():
    # Create folders if they don't exist
    os.makedirs(os.path.join(BASE_PATH, "model"), exist_ok=True)
    os.makedirs(os.path.join(BASE_PATH, "plots"), exist_ok=True)
    os.makedirs(os.path.join(BASE_PATH, "metrics"), exist_ok=True)

    print("📂 Loading datasets...")
    train_data, val_data = load_train_val_data(TRAIN_DIR, VAL_DIR)

    print("🧠 Building V1 Advanced CNN model...")
    model = build_advanced_cnn()
    model.summary()

    print("🚀 Training started...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10
    )

    # Save model
    model.save(MODEL_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")

    # Save training graph
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("V1 Training vs Validation Accuracy")
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"📊 Training graph saved at {PLOT_PATH}")

    # Save experiment notes (VERY IMPORTANT FOR VIVA)
    with open(NOTES_PATH, "w") as f:
        f.write("EXPERIMENT VERSION: V1 BASELINE CNN\n")
        f.write("Model: Custom Advanced CNN with SeparableConv + BatchNorm\n")
        f.write("Image Size: 128x128\n")
        f.write("Batch Size: 32\n")
        f.write("Epochs: 10\n")
        f.write("Optimizer: Adam\n")
        f.write("Loss: SparseCategoricalCrossentropy\n")
        f.write("Purpose: Establish baseline performance before augmentation or transfer learning.\n")

    print("📝 Experiment notes saved!")

if __name__ == "__main__":
    main()
