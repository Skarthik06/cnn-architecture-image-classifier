import os
import shutil
import random

# Source dataset folders
SOURCE_CAT = r"D:\SK\archive\PetImages\Cat"
SOURCE_DOG = r"D:\SK\archive\PetImages\Dog"

# Destination folders
TRAIN_CAT = r"D:\SK\dog_cat_cnn_project\dataset\train\cat"
TRAIN_DOG = r"D:\SK\dog_cat_cnn_project\dataset\train\dog"
VAL_CAT = r"D:\SK\dog_cat_cnn_project\dataset\validation\cat"
VAL_DOG = r"D:\SK\dog_cat_cnn_project\dataset\validation\dog"

def split_data(source_folder, train_folder, val_folder, split_ratio=0.8):
    images = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    print(f"📂 Copying {len(train_images)} images to {train_folder}")
    for img in train_images:
        try:
            shutil.copy(os.path.join(source_folder, img), os.path.join(train_folder, img))
        except:
            pass

    print(f"📂 Copying {len(val_images)} images to {val_folder}")
    for img in val_images:
        try:
            shutil.copy(os.path.join(source_folder, img), os.path.join(val_folder, img))
        except:
            pass

if __name__ == "__main__":
    split_data(SOURCE_CAT, TRAIN_CAT, VAL_CAT)
    split_data(SOURCE_DOG, TRAIN_DOG, VAL_DOG)
    print("\n✅ Dataset successfully split into train and validation!")
