from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense,
    BatchNormalization, Dropout,
    GlobalAveragePooling2D, SeparableConv2D,
    Input, RandomFlip, RandomRotation,
    RandomZoom, RandomContrast, Rescaling
)
from tensorflow.keras.applications import MobileNetV2
import tensorflow as tf

IMG_SIZE = 128

# ==============================
# ✅ V1 MODEL — BASELINE CNN
# ==============================
def build_advanced_cnn():
    model = Sequential([

        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        Rescaling(1./255),

        Conv2D(32, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        SeparableConv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        SeparableConv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        SeparableConv2D(256, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# =========================================
# 🚀 V2 MODEL — CNN + DATA AUGMENTATION
# =========================================
def build_augmented_cnn():
    data_augmentation = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1),
        RandomContrast(0.1),
    ])

    model = Sequential([

        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        data_augmentation,
        Rescaling(1./255),

        Conv2D(32, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        SeparableConv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        SeparableConv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        SeparableConv2D(256, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ===============================================
# ⭐ V3 MODEL — TRANSFER LEARNING (MobileNetV2)
# ===============================================
def build_transfer_model():

    # Pretrained feature extractor
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze pretrained layers initially
    base_model.trainable = False

    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        Rescaling(1./255),

        base_model,
        GlobalAveragePooling2D(),

        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),

        Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
