# CNN Architecture Based Image Classifier

A deep learning project that explores and compares multiple Convolutional Neural Network (CNN) architectures for image classification.  
This project evaluates how different CNN models and training strategies impact classification accuracy and model performance.

The study implements three progressively improved models and analyzes their performance using evaluation metrics and visualization techniques.

---

# Project Objective

The primary goal of this project is to build and compare different CNN architectures to understand how architectural changes, data augmentation, and transfer learning affect image classification performance.

The project includes:

- Baseline CNN architecture
- CNN with data augmentation
- Transfer learning using a pretrained CNN model

---

# Implemented Model Versions

## V1 — Advanced CNN (Baseline Model)

This model implements a custom convolutional neural network architecture built from scratch.

### Architecture components

- Convolutional layers
- Depthwise Separable Convolutions
- Batch Normalization
- Max Pooling
- Dropout
- Global Average Pooling
- Fully connected layers

### Purpose

- Establish a strong baseline CNN model
- Measure performance without augmentation or pretrained features

---

## V2 — CNN with Data Augmentation

This version enhances the baseline model by introducing real-time data augmentation techniques during training.

### Augmentation techniques

- Random Horizontal Flip
- Random Rotation
- Random Zoom
- Random Contrast Adjustment

### Purpose

- Increase dataset diversity
- Improve model generalization
- Reduce overfitting

---

## V3 — Transfer Learning (MobileNetV2)

The third model uses **Transfer Learning** with a pretrained CNN architecture.

### Pretrained model

- MobileNetV2

### Approach

1. Load MobileNetV2 without the top classification layer
2. Freeze pretrained layers
3. Add custom classification layers
4. Train on the dataset

### Purpose

- Leverage pretrained feature extraction
- Achieve higher accuracy
- Reduce training time

---

# Experimental Pipeline

The overall workflow followed in this project:

1. Dataset preparation
2. Image preprocessing
3. Model training
4. Model evaluation
5. Performance comparison
6. Visualization of results

---

# Dataset

The dataset used in this project consists of labeled images belonging to two classes.

### Dataset split

- Training set — 80%
- Validation set — 20%

Due to size limitations, the dataset is not included in this repository.

---

# Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---


---

# Training Results

| Model | Description | Validation Accuracy |
|------|-------------|--------------------|
| V1 | Advanced CNN Baseline | 88% |
| V2 | CNN with Data Augmentation | 78% |
| V3 | Transfer Learning (MobileNetV2) | 97% |

---

# Evaluation Metrics

The models are evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Evaluation reports for each model are stored in the **experiments** directory.

---

# Visualizations

The project generates several visualizations to analyze model performance:

- Training Accuracy vs Epochs
- Training Loss vs Epochs
- Confusion Matrix
- Model Accuracy Comparison
- Error Rate Comparison
- Model Complexity vs Accuracy
- Performance Gain Graphs

All visualizations are saved inside the **experiments** directory.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Skarthik06/cnn-architecture-image-classifier.git
cd cnn-architecture-image-classifier

Install dependencies:

pip install -r requirements.txt
Training the Models

Train baseline model

python src/train.py

Train augmented model

python src/train_v2.py

Train transfer learning model

python src/train_v3.py
Evaluating Models
python src/evaluate.py
python src/evaluate_v2.py
python src/evaluate_v3.py
Running Predictions

To perform prediction on a new image:

python src/predict.py
Key Insights

CNN architectures provide strong performance for image classification tasks.

Data augmentation improves generalization but may not always increase accuracy.

Transfer learning significantly boosts performance using pretrained models.

MobileNetV2 achieved the best accuracy with efficient feature extraction.

Future Improvements

Potential future enhancements include:

Adding additional architectures such as ResNet or EfficientNet

Hyperparameter tuning

Implementing model deployment using Flask or FastAPI

Supporting multi-class classification tasks
