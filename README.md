# 🧠 MNIST CNN Classifier with Data Augmentation

A robust Convolutional Neural Network (CNN) model trained on the classic MNIST dataset with data augmentation for enhanced generalization. Built with TensorFlow and Keras.

---

## 🚀 Features

- 🔢 Classifies handwritten digits (0-9) using a CNN
- ↻ Enhanced with **Image Data Augmentation** (rotation, shifting, zooming)
- 📊 Evaluated with precision, recall, F1-score, and confusion matrix
- 📂 Easily save and reload your trained model
- 🧪 Includes test set evaluation and predictions

---

## 🧰 Tech Stack

- Python 🐍
- TensorFlow / Keras
- NumPy
- scikit-learn (for evaluation metrics)
- Matplotlib (optional: for visualizations)

---

## 🖼️ Data Augmentation Techniques

- ↻ Rotation: ±15°
- ↔️ Horizontal/Vertical Shift: 10%
- 🔍 Zoom: ±10%
- ❌ No Flipping (digits are orientation-sensitive)

---

## 🏗️ Model Architecture

```plaintext
Input (28x28x1)
│
├── Conv2D (32 filters, 3x3, ReLU)
├── MaxPooling2D (2x2)
├── Conv2D (64 filters, 3x3, ReLU)
├── MaxPooling2D (2x2)
├── Flatten
├── Dense (128, ReLU)
└── Dense (10, Softmax)
```

---

## 📊 Performance Metrics

After training for 10 epochs:

- ✅ Accuracy: ~99%
- 📋 Precision, Recall, F1-score: Available via `classification_report()`
- 🔍 Confusion Matrix: Shows per-digit prediction accuracy

---

## 📦 Getting Started

```bash
git clone https://github.com/yourusername/mnist-cnn-augmented.git
cd mnist-cnn-augmented
pip install -r requirements.txt
```

Run the model training:
```bash
python mnist_cnn_augmented.py
```

---

## 📊 Evaluation

To evaluate and generate a classification report:

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

## 📂 Saving and Loading the Model

```python
# Save
model.save("mnist_cnn_augmented_model.h5")

# Load
from tensorflow.keras.models import load_model
model = load_model("mnist_cnn_augmented_model.h5")
```

---

## 👑 Author

**S.Arun Prakash**  
📧 [arunprak3@gmail.com](mailto:arunprak3@gmail.com)

---

## 🧠 Fun Fact

> The MNIST dataset was used to test early neural networks — and it's still a favorite for testing deep learning prototypes today!

