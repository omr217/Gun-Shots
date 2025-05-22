# 🔊 Spectrogram Classification with CNN (TensorFlow)

This project trains a **Convolutional Neural Network (CNN)** on **spectrogram images** labeled via XML files. The goal is to classify environmental sounds into various categories such as Aircraft, Traffic, Sirens, etc.

---

## 📦 Features

- Loads spectrogram images and XML-based labels.
- Automatically handles directory structure per class.
- Uses a CNN with data augmentation for classification.
- Tracks accuracy and plots training history.
- Uses one-hot encoded categorical labels.

---

## 🗃 Dataset Structure

The dataset consists of two folders:
.
├── Spectrograms/
│ ├── Aircraft/
│ ├── Traffic/
│ └── ...
├── Labels/
│ ├── Aircraft/
│ ├── Traffic/
│ └── ...


- Each `.xml` file must contain a `<label>` tag matching the directory name.
- Each spectrogram must have a corresponding XML file with the same filename (e.g., `sound1.png` ↔ `sound1.xml`).

---

## 🎓 Classes

There are 16 predefined classes:

```python
classes = ['Aircraft', 'Break', 'Construction', 'Crash', 'Electric', 'fire&firework',
           'Traffic', 'Wind', 'Wood', 'Weather', 'Water', 'Voices', 'Vehicles',
           'Handtool', 'Siren', 'Gunshots']
```

### 🚀 Model Architecture

Input: Spectrogram Image
↓
Conv2D (32 filters) + MaxPooling
↓
Conv2D (64 filters) + MaxPooling
↓
Conv2D (128 filters) + MaxPooling
↓
Flatten → Dense(128) + Dropout(0.5)
↓
Dense(16 - Softmax for classification)


#### 🧠 Training Workflow

**Data Loading**

  Loads images using matplotlib.image.imread().
  
  Reads class labels from XML using xml.etree.ElementTree.

**Data Preprocessing**

  One-hot encodes class labels.
  
  Splits dataset into training and testing (80/20 split).

**Model Training**

  CNN compiled with categorical_crossentropy and Adam optimizer.
  
  Data is augmented with ImageDataGenerator.

**Evaluation**

  Accuracy is printed and plotted after training.
  Final test accuracy is printed.

##### ⚠️ Notes

Ensure all spectrograms have matching XML files.

The image dimensions must be consistent across the dataset.

Adjust the CNN architecture or hyperparameters for better accuracy.




