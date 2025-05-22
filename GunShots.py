import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)


def load_data(spectrograms_folder, labels_folder, classes):
    images = []
    labels = []
    class_indices = {cls: idx for idx, cls in enumerate(classes)}

    for class_name in classes:
        image_dir = os.path.join(spectrograms_folder, class_name)
        label_dir = os.path.join(labels_folder, class_name)

        if not os.path.isdir(image_dir):
            logging.warning(f"Image directory {image_dir} does not exist.")
            continue

        if not os.path.isdir(label_dir):
            logging.warning(f"Label directory {label_dir} does not exist.")
            continue

        image_files = os.listdir(image_dir)
        logging.info(f"Found {len(image_files)} image files in class '{class_name}'.")

        for image_file in image_files:
            try:
                image_path = os.path.join(image_dir, image_file)
                logging.info(f"Reading image file {image_path}.")
                image = plt.imread(image_path)
                images.append(image)

                base_name = os.path.splitext(image_file)[0]
                xml_path = os.path.join(label_dir, f"{base_name}.xml")
                if not os.path.exists(xml_path):
                    logging.warning(f"XML file {xml_path} does not exist for image {image_path}.")
                    continue

                logging.info(f"Reading XML file {xml_path}.")
                tree = ET.parse(xml_path)
                root = tree.getroot()
                label_element = root.find('label')

                if label_element is not None:
                    label = label_element.text
                    if label == class_name:
                        labels.append(class_indices[label])
                    else:
                        logging.warning(
                            f"Label '{label}' in XML does not match directory name '{class_name}' for file {xml_path}.")
                else:
                    logging.warning(f"No 'label' element found in XML file {xml_path}.")

            except Exception as e:
                logging.error(f"Error processing file {image_path}: {e}")

    if len(images) == 0 or len(labels) == 0:
        logging.error("No images or labels loaded. Please check the file paths and data.")
        return None, None

    images = np.array(images)
    labels = to_categorical(np.array(labels), num_classes=len(classes))

    logging.info(f"Loaded {len(images)} images and {len(labels)} labels.")

    return images, labels


classes = ['Aircraft', 'Break', 'Construction', 'Crash', 'Electric', 'fire&firework',
           'Traffic', 'Wind', 'Wood', 'Weather', 'Water', 'Voices', 'Vehicles',
           'Handtool', 'Siren', 'Gunshots']

spectrograms_folder = r'C:\Users\Omer\Desktop\Spechtograms'
labels_folder = r'C:\Users\Omer\Desktop\labels'

# Check directory contents
logging.info(f"Contents of spectrograms directory: {os.listdir(spectrograms_folder)}")
logging.info(f"Contents of labels directory: {os.listdir(labels_folder)}")

images, labels = load_data(spectrograms_folder, labels_folder, classes)

if images is not None and labels is not None:
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
else:
    logging.error("Terminating due to lack of data.")

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


def create_cnn_model(input_shape, num_classes):
    model = Sequential()

    # Convolutional layer 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    # Convolutional layer 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Convolutional layer 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Flatten and Fully Connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


input_shape = X_train.shape[1:]  # Shape of the input images
num_classes = len(classes)

model = create_cnn_model(input_shape, num_classes)
model.summary()

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

datagen.fit(X_train)

history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=50)
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy * 100:.2f}%')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
