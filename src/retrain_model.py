import os
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils import to_categorical

# Load metadata
csv_path = "C:\\Users\\aishw\\PycharmProjects\\pythonProject\\data\\HAM10000_metadata_updated.csv"
metadata = pd.read_csv(csv_path)

# Define class mapping (make sure all classes are included)
class_mapping = {
    'bkl': 0,
    'mel': 1,
    'nv': 2,
    'akiec': 3,
    'bcc': 4,
    'df': 5,
    'vasc': 6  # Include 'vasc' class here
}

# Print unique labels in the dataset
print("Unique Labels in Dataset:", metadata['dx'].unique())


# Prepare image data and labels
def load_data():
    images = []
    labels = []
    for index, row in metadata.iterrows():
        image_id = row['image_id']
        label = row['dx']

        # Check if the label is in class mapping
        if label not in class_mapping:
            print(f"Warning: Label '{label}' not found in class mapping. Skipping.")
            continue

        label_index = class_mapping[label]
        image_path = os.path.join("C:\\Users\\aishw\\PycharmProjects\\pythonProject\\data\\merged", image_id)

        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Adjust as needed
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(label_index)
    return np.array(images), np.array(labels)


# Load the dataset
X, y = load_data()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=len(class_mapping))
y_val = to_categorical(y_val, num_classes=len(class_mapping))

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Build a simple CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(class_mapping), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(datagen.flow(X_train, y_train, batch_size=32),
          validation_data=(X_val, y_val),
          epochs=50,  # Adjust as needed
          steps_per_epoch=len(X_train) // 32)

# Save the model
model_path = "C:\\Users\\aishw\\PycharmProjects\\pythonProject\\models\\skin_cancer_model_retrained.h5"
model.save(model_path)

print("Model retrained and saved to", model_path)
