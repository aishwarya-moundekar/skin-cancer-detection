import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the metadata
csv_path = r'C:\Users\aishw\PycharmProjects\pythonProject\data\HAM10000_metadata_updated.csv'
metadata = pd.read_csv(csv_path)

# Filter the metadata for cancerous images only (you may want to adjust this)
cancerous_labels = ['mel', 'bkl', 'vasc', 'akiec']  # Adjust based on your dataset
metadata = metadata[metadata['dx'].isin(cancerous_labels)]

# Define image paths
image_dir = r'C:\Users\aishw\PycharmProjects\pythonProject\data\merged'
metadata['image_path'] = metadata['image_id'].apply(lambda x: os.path.join(image_dir, x))

# Load images and labels
def load_images_and_labels(metadata):
    images = []
    labels = []
    for _, row in metadata.iterrows():
        img = tf.keras.utils.load_img(row['image_path'], target_size=(128, 128))  # Adjust size as necessary
        img_array = tf.keras.utils.img_to_array(img) / 255.0  # Normalize the images
        images.append(img_array)
        labels.append(row['dx'])
    return np.array(images), np.array(labels)

X, y = load_images_and_labels(metadata)

# Encode labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Define the model
def create_model():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(np.unique(y_encoded)), activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create the model
model = create_model()

# Set up callbacks
model_save_path = r'C:\Users\aishw\PycharmProjects\pythonProject\models\skin_cancer_model.h5'
checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=50,
    steps_per_epoch=len(X_train) // 32,
    validation_steps=len(X_val) // 32,
    class_weight=class_weights_dict,
    callbacks=[checkpoint, reduce_lr, early_stopping]
)

# Optional: Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

# Optional: Save training history for analysis
history_df = pd.DataFrame(history.history)
history_df.to_csv(r'C:\Users\aishw\PycharmProjects\pythonProject\training_history.csv', index=False)

print("Training complete and model saved.")
