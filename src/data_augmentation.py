import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image

# Define paths
data_dir = 'C:\\Users\\aishw\\PycharmProjects\\pythonProject\\data'
metadata_file = os.path.join(data_dir, 'HAM10000_metadata_updated.csv')
images_dir = os.path.join(data_dir, 'merged')
augmented_data_dir = os.path.join(data_dir, 'augmented')

# Create augmented data directory if it doesn't exist
if not os.path.exists(augmented_data_dir):
    os.makedirs(augmented_data_dir)

# Load the metadata
metadata = pd.read_csv(metadata_file)

# Get image IDs
image_ids = metadata['image_id'].tolist()


# Augmentation function
def augment_data(img_path, augmented_data_dir):
    # Load the image
    img = Image.open(img_path)
    img_array = tf.convert_to_tensor(np.array(img))

    # Perform augmentations
    # Rotate the image by a random angle (0, 90, 180, 270 degrees)
    angles = [0, 90, 180, 270]
    angle = np.random.choice(angles)
    img_array = tf.image.rot90(img_array, k=angle // 90)

    # Randomly flip the image horizontally
    if np.random.rand() > 0.5:
        img_array = tf.image.random_flip_left_right(img_array)

    # Randomly adjust brightness
    img_array = tf.image.random_brightness(img_array, max_delta=0.1)

    # Randomly adjust contrast
    img_array = tf.image.random_contrast(img_array, lower=0.8, upper=1.2)

    # Convert back to PIL Image and save
    augmented_image = Image.fromarray(img_array.numpy())
    augmented_image.save(os.path.join(augmented_data_dir, os.path.basename(img_path)))


# Augment images
print("Augmenting images...")
for img_id in tqdm(image_ids):
    img_path = os.path.join(images_dir, img_id)
    augment_data(img_path, augmented_data_dir)

print("Data augmentation completed.")
