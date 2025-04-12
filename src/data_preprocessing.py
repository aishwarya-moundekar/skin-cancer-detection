import pandas as pd
import os
from PIL import Image
from sklearn.impute import SimpleImputer

# Define paths
data_dir = 'C:\\Users\\aishw\\PycharmProjects\\pythonProject\\data'
metadata_file = os.path.join(data_dir, 'HAM10000_metadata_updated.csv')
images_dir = os.path.join(data_dir, 'merged')

# Load the metadata
metadata = pd.read_csv(metadata_file)

# Check for missing values in the metadata
missing_values = metadata.isnull().sum()
print("Missing values in metadata:")
print(missing_values[missing_values > 0])

# Impute missing values in the 'age' column
age_imputer = SimpleImputer(strategy='mean')  # or use 'median', 'most_frequent'
metadata['age'] = age_imputer.fit_transform(metadata[['age']])
print("Missing values in 'age' have been imputed.")

# Check for duplicate image IDs
duplicates = metadata['image_id'].duplicated().sum()
if duplicates > 0:
    print(f"Duplicate image IDs found: {duplicates}")
else:
    print("No duplicate image IDs found.")

# Load and check images
def check_images(images_dir, image_ids):
    missing_images = []
    for img_id in image_ids:
        img_path = os.path.join(images_dir, img_id)
        if not os.path.exists(img_path):
            missing_images.append(img_id)
    return missing_images

# Get image IDs from the metadata
image_ids = metadata['image_id'].tolist()
missing_images = check_images(images_dir, image_ids)

if missing_images:
    print("Missing images:")
    print(missing_images)
else:
    print("All images are present.")

# Optional: Display some basic info about the dataset
print("\nBasic info about the dataset:")
print(metadata.info())
