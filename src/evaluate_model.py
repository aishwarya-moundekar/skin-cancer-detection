import os
import pandas as pd
import cv2
import numpy as np
from keras.models import load_model

# Load the model
model_path = "C:\\Users\\aishw\\PycharmProjects\\pythonProject\\models\\skin_cancer_model.h5"
model = load_model(model_path)

# Load the metadata
csv_path = "C:\\Users\\aishw\\PycharmProjects\\pythonProject\\data\\HAM10000_metadata_updated.csv"
metadata = pd.read_csv(csv_path)

# Define class mapping (update with your actual classes)
class_mapping = {
    'bkl': 0,
    'mel': 1,
    'nv': 2,
    'akiec': 3,
    'bcc': 4,
    'df': 5,
    'vasc': 6  # Add this if it's in your dataset
}

# Preprocess function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Unable to read image at {image_path}. Skipping.")
        return None
    img = cv2.resize(img, (128, 128))  # Adjust size as per your model
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Lists to store results
results = []

# Iterate over the rows in the metadata DataFrame
for index, row in metadata.iterrows():
    image_id = row['image_id']
    true_label = row['dx']
    image_path = os.path.join("C:\\Users\\aishw\\PycharmProjects\\pythonProject\\data\\merged", image_id)

    processed_image = preprocess_image(image_path)

    if processed_image is not None:
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # Map the predicted class index back to label
        predicted_label = list(class_mapping.keys())[predicted_class]

        # Store results
        results.append({
            'image_id': image_id,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'confidence': confidence
        })

# Create a DataFrame for results
results_df = pd.DataFrame(results)

# Ensure results directory exists
results_dir = "C:\\Users\\aishw\\PycharmProjects\\pythonProject\\results"
os.makedirs(results_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Save results to CSV
results_df.to_csv(os.path.join(results_dir, "evaluation_results.csv"), index=False)

print("Evaluation completed. Results saved to evaluation_results.csv.")
