import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Define file paths
predictions_path = r'C:\Users\aishw\PycharmProjects\pythonProject\results\all_predictions_results_updated.csv'
metadata_path = r'C:\Users\aishw\PycharmProjects\pythonProject\data\HAM10000_metadata_updated.csv'

# Load predictions and metadata
predictions_df = pd.read_csv(predictions_path)
metadata_df = pd.read_csv(metadata_path)

# Print DataFrames for debugging
print("Predictions DataFrame:")
print(predictions_df.head())
print("\nMetadata DataFrame:")
print(metadata_df.head())

# Ensure all image_ids are in the same format (e.g., lower case)
predictions_df['image_id'] = predictions_df['image_id'].str.lower()
metadata_df['image_id'] = metadata_df['image_id'].str.lower()

# Check for unique image_ids in both DataFrames
unique_predictions = predictions_df['image_id'].unique()
unique_metadata = metadata_df['image_id'].unique()
print("\nUnique image_ids in Predictions:", unique_predictions)
print("Unique image_ids in Metadata:", unique_metadata)

# Merge predictions with metadata
merged_df = predictions_df.merge(metadata_df, on='image_id', how='left')
print("\nMerged DataFrame shape:", merged_df.shape)
print("Merged DataFrame:")
print(merged_df.head())

# Check for missing true labels
missing_true_labels = merged_df['true_label'].isnull().sum()
print(f"\nNumber of NaN true labels: {merged_df['true_label'].isnull().value_counts()}")

# Collect true labels and predicted labels
y_true = merged_df['dx'].fillna('unknown').to_numpy()  # Fill NaN with a placeholder if necessary
y_pred = merged_df['predicted_label'].to_numpy()

# Print unique values for analysis
print("\nUnique true labels:", np.unique(y_true))
print("Unique predicted labels:", np.unique(y_pred))
print("Type of true labels:", type(y_true[0]))
print("Type of predicted labels:", type(y_pred[0]))

# Generate classification report if valid labels exist
if np.any(y_true != 'unknown') and np.any(y_pred != ''):
    report = classification_report(y_true, y_pred)
    print("\nClassification Report:")
    print(report)
else:
    print("\nNo valid true labels or predicted labels to analyze.")
