import pandas as pd
from sklearn.metrics import classification_report

# Load the predictions and metadata DataFrames
predictions_path = 'C:\\Users\\aishw\\PycharmProjects\\pythonProject\\results\\all_predictions_results_updated.csv'
metadata_path = 'C:\\Users\\aishw\\PycharmProjects\\pythonProject\\data\\HAM10000_metadata_updated.csv'

# Read the CSV files into DataFrames
predictions_df = pd.read_csv(predictions_path)
metadata_df = pd.read_csv(metadata_path)

# Merge predictions with metadata on 'image_id'
merged_df = predictions_df.merge(metadata_df, on='image_id', how='left')

# Get true and predicted labels
y_true = merged_df['dx'].values  # True labels from metadata
y_pred = merged_df['predicted_label'].values  # Predicted labels from model

# Generate classification report, handling zero division
report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

# Save report to a text file
report_file_path = 'C:\\Users\\aishw\\PycharmProjects\\pythonProject\\results\\classification_report.txt'
with open(report_file_path, 'w') as f:
    f.write(classification_report(y_true, y_pred, zero_division=0))

print(f"Classification report saved to: {report_file_path}")
