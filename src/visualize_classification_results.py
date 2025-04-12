import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load the predictions CSV
predictions_path = 'C:\\Users\\aishw\\PycharmProjects\\pythonProject\\results\\all_predictions_results_updated.csv'
metadata_path = 'C:\\Users\\aishw\\PycharmProjects\\pythonProject\\data\\HAM10000_metadata_updated.csv'

# Read the predictions
predictions_df = pd.read_csv(predictions_path)
metadata_df = pd.read_csv(metadata_path)

# Merge predictions with metadata
merged_df = pd.merge(predictions_df, metadata_df, on='image_id', how='left')

# Prepare true and predicted labels
y_true = merged_df['dx'].values  # Assuming 'dx' is the true label column in metadata
y_pred = merged_df['predicted_label'].values

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred, labels=merged_df['dx'].unique())

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=merged_df['dx'].unique(), yticklabels=merged_df['dx'].unique())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('C:\\Users\\aishw\\PycharmProjects\\pythonProject\\results\\confusion_matrix.png')  # Save the figure
plt.show()

# Generate and print classification report
report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Save classification report as a CSV
report_df.to_csv('C:\\Users\\aishw\\PycharmProjects\\pythonProject\\results\\classification_report_visualization.csv')
print("Confusion matrix and classification report saved.")
