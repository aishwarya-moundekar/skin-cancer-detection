import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# Load the model
model = tf.keras.models.load_model('C:\\Users\\aishw\\PycharmProjects\\pythonProject\\models\\skin_cancer_model.h5')

# Load the metadata CSV
csv_path = 'C:\\Users\\aishw\\PycharmProjects\\pythonProject\\data\\HAM10000_metadata_updated.csv'
df = pd.read_csv(csv_path)

# Set up ImageDataGenerator with rescaling for the test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Define image directory and use flow_from_dataframe
image_directory = 'C:\\Users\\aishw\\PycharmProjects\\pythonProject\\data\\merged'
test_data = test_datagen.flow_from_dataframe(
    dataframe=df,
    directory=image_directory,
    x_col='image_id',  # Column in CSV with image file names
    y_col='dx',        # Column in CSV with class labels
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
