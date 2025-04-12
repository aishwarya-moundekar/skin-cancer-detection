
# **Skin Cancer Detection Using Deep Learning**

## **Overview**

This project focuses on developing a deep learning-based system for skin cancer detection by classifying skin lesions as cancerous (malignant) and its type or non-cancerous (benign). The system is trained on images from the **HAM10000 dataset**, which have been manually merged all images into a separate folder named merged, and utilizes an updated custom metadata csv file. 

A **Convolutional Neural Network (CNN)** is employed to classify the lesions, and the system includes a user-friendly **Graphical User Interface (GUI)** for easy prediction. The application allows users to upload images and receive predictions on whether the lesions are cancerous or not.

---

## **Project Structure**

```
/skin_cancer_detection_project
│
├── /data/
│   ├── /merged/                           # Folder containing merged images for training and prediction
│   └── HAM10000_metadata_updated.csv      # Custom CSV with updated metadata for the project
│
├── /models/
│   ├── skin_cancer_model.h5              # Trained model
│   └── skin_cancer_model_retrained.h5    # Retrained model (optional)
│
├── /results/
│   ├── all_predictions_results_updated.csv # Prediction results file
│   └── classification_report.txt          # Classification report file
│
├── /src/
│   ├── data_preprocessing.py             # Script for data preprocessing
│   ├── train_model.py                   # Script to train the deep learning model
│   ├── analyze_predictions.py           # Script to analyze prediction results
│   ├── save_classification_report.py    # Script to save the classification report
│   ├── visualize_classification_results.py # Script to visualize results
│   └── gui_application_final.py         # Script for the GUI application
│
└── requirements.txt                     # Required Python packages
```

---

## **Dataset**

This project uses a modified version of the **HAM10000 dataset**. The images have been manually merged into the **`/data/merged/`** folder, containing 10,051 skin lesion images. The metadata for the dataset has also been customized and is provided in **`HAM10000_metadata_updated.csv`**, which is located in the **`/data/`** folder.

The updated metadata includes:
- **lesion_id**: Unique identifier for each lesion
- **image_id**: Unique identifier for each image (with `.jpg` extension)
- **dx**: Diagnosis label indicating the type of lesion (e.g., `mel` for melanoma, `bkl` for benign keratosis)
- **dx_type**: Source of the diagnosis (e.g., `manual`, `verified`)
- **age**: Age of the patient
- **sex**: Sex of the patient (e.g., `male`, `female`)
- **localization**: Anatomical location of the lesion (e.g., `upper back`, `leg`)

### **Custom Dataset**

- **Images**: All images are located in the **`/data/merged/`** folder and are used for both training and predictions.
- **Metadata**: The custom **`HAM10000_metadata_updated.csv`** file is used in place of the original dataset metadata to ensure better and more relevant information.

---

## **Model Overview**

The skin cancer detection system utilizes a **Convolutional Neural Network (CNN)** for classifying skin lesions based on their visual characteristics. The model is trained using the processed images from the **`/data/merged/`** folder, along with the updated metadata.

### **Key Features:**
1. **Data Preprocessing**: Images are resized and normalized to maintain consistency for training.
2. **CNN Architecture**: A Convolutional Neural Network is used to train the model for skin lesion classification.
3. **Model Evaluation**: The model is evaluated using metrics such as accuracy, precision, recall, and F1-score.
4. **Retraining**: The system supports retraining the model using new data or hyperparameter adjustments.

---

## **Requirements**

Before running the project, ensure you have the necessary dependencies installed. You can install them by using the **`requirements.txt`** file:

```bash
pip install -r requirements.txt
```

---

## **How to Run the Project**

### **1. Set Up the Dataset**

Ensure that all images are stored in the **`/data/merged/`** folder, and the **`HAM10000_metadata_updated.csv`** file is in the **`/data/`** folder.

### **2. Train the Model**

To train the model, execute the following script:

```bash
python src/train_model.py
```

This will:
- Load the images from the **`/data/merged/`** folder.
- Preprocess the images.
- Train the CNN model and save it as **`skin_cancer_model.h5`**.

### **3. Retrain the Model (Optional)**

If you want to retrain the model with new data or different hyperparameters, use the **`retrain_model.py`** script:

```bash
python src/retrain_model.py
```

The retrained model will be saved as **`skin_cancer_model_retrained.h5`**.

### **4. Evaluate the Model**

To evaluate the trained model, run:

```bash
python src/analyze_predictions.py
```

This will generate a classification report and save it as **`classification_report.txt`** in the **`/results/`** folder.

### **5. Use the GUI**

To use the GUI application, run:

```bash
python src/gui_application_final.py
```

This will open a Tkinter window where users can:
- Upload a skin lesion image.
- Get a prediction on whether the lesion is cancerous or non-cancerous.
- If a cancerous lesion is detected, a reference image from the dataset will also be displayed for verification.

---

## **Evaluation Metrics**

The model is evaluated using the following metrics:
- **Accuracy**: The percentage of correct predictions.
- **Precision**: The proportion of true positives out of all positive predictions.
- **Recall**: The proportion of true positives out of all actual positive instances.
- **F1-Score**: The harmonic mean of precision and recall, balancing both metrics.

---

## **Conclusion**

This project demonstrates the application of deep learning for skin cancer detection. By classifying skin lesions as either cancerous or non-cancerous, the system can assist in early detection. The custom dataset and metadata, along with a retrainable CNN model, make this system adaptable for real-world use. The user-friendly GUI provides an intuitive interface for easy predictions.

---
