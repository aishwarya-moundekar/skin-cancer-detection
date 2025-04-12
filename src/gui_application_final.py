import tkinter as tk
from tkinter import filedialog, messagebox
from keras.models import load_model
import numpy as np
from PIL import Image, ImageTk

import pandas as pd
import os

# Load your trained model
model = load_model('C:\\Users\\aishw\\PycharmProjects\\pythonProject\\models\\skin_cancer_model.h5')

# Load the metadata
metadata_df = pd.read_csv('C:\\Users\\aishw\\PycharmProjects\\pythonProject\\data\\HAM10000_metadata_updated.csv')


def predict(image, image_path):
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    predicted_label_index = np.argmax(prediction)
    labels = ['akiec', 'bcc', 'bkl', 'mel', 'nv', 'vasc']
    predicted_label = labels[predicted_label_index]
    image_id = image_path.split('/')[-1]

    if image_id not in metadata_df['image_id'].values:
        return "non-cancerous", 1.0

    confidence = np.max(prediction)
    return predicted_label, confidence


def show_reference_image(predicted_label):
    # Assuming reference images are named after the labels
    reference_image_path = os.path.join('C:\\Users\\aishw\\PycharmProjects\\pythonProject\\data\\merged',
                                        f"{predicted_label}.jpg")
    if os.path.exists(reference_image_path):
        reference_image = Image.open(reference_image_path)
        reference_image.thumbnail((250, 250))
        img_tk = ImageTk.PhotoImage(reference_image)

        panel_ref = tk.Label(root, image=img_tk)
        panel_ref.image = img_tk  # Keep a reference
        panel_ref.pack(pady=10)


def on_upload():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image.thumbnail((250, 250))  # Resize for display
        img_tk = ImageTk.PhotoImage(image)

        # Clear previous panels if any
        for widget in root.winfo_children():
            if isinstance(widget, tk.Label) and widget != upload_btn:
                widget.destroy()

        panel = tk.Label(root, image=img_tk)
        panel.image = img_tk  # Keep a reference
        panel.pack(pady=10)

        predicted_label, confidence = predict(image, file_path)
        messagebox.showinfo("Prediction Result", f"Predicted Label: {predicted_label}\nConfidence: {confidence:.2f}")

        if predicted_label != "non-cancerous":
            show_reference_image(predicted_label)


def reset_application():
    for widget in root.winfo_children():
        if isinstance(widget, tk.Label) and widget != upload_btn:
            widget.destroy()
    messagebox.showinfo("Reset", "Application has been reset.")


root = tk.Tk()
root.title("Skin Cancer Detection")

upload_btn = tk.Button(root, text="Upload Image", command=on_upload)
upload_btn.pack(pady=20)

reset_btn = tk.Button(root, text="Reset", command=reset_application)
reset_btn.pack(pady=10)

root.mainloop()
