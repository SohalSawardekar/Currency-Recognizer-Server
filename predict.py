import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import json

# Load class labels once
with open("class_names.json", "r") as f:
    class_names = json.load(f)

MODEL_PATH = 'models/currency_model.keras'
INPUT_DATA_PATH = 'uploads'

def load_and_predict() -> dict:
    # Load the pre-trained model
    model = load_model(MODEL_PATH)

    # Get the first file from the uploads directory
    uploaded_files = os.listdir(INPUT_DATA_PATH)
    if not uploaded_files:
        return {"status": "error", "message": "No input image found."}

    image_file = uploaded_files[0]
    image_path = os.path.join(INPUT_DATA_PATH, image_file)

    try:
        # Load and preprocess the image
        img = Image.open(image_path).convert("RGB")
        img = img.resize((256, 256))
        input_data = np.array(img) / 255.0
        input_data = input_data.reshape(1, 256, 256, 3)

        # Make prediction
        prediction = model.predict(input_data)
        predicted_index = int(np.argmax(prediction))
        predicted_label = class_names[predicted_index]
        confidence = float(np.max(prediction)) * 100  # As percentage

        # Clean up: remove uploaded file
        os.remove(image_path)

        return {
            "status": "success",
            "prediction": predicted_label,
            "confidence": f"{confidence:.2f}%",
            "probabilities": prediction[0].tolist()
        }


    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
