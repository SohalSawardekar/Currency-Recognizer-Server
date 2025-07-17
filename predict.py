import numpy as np
import tensorflow as tf
import json

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

MODEL_PATH = "models/currency_model.keras"
IMAGE_SIZE = 224

def load_and_predict(image_path: str) -> dict:
    try:
        # Load and preprocess the image - EXACTLY like raw model
        image = tf.keras.utils.load_img(
            image_path,
            target_size=(IMAGE_SIZE, IMAGE_SIZE)
        )
        
        image_arr = tf.keras.utils.img_to_array(image)
        image_bat = tf.expand_dims(image_arr, axis=0)
        
        # Load the model
        model = tf.keras.models.load_model(MODEL_PATH)

        # Predict
        prediction = model.predict(image_bat)
        
        # FIXED: Apply softmax to entire prediction array (like raw model)
        score = tf.nn.softmax(prediction)  
        
        predicted_label = class_names[np.argmax(score)]
        confidence = float(100 * np.max(score))
        
        return {
            "status": "success",
            "prediction": predicted_label,
            "confidence": confidence,
        }

    except Exception as e:
        print(f"[ERROR] Exception occurred: {str(e)}")
        return {"status": "error", "message": str(e)}