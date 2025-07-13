import numpy as np
from PIL import Image
import os
import json
import tensorflow as tf

INPUT_DATA_PATH = 'uploads'
TFLITE_MODEL_PATH = 'models/currency_model.tflite'

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def load_and_predict():
    uploaded_files = os.listdir(INPUT_DATA_PATH)
    if not uploaded_files:
        return {"status": "error", "message": "No input image found."}

    image_file = uploaded_files[0]
    image_path = os.path.join(INPUT_DATA_PATH, image_file)

    try:
        # Preprocess
        img = Image.open(image_path).convert("RGB")
        img = img.resize((256, 256))
        input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_index = int(np.argmax(output_data))
        predicted_label = class_names[predicted_index]
        confidence = float(np.max(output_data)) * 100

        os.remove(image_path)

        return {
            "status": "success",
            "prediction": predicted_label,
            "confidence": f"{confidence:.2f}%",
            "probabilities": output_data[0].tolist()
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
