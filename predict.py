import numpy as np
from PIL import Image
import tensorflow.lite as tflite
import json

# Load class names once
with open("class_names.json", "r") as f:
    class_names = json.load(f)

MODEL_PATH = 'models/currency_model.tflite'

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def load_and_predict(image_path: str) -> dict:
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((256, 256))
        input_data = np.array(img, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = int(np.argmax(output_data))
        predicted_label = class_names[predicted_index]
        confidence = float(np.max(output_data)) * 100

        return {
            "status": "success",
            "prediction": predicted_label,
            "confidence": f"{confidence:.2f}%",
            "probabilities": output_data[0].tolist()
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
