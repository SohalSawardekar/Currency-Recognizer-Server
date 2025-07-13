# Conversion script to convert a Keras model to TensorFlow Lite format

import tensorflow as tf

# Load your model
model = tf.keras.models.load_model("models/currency_model.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Apply dynamic range quantization
tflite_model = converter.convert()

# Save the .tflite model
with open("models/currency_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model converted to TensorFlow Lite.")
