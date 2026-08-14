import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('malaria_detector.keras')

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Optimize for mobile (reduces file size, faster inference)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert
tflite_model = converter.convert()

# Save the model
with open('malaria_detector.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Model converted to TensorFlow Lite!")
print(f"📁 File size: {len(tflite_model) / 1024 / 1024:.2f} MB")