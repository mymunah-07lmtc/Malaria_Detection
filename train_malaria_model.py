import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# -------------------------------
# CONFIGURATION
# -------------------------------
IMG_SIZE = 128          # Image size (128x128 pixels)
BATCH_SIZE = 32         # Number of images processed at once
EPOCHS = 15             # Number of training cycles (15 is usually enough for this dataset)
DATASET_PATH = 'cell_images'  # Folder containing 'Parasitized' and 'Uninfected'

print("🦟 Starting Malaria Detection Model Training...")

# -------------------------------
# 1. LOAD DATASET
# -------------------------------
print("📂 Loading dataset...")
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,          # 20% of data for testing/validation
    subset="both",
    seed=123,                       # Ensures the same split every time
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Normalize pixel values (from 0-255 to 0-1)
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# -------------------------------
# 2. DATA AUGMENTATION (To make the model robust)
# -------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Apply augmentation ONLY to the training set
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Optimize performance (prefetching)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# -------------------------------
# 3. BUILD THE CNN MODEL
# -------------------------------
print("🧠 Building the Convolutional Neural Network...")
def create_model():
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dropout(0.5),  # Prevents overfitting
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # 0 = Uninfected, 1 = Parasitized
    ])
    return model

model = create_model()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# -------------------------------
# 4. TRAIN THE MODEL
# -------------------------------
print("⏳ Training started... (This will take 10-20 minutes on a CPU)")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# -------------------------------
# 5. EVALUATE AND SAVE
# -------------------------------
print("📊 Evaluating the model...")
test_loss, test_acc = model.evaluate(val_ds)
print(f'✅ Test Accuracy: {test_acc:.4f}')

# Save the model
model.save('malaria_detector.keras')
print("💾 Model saved as 'malaria_detector.keras'")

# -------------------------------
# 6. PLOT TRAINING HISTORY
# -------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('training_history.png')
print("📈 Training plot saved as 'training_history.png'")
plt.show()

print("🎉 Day 2 complete! Model is ready.")