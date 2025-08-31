import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Paths
train_dir = "data/images/train"
test_dir = "data/images/test"
model_path = "model/stress_model.h5"

# Parameters
img_height, img_width = 48, 48
batch_size = 32
epochs = 15

# Data generators
datagen = ImageDataGenerator(rescale=1.0 / 255)
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
    color_mode="grayscale",
)
test_data = datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
    color_mode="grayscale",
)

# Model architecture
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(img_height, img_width, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

# Compile
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train
checkpoint = ModelCheckpoint(
    model_path, monitor="val_accuracy", save_best_only=True, verbose=1
)
model.fit(train_data, validation_data=test_data, epochs=epochs, callbacks=[checkpoint])

print(f"\n✅ Model trained and saved at: {model_path}")
