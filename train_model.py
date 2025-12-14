import tensorflow as tf
import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# -----------------------------
# Image settings
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30   # increased for better learning

# -----------------------------
# Correct paths (UPDATE ONLY IF NEEDED)
# -----------------------------
train_dir = r"E:\dlproj\train"
val_dir   = r"E:\dlproj\validate"
test_dir  = r"E:\dlproj\test"

# -----------------------------
# Data generators
# -----------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.25,
    horizontal_flip=True
)

val_test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_test_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = val_test_gen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = train_data.num_classes
class_names = list(train_data.class_indices.keys())

print("Class Names:", class_names)

# -----------------------------
# Save class names for Streamlit
# -----------------------------
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# -----------------------------
# MobileNetV2 Model (TRANSFER LEARNING)
# -----------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False  # freeze pretrained layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Train
# -----------------------------
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# -----------------------------
# Evaluate
# -----------------------------
test_loss, test_acc = model.evaluate(test_data)
print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")

# -----------------------------
# Save model (modern format)
# -----------------------------
model.save("face_model.keras")
print("✅ Model saved as face_model.keras")
