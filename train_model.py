import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --------------------------
# Create folder if missing
# --------------------------
os.makedirs("saved_models", exist_ok=True)

# --------------------------
# Dataset paths
# --------------------------
train_dir = r"C:\Users\abhir\OneDrive\Desktop\plant_disease_ai\backend\dataset\Train"
val_dir = r"C:\Users\abhir\OneDrive\Desktop\plant_disease_ai\backend\dataset\Validation"

# --------------------------
# Parameters (FAST TRAINING)
# --------------------------
img_size = 224
batch_size = 32
epochs = 2     # ✅ Only 2 epochs for fast training

# --------------------------
# Data generators
# --------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# --------------------------
# Save label map (NOW WORKS)
# --------------------------
np.save("saved_models/label_map.npy", train.class_indices)
print("label_map.npy saved ✔")

val = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# --------------------------
# Build MobileNetV2 model
# --------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_size, img_size, 3),
    include_top=False,
    pooling="avg",
    weights='imagenet'
)

# ❌ TRAINING FROZEN FOR SPEED
base_model.trainable = False     

x = Dense(128, activation='relu')(base_model.output)
x = Dropout(0.3)(x)
output = Dense(len(train.class_indices), activation='softmax')(x)

final_model = Model(inputs=base_model.input, outputs=output)

final_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

final_model.summary()

# --------------------------
# Callbacks
# --------------------------
checkpoint = ModelCheckpoint(
    "saved_models/crop_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

# --------------------------
# FAST TRAINING
# --------------------------
final_model.fit(
    train,
    validation_data=val,
    epochs=epochs,
    callbacks=[checkpoint, early_stop]
)

print("\n🎉 FAST TRAINING FINISHED!")
print("Model saved in saved_models/")
