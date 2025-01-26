from tensorflow import keras

# Create the data generator
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255,  # Normalize pixel values to [0, 1]
    validation_split=0.2  # 20% of data for validation
)

# Training data
train_generator = train_datagen.flow_from_directory(
    r"E:\EECE\2nd year\1st term\Neuro-Notes\Dataset\archive\Images",  # Path to the dataset directory
    target_size=(128, 128),  # Resize all images to 128x128
    batch_size=32,
    class_mode='categorical',  # For multi-class classification
    subset='training'  # Training subset
)

# Validation data
validation_generator = train_datagen.flow_from_directory(
    r"E:\EECE\2nd year\1st term\Neuro-Notes\Dataset\archive\Images",
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Validation subset
)

# Display class indices
print("Class indices:", train_generator.class_indices)

# Build the CNN model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4, activation='softmax')  # 4 output classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define Early Stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy
    patience=5,             # Stop training if validation accuracy doesn't improve for 5 epochs
    restore_best_weights=True  # Restore weights from the best epoch
)

# Train the model with early stopping
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,  # Adjust epochs as needed
    callbacks=[early_stopping]  # Add EarlyStopping to callbacks list
)

# Save the entire model
model.save("E:\MathProject\.venv\Models\emotion_detection_by_image_model_CNN12.h5")

# Print training and validation accuracy for each epoch (up to the stopping point)
for epoch, acc, val_acc in zip(range(1, len(history.history['accuracy']) + 1),
                               history.history['accuracy'], history.history['val_accuracy']):
    print(f"Epoch {epoch}: Training Accuracy = {acc * 100:.2f}%, Validation Accuracy = {val_acc * 100:.2f}%")

# Evaluate the model on validation data (using best weights due to restore_best_weights)
final_loss, final_accuracy = model.evaluate(validation_generator)
print(f"\nFinal Validation Accuracy: {final_accuracy * 100:.2f}%")