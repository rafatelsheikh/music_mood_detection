import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import tensorflow as tf


def create_spectrogram_for_model(audio_path, model_input_shape=(128, 128, 3)):
    """
    Converts an audio file to a spectrogram and preprocesses it for CNN input.

    Parameters:
    - audio_path: Path to the audio file.
    - model_input_shape: Expected input shape for the CNN model (default: (128, 128, 3)).

    Returns:
    - Preprocessed spectrogram image ready for CNN model inference.
    """
    # Load 30 seconds of audio
    y, sr = librosa.load(audio_path, duration=30)

    # Create mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.axis('off')  # Hide axes for a clean image

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=False)
    plt.close()

    # Convert the plot to a NumPy array
    buf.seek(0)
    img = Image.open(buf).convert('RGB')  # Convert to RGB to ensure 3 channels
    buf.close()

    # Resize to model input size
    resized_spectrogram = img.resize((model_input_shape[1], model_input_shape[0]))
    spectrogram_array = np.array(resized_spectrogram)

    # Normalize pixel values to [0, 1]
    normalized_spectrogram = spectrogram_array / 255.0

    return normalized_spectrogram


# Function to predict the emotion from an audio file
def predict_emotion(audio_path, model_path):
    """
    Predict the emotion from an audio file using a saved CNN model.

    Parameters:
    - audio_path: Path to the audio file.
    - model_path: Path to the saved CNN model.

    Returns:
    - Predicted emotion as a string.
    """
    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    # Create and preprocess the spectrogram
    spectrogram_input = create_spectrogram_for_model(audio_path)

    # Add batch dimension (model expects input shape as (batch_size, height, width, channels))
    spectrogram_input = np.expand_dims(spectrogram_input, axis=0)

    # Predict the emotion
    predictions = model.predict(spectrogram_input)

    # Map index to class names (based on training class indices)
    class_indices = {'Excited': 0, 'Happy': 1, 'Relax': 2, 'Sad': 3}  # Adjust as per your model
    class_labels = {v: k for k, v in class_indices.items()}

    # Print percentages for all classes
    print("Prediction Probabilities:")
    for idx, probability in enumerate(predictions[0]):
        print(f"{class_labels[idx]}: {probability * 100:.2f}%")

    # Get the class with the highest probability
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]

    return predicted_class


# Example usage
audio_file = r"E:\EECE\2nd year\1st term\Neuro-Notes\Music samples\Video Games.mp3"
model_file = r"E:\MathProject\.venv\Models\emotion_detection_by_image_model_CNN1.h5"  # Path to the saved model

predicted_emotion = predict_emotion(audio_file, model_file)
print(f"Predicted Emotion: {predicted_emotion}")
