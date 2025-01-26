import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np

def create_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path, duration=30)  # Load 30 seconds of audio
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Plot and save spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.axis('off')  # Hide axes for clean images
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Example usage
audio_folder = r"E:\EECE\2nd year\1st term\Neuro-Notes\Dataset\archive\Augmented data 5\Sad"
output_folder = r"E:\EECE\2nd year\1st term\Neuro-Notes\Dataset\archive\Images 8\Sad"

os.makedirs(output_folder, exist_ok=True)
for file in os.listdir(audio_folder):
    if file.endswith(".mp3") or file.endswith(".wav"):
        create_spectrogram(os.path.join(audio_folder, file), os.path.join(output_folder, file + ".png"))
