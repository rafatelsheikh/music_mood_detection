import os
import librosa
import numpy as np
from pydub import AudioSegment


def add_noise(audio, noise_level=0.005):
    """Add Gaussian noise to audio."""
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise


def time_stretch(audio, rate=1.2):
    """Time-stretch the audio."""
    if audio.ndim > 1:
        raise ValueError("time_stretch expects a 1D NumPy array (mono audio).")
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(audio, sr, n_steps=2):
    """Pitch-shift the audio."""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def process_audio(file_path, output_dir, num_versions=3, sr=22050):
    """Apply augmentations and save MP3 files."""
    # Load MP3 file with librosa
    audio, sr = librosa.load(file_path, sr=sr, mono=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    for i in range(num_versions):
        augmented_audio = audio.copy()
        if i % 3 == 0:
            augmented_audio = add_noise(augmented_audio)
        elif i % 3 == 1:
            augmented_audio = time_stretch(augmented_audio, rate=1.1)
        elif i % 3 == 2:
            augmented_audio = pitch_shift(augmented_audio, sr, n_steps=1)

        # Convert back to AudioSegment for MP3 export
        augmented_audio_segment = AudioSegment(
            (augmented_audio * 32767).astype(np.int16).tobytes(),
            frame_rate=sr,
            sample_width=2,  # 16-bit audio
            channels=1
        )

        # Save the augmented file as MP3
        output_path = os.path.join(output_dir, f"{base_name}_augmented_{i + 1}.mp3")
        augmented_audio_segment.export(output_path, format="mp3")
        print(f"Saved augmented file: {output_path}")


def augment_dataset(input_dir, output_dir, num_versions=3):
    """Augment all MP3 files in a dataset."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mp3"):
                file_path = os.path.join(root, file)
                process_audio(file_path, output_dir, num_versions=num_versions)


# Paths
input_dir = r"E:\EECE\2nd year\1st term\Neuro-Notes\Dataset\archive\sad"  # Path to original audio files
output_dir = r"E:\EECE\2nd year\1st term\Neuro-Notes\Dataset\archive\Augmented data 4\Sad"  # Path for saving augmented files

# Number of augmented versions per file
num_versions = 3

# Perform augmentation
augment_dataset(input_dir, output_dir, num_versions=num_versions)
