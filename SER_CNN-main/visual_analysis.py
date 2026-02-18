import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Load the metadata you just created
df = pd.read_csv("ravdess_metadata.csv")

# 2. Pick one 'angry' and one 'sad' file
# We use .iloc[0] to just grab the first one it finds for each emotion
angry_row = df[df['emotion'] == 'angry'].iloc[0]
sad_row = df[df['emotion'] == 'sad'].iloc[0]

angry_file = angry_row['path']
sad_file = sad_row['path']

print(f"Comparing:\nAngry: {angry_file}\nSad: {sad_file}")

def plot_comparison(angry_path, sad_path):
    plt.figure(figsize=(15, 6))

    # --- PLOT 1: ANGRY ---
    y_angry, sr = librosa.load(angry_path)
    # Trim silence (removes dead air at start/end)
    y_angry, _ = librosa.effects.trim(y_angry, top_db=20)
    # Compute Mel Spectrogram
    S_angry = librosa.feature.melspectrogram(y=y_angry, sr=sr, n_mels=128)
    S_dB_angry = librosa.power_to_db(S_angry, ref=np.max)

    plt.subplot(1, 2, 1)
    librosa.display.specshow(S_dB_angry, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title("High Arousal: Angry")

    # --- PLOT 2: SAD ---
    y_sad, sr = librosa.load(sad_path)
    # Trim silence
    y_sad, _ = librosa.effects.trim(y_sad, top_db=20)
    # Compute Mel Spectrogram
    S_sad = librosa.feature.melspectrogram(y=y_sad, sr=sr, n_mels=128)
    S_dB_sad = librosa.power_to_db(S_sad, ref=np.max)

    plt.subplot(1, 2, 2)
    librosa.display.specshow(S_dB_sad, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title("Low Arousal: Sad")

    plt.tight_layout()
    plt.show()

# Run the function
plot_comparison(angry_file, sad_file)