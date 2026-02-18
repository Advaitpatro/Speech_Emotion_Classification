import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import librosa

class RAVDESSDataset(Dataset):
    def __init__(self, metadata_file, target_sr=22050, duration=3.0, augment=False):
        """
        Args:
            metadata_file (str): Path to the CSV we created in Step 1.
            target_sr (int): Sampling rate (22050 is standard for speech).
            duration (float): Fixed duration in seconds (3s is the prompt requirement).
            augment (bool): If True, applies random noise/pitch shifting.
        """
        self.metadata = pd.read_csv(metadata_file)
        self.target_sr = target_sr
        self.num_samples = int(target_sr * duration) # 22050 * 3 = 66150 samples
        self.augment = augment
        
        # Mapping string emotions to integer labels (0-7)
        self.emotion_map = {
            'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3,
            'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 1. Get file path and label
        audio_path = self.metadata.iloc[idx]['path']
        emotion_label = self.emotion_map[self.metadata.iloc[idx]['emotion']]

        # 2. Load Audio
        y, sr = librosa.load(audio_path, sr=self.target_sr)

        # 3. Trim Silence (Step from Phase 1)
        y, _ = librosa.effects.trim(y, top_db=20)

        # 4. Data Augmentation (Only if augment=True)
        if self.augment:
            # Random Noise Injection
            if np.random.rand() < 0.5:
                noise = np.random.randn(len(y))
                y = y + 0.005 * noise
            
            # Random Pitch Shift (computationally expensive, but good for small data)
            if np.random.rand() < 0.5:
                # shift by -2 to +2 semitones
                steps = np.random.randint(-2, 3) 
                y = librosa.effects.pitch_shift(y, sr=self.target_sr, n_steps=steps)

        # 5. Fix Length (Padding/Truncating)
        # We need every image to be exactly the same width.
        if len(y) > self.num_samples:
            # Truncate (cut off the end)
            y = y[:self.num_samples]
        else:
            # Pad (add zeros to the end)
            padding = self.num_samples - len(y)
            y = np.pad(y, (0, padding), mode='constant')

        # 6. Feature Extraction (Log-Mel Spectrogram)
        melspec = librosa.feature.melspectrogram(y=y, sr=self.target_sr, n_mels=128)
        log_melspec = librosa.power_to_db(melspec, ref=np.max)

        # 7. Normalization (Min-Max scaling to [0, 1])
        # This helps the CNN learn faster.
        norm_spec = (log_melspec - log_melspec.min()) / (log_melspec.max() - log_melspec.min())

        # 8. Convert to Tensor
        # Shape becomes: (Channels, Height, Width) -> (1, 128, 130)
        spec_tensor = torch.tensor(norm_spec, dtype=torch.float32).unsqueeze(0)
        
        return spec_tensor, torch.tensor(emotion_label, dtype=torch.long)