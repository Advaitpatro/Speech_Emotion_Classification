import torch
import librosa
import numpy as np
import sys
from model import SER_CNN

# Configuration
MODEL_PATH = "models/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS = {0:'Neutral', 1:'Calm', 2:'Happy', 3:'Sad', 4:'Angry', 5:'Fearful', 6:'Disgust', 7:'Surprised'}

def preprocess_audio(file_path):
    # 1. Load
    y, sr = librosa.load(file_path, sr=22050)
    # 2. Trim Silence
    y, _ = librosa.effects.trim(y, top_db=20)
    # 3. Pad/Truncate to 3s (66150 samples)
    max_len = 66150
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)), mode='constant')
        
    # 4. Mel Spectrogram
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_melspec = librosa.power_to_db(melspec, ref=np.max)
    
    # 5. Normalize
    norm_spec = (log_melspec - log_melspec.min()) / (log_melspec.max() - log_melspec.min())
    
    # 6. To Tensor (1, 1, 128, 130)
    tensor = torch.tensor(norm_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor

def predict(file_path):
    # Load Model
    model = SER_CNN(num_classes=8).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print("Error: Train the model first to generate 'models/best_model.pth'!")
        return

    model.eval()
    
    # Process Audio
    input_tensor = preprocess_audio(file_path).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)
        
    emotion = EMOTIONS[predicted_class.item()]
    print(f"\nFile: {file_path}")
    print(f"Prediction: {emotion.upper()}")
    print(f"Confidence: {confidence.item()*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_wav_file>")
    else:
        predict(sys.argv[1])