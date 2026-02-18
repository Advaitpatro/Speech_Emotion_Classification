import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, Subset

# Import your code
from dataset import RAVDESSDataset
from model import SER_CNN

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METADATA_PATH = "ravdess_metadata.csv"
MODEL_PATH = "models/best_model.pth"

def evaluate_model():
    print("Loading Test Data...")
    # 1. Re-create the EXACT same split using random_state=42
    df = pd.read_csv(METADATA_PATH)
    indices = np.arange(len(df))
    labels = df['emotion']
    
    # Replicate the split logic from train.py
    _, temp_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
    _, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=labels.iloc[temp_idx], random_state=42)
    
    # Create Test Set
    test_ds = Subset(RAVDESSDataset(METADATA_PATH, augment=False), test_idx)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # 2. Load Model
    model = SER_CNN(num_classes=8).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 3. Run Predictions
    all_preds = []
    all_labels = []
    
    print("Running Inference on Test Set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- DELIVERABLE 1: CONFUSION MATRIX ---
    emotion_names = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_names, yticklabels=emotion_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("\n[+] Confusion Matrix saved as 'confusion_matrix.png'")
    
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=emotion_names))

    # --- DELIVERABLE 2: PITCH BIAS CHECK (Gender Analysis) ---
    # We map our predictions back to the dataframe rows to check gender
    test_df = df.iloc[test_idx].copy()
    test_df['true_label_idx'] = all_labels
    test_df['pred_label_idx'] = all_preds
    
    # Calculate accuracy per gender
    male_acc = np.mean(test_df[test_df['gender'] == 'male']['true_label_idx'] == test_df[test_df['gender'] == 'male']['pred_label_idx'])
    female_acc = np.mean(test_df[test_df['gender'] == 'female']['true_label_idx'] == test_df[test_df['gender'] == 'female']['pred_label_idx'])
    
    print("\n--- Gender Bias / Pitch Analysis ---")
    print(f"Male Accuracy:   {male_acc*100:.2f}%")
    print(f"Female Accuracy: {female_acc*100:.2f}%")
    
    if abs(male_acc - female_acc) > 0.05:
        print(">> Analysis: The model shows significant pitch bias (>5% difference).")
    else:
        print(">> Analysis: The model generalizes well across genders.")

if __name__ == "__main__":
    evaluate_model()