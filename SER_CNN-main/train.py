import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import os
import time

# Import our custom modules
from dataset import RAVDESSDataset
from model import SER_CNN

# --- CONFIGURATION ---
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METADATA_PATH = "ravdess_metadata.csv"
MODEL_SAVE_PATH = "models/best_model.pth"

def train():
    print("="*40)
    print(f"Initializing Training on: {DEVICE}")
    print("="*40)
    
    # 1. Load Metadata & Perform Stratified Split
    df = pd.read_csv(METADATA_PATH)
    indices = np.arange(len(df))
    labels = df['emotion'] 

    # Split: 80% Train, 20% Temp (Val + Test)
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
    
    # Split Temp: 50% Val, 50% Test (Results in 10% Val, 10% Test total)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=labels.iloc[temp_idx], random_state=42)

    print(f"Dataset Split -> Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # 2. Create Datasets
    # Train set gets augmentation, Val/Test do not
    full_ds_train = RAVDESSDataset(METADATA_PATH, augment=True)
    full_ds_eval = RAVDESSDataset(METADATA_PATH, augment=False)

    train_ds = Subset(full_ds_train, train_idx)
    val_ds = Subset(full_ds_eval, val_idx)
    
    # 3. Create DataLoaders
    # num_workers=0 is safer for Ubuntu/VSCode to prevent hanging
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. Initialize Model, Loss, Optimizer
    model = SER_CNN(num_classes=8).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    best_val_loss = float('inf')
    
    print("\nStarting training loop...")

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # --- TRAINING BATCH LOOP ---
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Status Update Every 10 Batches
            if (i + 1) % 10 == 0:
                print(f"  [Epoch {epoch+1}] Batch {i+1}/{len(train_loader)} -> Loss: {loss.item():.4f}")

        train_acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        epoch_time = time.time() - start_time

        # Print Epoch Summary
        print(f"Epoch [{epoch+1}/{EPOCHS}] ({epoch_time:.1f}s) "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}% F1: {val_f1:.4f}")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if not os.path.exists("models"):
                os.makedirs("models")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("  >>> Best Model Saved!")

    print("\n" + "="*40)
    print("TRAINING COMPLETE")
    print("="*40)

if __name__ == "__main__":
    train()