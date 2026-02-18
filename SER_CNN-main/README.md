# 🎤 Speech Emotion Recognition (SER) using CNNs

This project focuses on **Speech Emotion Recognition (SER)** — identifying human emotions such as *happy, sad, angry, fearful, neutral*, etc., from raw speech audio using **Convolutional Neural Networks (CNNs)** implemented in **PyTorch**.  
The system is trained and evaluated on the **RAVDESS dataset** and was developed as part of an **AI Club project**.

---

## 📌 Problem Statement
Human speech carries rich emotional information beyond words. Automatically detecting emotions from speech has applications in:
- Human–Computer Interaction
- Mental health analysis
- Virtual assistants
- Call-center analytics

This project aims to classify emotions from speech signals using **log-Mel spectrogram features** and a **2D CNN architecture**.

---

## 📂 Dataset
- **Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Modality used:** Audio (speech)
- **Emotions:** Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
- **Speakers:** Male & Female actors
- **Sampling rate:** 16 kHz

The dataset is split into **training, validation, and test sets**.

---

## 🔊 Audio Preprocessing
Each audio sample undergoes the following preprocessing steps:
- Resampling to 16 kHz
- Conversion to **log-Mel spectrograms**
- Fixed-length padding / truncation
- Normalization

The resulting spectrograms are used as 2D inputs to the CNN.

---

## 🧠 Model Architecture
- **Type:** Custom 4-Layer 2D CNN
- **Framework:** PyTorch
- **Components:**
  - Convolution + ReLU layers
  - MaxPooling layers
  - Fully connected layers
  - Softmax output layer

The architecture is designed to balance accuracy and efficiency.

---

## ⚙️ Training Details
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam
- **Batch Size:** Configurable
- **Epochs:** Trained until validation performance stabilizes
- **Metrics:** Accuracy, F1-score, Confusion Matrix

Best model weights are saved based on validation accuracy.

---

## 📊 Results
- **Overall Accuracy:** ~60% on the RAVDESS dataset
- **Observations:**
  - Acoustic overlap between emotions like *happy* and *neutral*
  - Better performance on high-energy emotions (e.g., angry)
  - Gender-wise performance analyzed to check bias

A confusion matrix and gender bias report are generated.

---

## 📁 Project Structure
SER_CNN/
├── data/ 
├── model.py
├── dataset.py 
├── train.py 
├── evaluate.py 
├── predict.py 
├── requirements.txt 
└── README.md 

---

## 🚀 How to Run

### 1️⃣ Install dependencies
pip install -r requirements.txt

### 2️⃣ Train the model
python train.py

### 3️⃣ Evaluate the model
python evaluate.py

### 4️⃣ Predict emotion from an audio file
python predict.py samples/demo_audio.wav


