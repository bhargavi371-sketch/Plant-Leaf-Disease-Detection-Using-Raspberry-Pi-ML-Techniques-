import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# === Feature Extraction Function ===
def extract_features(image):
    image = cv2.resize(image, (128, 128))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Dataset info
dataset_path = "Test"
labels = ["Healthy", "Rust", "Powdery"]
IMG_SIZE = 128

data = []
target = []

# Load dataset and extract features
for label in labels:
    folder = os.path.join(dataset_path, label)
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        features = extract_features(img)
        data.append(features)
        target.append(label)

print(f"[INFO] Loaded {len(data)} samples.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("[INFO] Model Evaluation:")
print(classification_report(y_test, model.predict(X_test)))

# Save model
joblib.dump(model, "crop_disease_model.pkl")
print("[✔] Model saved as crop_disease_model.pkl")
