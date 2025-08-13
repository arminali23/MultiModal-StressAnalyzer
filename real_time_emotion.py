import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from collections import Counter
import time
import numpy as np

# ----------- CNN MODEL -----------
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ----------- EMOTION DETECTION FUNCTION -----------
def run_emotion_detection():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = EmotionCNN(num_classes=7).to(device)
    model.load_state_dict(torch.load("emotion_modelv1.pt", map_location=device))
    model.eval()

    emotion_names = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
    stress_emotions = {"fear", "disgust", "angry", "sad"}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    predicted_emotions = []

    start_time = time.time()
    print("Detecting emotion for 10 seconds...")

    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (96, 96))
            face_tensor = transform(face).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(face_tensor)
                _, predicted = torch.max(output, 1)
                emotion = emotion_names[predicted.item()]
                predicted_emotions.append(emotion)

    cap.release()
    cv2.destroyAllWindows()

    counts = Counter(predicted_emotions)
    stress_count = sum(counts[e] for e in stress_emotions)
    total = sum(counts.values())
    stress_ratio = (stress_count / total) * 100 if total > 0 else 0

    if stress_ratio < 32.5:
        stress_level = "low"
    elif stress_ratio < 62.5:
        stress_level = "medium"
    else:
        stress_level = "high"

    print(f"Stress level: {stress_level} ({stress_ratio:.2f}%)")
    with open("stress_level.txt", "w") as f:
        f.write(stress_level)

if __name__ == "__main__":
    run_emotion_detection()