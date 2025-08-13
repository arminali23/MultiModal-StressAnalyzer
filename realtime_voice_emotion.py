import torch
import torch.nn as nn
import numpy as np
import librosa
import sounddevice as sd

# Settings
SAMPLE_RATE = 22050
DURATION = 10  # seconds
N_MFCC = 40
MAX_LEN = 200
LABELS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
MODEL_PATH = "voice_emotion_model.pt"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# === Model architecture ===
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.attn(lstm_output), dim=1)  # [batch, seq_len, 1]
        context = torch.sum(attn_weights * lstm_output, dim=1)       # [batch, hidden_dim * 2]
        return context, attn_weights

class AdvancedEmotionModel(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, num_layers=2, num_classes=8, dropout=0.3):
        super(AdvancedEmotionModel, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.attention = Attention(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        context, _ = self.attention(lstm_out)
        return self.classifier(context)


# === Audio recording ===
def record_audio(duration=3, sample_rate=22050):
    print("Speak now...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Done recording.\n")
    return np.squeeze(audio)


# === Feature extraction ===
def extract_mfcc(audio, sr=22050, n_mfcc=40, max_len=200):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T
    if len(mfcc) < max_len:
        pad_width = max_len - len(mfcc)
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len]
    return torch.tensor(mfcc, dtype=torch.float32)


# === Prediction ===
def predict(model, mfcc_tensor):
    model.eval()
    with torch.no_grad():
        mfcc_tensor = mfcc_tensor.unsqueeze(0).to(DEVICE)  # [1, time, mfcc]
        output = model(mfcc_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        label_idx = np.argmax(probs)
        return LABELS[label_idx], probs


# === Main ===
if __name__ == "__main__":
    audio = record_audio(DURATION, SAMPLE_RATE)
    mfcc_tensor = extract_mfcc(audio, SAMPLE_RATE, N_MFCC, MAX_LEN)

    model = AdvancedEmotionModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)

    predicted_label, probabilities = predict(model, mfcc_tensor)

    print(f"Predicted Emotion: {predicted_label.upper()}")
    print("Probabilities:")
    for lbl, prob in zip(LABELS, probabilities):
        print(f"  {lbl.ljust(10)}: {prob:.2f}")