
import librosa
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
audio_path = ROOT / "data" / "ESC-50" / "audio"

# Select one recording
file_path = audio_path / "1-100032-A-0.wav"

# Load audio
signal, sr = librosa.load(file_path, sr=22050)

print("Sample rate:", sr)
print("Audio duration:", len(signal) / sr)

# Extract 13 MFCC coefficients
mfcc = librosa.feature.mfcc(
    y=signal,
    sr=sr,
    n_mfcc=13
)

print("MFCC shape:", mfcc.shape)

# Average each coefficient across time
features = np.mean(mfcc, axis=1)

print("Feature vector shape:", features.shape)
print("Feature vector:")
print(features)
