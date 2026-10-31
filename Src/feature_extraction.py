
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


# Extract MFCCs
mfcc = librosa.feature.mfcc(
    y=signal,
    sr=sr,
    n_mfcc=13
)

# Calculate first-order delta coefficients
delta = librosa.feature.delta(mfcc)

# Calculate second-order delta coefficients
delta2 = librosa.feature.delta(mfcc, order=2)

# Extract mean and standard deviation
features = np.concatenate([
    np.mean(mfcc, axis=1),
    np.std(mfcc, axis=1),
    np.mean(delta, axis=1),
    np.std(delta, axis=1)
])


print("Feature vector shape:", features.shape)
print("Feature vector:")
print(features)
