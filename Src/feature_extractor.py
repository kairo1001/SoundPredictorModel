
import librosa
import numpy as np

SAMPLE_RATE = 22050
N_MFCC = 13
EXPECTED_FEATURES = 52


def extract_features(audio_path):

    # Load audio and resample to 22050 Hz
    signal, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Extract 13 MFCC coefficients
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # First-order and second-order deltas
    delta = librosa.feature.delta(mfcc)

    # Combine mean and standard deviation statistics
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.std(delta, axis=1)
    ])

    if features.shape[0] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, "
            f"got {features.shape[0]}"
        )

    return features
