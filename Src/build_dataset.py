import numpy as np
import pandas as pd

from pathlib import Path
from Src.feature_extractor import extract_features

ROOT = Path(__file__).parent.parent

DATA_DIR = ROOT / "data" / "ESC-50"
AUDIO_DIR = DATA_DIR / "audio"
METADATA_PATH = DATA_DIR / "meta" / "esc50.csv"

OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

metadata = pd.read_csv(METADATA_PATH)

X = []
y = []
folds = []

print(f"Processing {len(metadata)} recordings...")

for index, row in metadata.iterrows():

    audio_path = AUDIO_DIR / row["filename"]

    try:
        features = extract_features(audio_path)

        X.append(features)
        y.append(row["category"])
        folds.append(row["fold"])

    except Exception as error:
        print(f"Error: {row['filename']} - {error}")

    if (index + 1) % 100 == 0:
        print(f"Processed {index + 1}/{len(metadata)}")

X = np.array(X)
y = np.array(y)
folds = np.array(folds)

np.save(OUTPUT_DIR / "X.npy", X)
np.save(OUTPUT_DIR / "y.npy", y)
np.save(OUTPUT_DIR / "folds.npy", folds)

print("\nDataset rebuilt successfully!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Folds shape:", folds.shape)
print("Classes:", len(np.unique(y)))
