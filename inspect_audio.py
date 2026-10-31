
from pathlib import Path
import pandas as pd
import soundfile as sf
import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "ESC-50"

metadata = pd.read_csv(DATA / "meta" / "esc50.csv")

for category in ["dog", "chainsaw", "rain"]:
    row = metadata[metadata["category"] == category].iloc[0]
    audio_path = DATA / "audio" / row["filename"]

    audio, sample_rate = sf.read(audio_path)

    print(f"\nCategory: {category}")
    print(f"File: {row['filename']}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(audio) / sample_rate:.2f} seconds")
    print(f"Shape: {audio.shape}")
    print(f"Peak amplitude: {np.max(np.abs(audio)):.3f}")
    print(f"Audio path: {audio_path.resolve()}")
