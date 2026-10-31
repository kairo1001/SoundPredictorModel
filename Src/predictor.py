
import numpy as np
import joblib

from pathlib import Path
from Src.feature_extractor import extract_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sound_classifier.pkl"


def load_model():
    """Load the trained classifier."""
    return joblib.load(MODEL_PATH)


def predict_audio(audio_path, model):
    """Predict the sound category of an audio recording."""

    # Extract the same 52 features used during training
    features = extract_features(audio_path)

    # Convert to 2D array: one recording, 52 features
    features = features.reshape(1, -1)

    # Get predictions and calibrated probability estimates
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    # Find the three highest scores
    top_indices = np.argsort(probabilities)[::-1][:3]

    top_predictions = []

    for index in top_indices:
        top_predictions.append({
            "category": str(model.classes_[index]),
            "probability": round(
                float(probabilities[index]) * 100, 2
            )
        })

    return {
        "predicted_category": str(prediction),
        "top_predictions": top_predictions
    }


if __name__ == "__main__":

    import pandas as pd

    DATA_DIR = ROOT / "data" / "ESC-50"

    metadata = pd.read_csv(
        DATA_DIR / "meta" / "esc50.csv"
    )

    # Select an example from the previously held-out fold
    row = metadata[metadata["fold"] == 5].iloc[0]

    audio_path = DATA_DIR / "audio" / row["filename"]

    model = load_model()

    result = predict_audio(audio_path, model)

    print("\nAudio:", row["filename"])
    print("Actual category:", row["category"])
    print("Predicted category:", result["predicted_category"])

    print("\nTop predictions:")

    for item in result["top_predictions"]:
        print(
            item["category"],
            ":", item["probability"], "%"
        )
