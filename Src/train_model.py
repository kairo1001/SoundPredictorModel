
import numpy as np
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

# Load complete dataset
X = np.load(DATA_DIR / "X.npy")
y = np.load(DATA_DIR / "y.npy")

print("Dataset:", X.shape)
print("Training on all", len(X), "recordings")

# Base SVM pipeline
base_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(
        kernel="rbf",
        C=10,
        gamma="scale"
    ))
])

# Calibrate probability estimates
model = CalibratedClassifierCV(
    estimator=base_model,
    method="sigmoid",
    cv=5,
    ensemble=False
)

# Train final model
print("\nTraining final calibrated SVM...")

model.fit(X, y)

# Save model
model_path = MODEL_DIR / "sound_classifier.pkl"

joblib.dump(model, model_path)

print("\nFinal model trained successfully!")
print("Training samples:", len(X))
print("Number of classes:", len(model.classes_))
print("Model saved at:", model_path)
