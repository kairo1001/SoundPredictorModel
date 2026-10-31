
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
X = np.load(DATA_DIR / "X.npy")
y = np.load(DATA_DIR / "y.npy")
folds = np.load(DATA_DIR / "folds.npy")

# Define candidate models
models = {
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=10,
            gamma="scale"
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5,
            weights="distance"
        ))
    ])
}

results = []

# Five-fold cross-validation using ESC-50 folds
for model_name, model in models.items():

    print(f"\nEvaluating {model_name}...")

    fold_metrics = []

    for fold in range(1, 6):

        train_mask = folds != fold
        test_mask = folds == fold

        X_train = X[train_mask]
        X_test = X[test_mask]

        y_train = y[train_mask]
        y_test = y[test_mask]

        # Train model
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions,
                average="macro", zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions,
                average="macro", zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions,
                average="macro", zero_division=0
            )
        }

        fold_metrics.append(metrics)

        print(
            f"Fold {fold}: "
            f"Accuracy={metrics['accuracy']:.3f}, "
            f"Macro F1={metrics['f1']:.3f}"
        )

    # Average metrics across five folds
    average_metrics = {
        metric: np.mean([result[metric] for result in fold_metrics])
        for metric in fold_metrics[0]
    }

    results.append({
        "Model": model_name,
        **average_metrics
    })

# Display comparison
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# Save comparison results
results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

print("\nComparison saved to models/model_comparison.csv")
