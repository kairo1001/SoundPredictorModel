# SoundScape --- Environmental Sound Classification

SoundScape is a machine-learning web application that analyzes an
uploaded audio recording and predicts which of 50 environmental sound
categories it most closely resembles.

The application uses: - **Python + librosa** for audio feature
extraction - **scikit-learn (calibrated SVM)** for classification -
**FastAPI** for the prediction API - **React + Vite** for the web
dashboard

> **Important:** The trained model is already included in
> `models/sound_classifier.pkl`. You do **not** need to download ESC-50,
> rebuild the dataset, or train the model just to run the dashboard.

------------------------------------------------------------------------

## 1. Requirements

Install these before starting:

-   Python 3.10 or 3.11 (Python 3.11 is recommended)
-   Node.js LTS (includes npm)
-   Git (only if cloning the repository)

Check that Python and Node.js are available:

``` bash
python --version
node --version
npm --version
```

------------------------------------------------------------------------

## 2. Get the project

### Option A --- Clone with Git

``` bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd SoundScape_ML_miniProject
```

Replace `<YOUR-GITHUB-REPOSITORY-URL>` with the repository URL.

### Option B --- Download ZIP

1.  On GitHub, select **Code → Download ZIP**.
2.  Extract the ZIP.
3.  Open a terminal in the extracted project folder.

The project should contain at least:

``` text
SoundScape_ML_miniProject/
├── Src/
│   ├── __init__.py
│   ├── api.py
│   ├── feature_extractor.py
│   └── predictor.py
├── models/
│   └── sound_classifier.pkl
├── frontend/
│   ├── package.json
│   └── src/
├── data/
│   └── processed/              # Not needed to run inference
└── README.md
```

Training scripts and other project files may also be present.

------------------------------------------------------------------------

## 3. Set up the Python backend

Run these commands from the **project root** (the folder containing
`Src` and `frontend`).

### Windows PowerShell

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas librosa scikit-learn fastapi uvicorn python-multipart soundfile
```

If PowerShell blocks virtual-environment activation, use Command Prompt
instead:

``` bat
.venv\Scripts\activate.bat
```

Then run the `pip install` command above.

### macOS / Linux

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas librosa scikit-learn fastapi uvicorn python-multipart soundfile
```

**Model compatibility:** the saved `.pkl` model was created with a
particular scikit-learn version. Ideally, install the same version used
during training. Loading a scikit-learn pickle with a substantially
different version can produce warnings or errors. If the project
includes a `requirements.txt`, prefer `pip install -r requirements.txt`
instead of the package-install command above.

------------------------------------------------------------------------

## 4. Set up the React frontend

Open a **second terminal** in the project root, then:

``` bash
cd frontend
npm install
```

`npm install` reads `package.json` and installs the frontend
dependencies. You only need to do this once per fresh clone or
extraction.

------------------------------------------------------------------------

## 5. Run SoundScape

You need **two terminals**: one for the backend and one for the
frontend.

### Terminal 1 --- Start the FastAPI backend

From the project root, activate the virtual environment if it is not
already active, then run:

``` powershell
python -m uvicorn Src.api:app --reload
```

The API will be available at:

-   API: http://127.0.0.1:8000
-   Interactive API documentation: http://127.0.0.1:8000/docs

Keep this terminal running.

### Terminal 2 --- Start the React dashboard

From the `frontend` directory:

``` bash
npm run dev
```

Open the local URL printed by Vite. By default, it is:

http://localhost:5173

Keep this terminal running too.

------------------------------------------------------------------------

## 6. Use the dashboard

1.  Open the dashboard in your browser.
2.  Upload a `.wav` audio file.
3.  Play the audio in the dashboard if desired.
4.  Select **Analyze**.
5.  SoundScape sends the recording to the FastAPI backend.
6.  The backend extracts the 52 audio features and passes them to the
    saved SVM model.
7.  The dashboard displays the predicted category and the top three
    model scores.

For the current version, upload WAV audio. Recordings of a few seconds
are a good starting point.

------------------------------------------------------------------------

## 7. Understanding the prediction

ESC-50 contains 50 environmental sound categories. The current SVM is a
**closed-set classifier**: it always selects among those 50 classes.

That means an audio clip outside the ESC-50 categories (for example,
ordinary speech or music) will still receive one of the 50 labels. The
top scores should therefore be interpreted as the model's relative
estimates among its known classes---not proof that the sound actually
belongs to the predicted category.

Performance can also vary with recording quality, background noise,
microphone characteristics, and differences between the uploaded audio
and the training recordings.

------------------------------------------------------------------------

## 8. Dataset and model files

-   `models/sound_classifier.pkl` is required to run predictions and
    should be present in the repository.
-   The original `data/ESC-50` dataset is required only if you want to
    rebuild the feature dataset or retrain/reevaluate the model.
-   The processed training arrays are also not needed to run the
    dashboard.

To keep the Git repository straightforward, the ESC-50 dataset directory
can be excluded from Git. In particular, do not commit
`data/ESC-50/.git` as an embedded Git repository. If the dataset is
excluded, a fresh clone or ZIP extraction can still run the dashboard
because inference uses the saved model, not the training dataset.

If you want to reproduce training later, obtain ESC-50 separately from
its official repository and follow the project's dataset-building and
training scripts.

------------------------------------------------------------------------

## 9. Troubleshooting

### `ModuleNotFoundError` for a Python package

Make sure the virtual environment is activated in the backend terminal,
then install the missing dependency with `pip`.

### The frontend cannot connect to the API

-   Confirm the backend is running at `http://127.0.0.1:8000`.
-   Confirm the frontend is running at `http://localhost:5173`.
-   Check that the backend CORS configuration allows
    `http://localhost:5173`.

### The model file cannot be found

Confirm that this file exists:

``` text
models/sound_classifier.pkl
```

Run Uvicorn from the project root, not from inside `Src`.

### The model fails to load

Check the installed scikit-learn version against the version used when
the model was trained. Pickled scikit-learn estimators are not
guaranteed to be compatible across versions.

### Vite uses a different port

If port `5173` is already occupied, Vite may choose another port. The
backend CORS configuration currently allows `http://localhost:5173`;
update `allow_origins` in `Src/api.py` if Vite uses a different one.

------------------------------------------------------------------------

## 10. Notes for developers

The main inference path is:

``` text
Audio file
   ↓
React dashboard
   ↓  POST /predict
FastAPI (Src/api.py)
   ↓
Feature extraction (Src/feature_extractor.py)
   ↓
Saved model (models/sound_classifier.pkl)
   ↓
Prediction response
   ↓
React results display
```

The model is loaded once when the API starts, rather than being loaded
again for every uploaded recording.

To retrain or rebuild the model, use the project's dataset-building and
training scripts. This is separate from the normal steps required to
launch the dashboard.
