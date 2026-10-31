
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
from contextlib import asynccontextmanager
import tempfile
import os
import shutil

from Src.predictor import load_model, predict_audio


# Load model once when the application starts
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    print("Loading SoundScape ML model...")
    model = load_model()
    print("Model loaded successfully!")

    yield

    print("SoundScape API shutting down...")


app = FastAPI(
    title="SoundScape API",
    description="ML-powered environmental sound classification",
    version="1.0",
    lifespan=lifespan
)


# Allow requests from our future React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Welcome to SoundScape API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Validate file extension
    if Path(file.filename).suffix.lower() != ".wav":
        raise HTTPException(
            status_code=400,
            detail="Only WAV audio files are supported."
        )

    temp_path = None

    try:
        # Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:

            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # Run ML prediction
        result = predict_audio(temp_path, model)

        return {
            "filename": file.filename,
            "predicted_category": result["predicted_category"],
            "top_predictions": result["top_predictions"]
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Audio prediction failed: {str(error)}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        await file.close()
