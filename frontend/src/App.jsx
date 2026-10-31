
import { useRef, useState } from "react";
import {
  AudioLines,
  UploadCloud,
  FileAudio,
  Play,
  Pause,
  X,
  LoaderCircle,
  CheckCircle2,
  AlertCircle,
  Waves,
  Clock3,
  BarChart3,
  Sparkles
} from "lucide-react";

import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const inputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);

  const selectFile = (selectedFile) => {
    if (!selectedFile) return;

    if (!selectedFile.name.toLowerCase().endsWith(".wav")) {
      setError("Please select a WAV audio file.");
      return;
    }

    setError("");
    setResult(null);
    setFile(selectedFile);

    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }

    setAudioUrl(URL.createObjectURL(selectedFile));
  };

  const handleFileChange = (event) => {
    selectFile(event.target.files[0]);
    event.target.value = "";
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    selectFile(event.dataTransfer.files[0]);
  };

  const removeFile = () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);

    setFile(null);
    setAudioUrl(null);
    setResult(null);
    setError("");
  };

  const analyzeAudio = async () => {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setResult(data);

    } catch (err) {
      setError(
        err.message || "Unable to connect to the SoundScape API."
      );
    } finally {
      setLoading(false);
    }
  };

  const formatSize = (bytes) => {
    return bytes < 1024 * 1024
      ? `${(bytes / 1024).toFixed(1)} KB`
      : `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="app-shell">

      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <AudioLines size={23} />
          </div>
          <span>SoundScape</span>
        </div>

        <div className="system-status">
          <span className="status-dot"></span>
          ML ENGINE
          <span className="status-text">READY</span>
        </div>
      </header>

      <main className="main-content">

        <section className="hero">
          <div className="eyebrow">
            <Waves size={15} />
            ENVIRONMENTAL SOUND CLASSIFICATION
          </div>

          <h1>
            Understand the sounds
            <br />
            <span>around you.</span>
          </h1>

          <p>
            Upload an audio recording and let our machine learning
            model identify the environmental sound.
          </p>
        </section>

        <section className="workspace">

          <div className="section-heading">
            <div>
              <h2>Audio analysis</h2>
              <p>Upload a recording to begin classification.</p>
            </div>

            <div className="format-label">
              WAV SUPPORTED
            </div>
          </div>

          {!file ? (

            <div
              className={`upload-area ${dragging ? "dragging" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => inputRef.current.click()}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".wav,audio/wav"
                hidden
                onChange={handleFileChange}
              />

              <div className="upload-icon">
                <UploadCloud size={31} />
              </div>

              <h3>Drop your audio file here</h3>

              <p>
                or <span>browse files</span> from your device
              </p>

              <div className="upload-limit">
                <FileAudio size={14} />
                WAV AUDIO FILES
              </div>
            </div>

          ) : (

            <div className="file-card">
              <div className="file-icon">
                <FileAudio size={25} />
              </div>

              <div className="file-details">
                <h3>{file.name}</h3>
                <p>{formatSize(file.size)}</p>
              </div>

              <button
                className="icon-button"
                onClick={removeFile}
                disabled={loading}
                aria-label="Remove file"
              >
                <X size={19} />
              </button>
            </div>

          )}

          {audioUrl && (
            <div className="audio-player">
              <div className="player-label">
                <Waves size={16} />
                AUDIO PREVIEW
              </div>

              <audio
                src={audioUrl}
                controls
                className="audio-control"
              />
            </div>
          )}

          {error && (
            <div className="error-message">
              <AlertCircle size={17} />
              {error}
            </div>
          )}

          <button
            className="analyze-button"
            onClick={analyzeAudio}
            disabled={!file || loading}
          >
            {loading ? (
              <>
                <LoaderCircle className="spin" size={19} />
                Analyzing audio...
              </>
            ) : (
              <>
                <Sparkles size={18} />
                Analyze recording
              </>
            )}
          </button>

        </section>

        {result && (

          <section className="results-section">

            <div className="results-heading">
              <div>
                <div className="eyebrow">
                  <CheckCircle2 size={15} />
                  ANALYSIS COMPLETE
                </div>
                <h2>Classification results</h2>
              </div>

              <span className="result-status">SUCCESS</span>
            </div>

            <div className="prediction-card">
              <div className="prediction-label">
                DETECTED SOUND
              </div>

              <h2>{result.predicted_category.replaceAll("_", " ")}</h2>

              <p>
                The model identified this as the most likely
                sound category.
              </p>
            </div>

            <div className="scores-card">
              <div className="scores-title">
                <div>
                  <BarChart3 size={19} />
                  <h3>Top predictions</h3>
                </div>
                <span>MODEL SCORES</span>
              </div>

              {result.top_predictions.map((item, index) => (
                <div className="score-row" key={item.category}>

                  <div className="score-info">
                    <div className="score-name">
                      <span className="rank">
                        0{index + 1}
                      </span>
                      {item.category.replaceAll("_", " ")}
                    </div>

                    <strong>{item.probability.toFixed(2)}%</strong>
                  </div>

                  <div className="score-track">
                    <div
                      className={`score-fill score-${index + 1}`}
                      style={{
                        width: `${item.probability}%`
                      }}
                    />
                  </div>

                </div>
              ))}

              <p className="score-note">
                Scores represent calibrated model probability
                estimates, not guaranteed certainty.
              </p>
            </div>

          </section>

        )}

      </main>

      <footer className="footer">
        <div>
          <AudioLines size={16} />
          SoundScape
        </div>
        <span>50 environmental sound categories</span>
      </footer>

    </div>
  );
}

export default App;
