
import { useEffect, useRef, useState } from "react";
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


function Waveform({ audioUrl }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    const drawWaveform = async () => {
      try {
        const response = await fetch(audioUrl);
        const buffer = await response.arrayBuffer();

        const AudioContextClass =
          window.AudioContext || window.webkitAudioContext;

        const audioContext = new AudioContextClass();

        try {
          const audioBuffer =
            await audioContext.decodeAudioData(buffer);

          if (cancelled) return;

          const samples = audioBuffer.getChannelData(0);
          const canvas = canvasRef.current;

          if (!canvas) return;

          const ctx = canvas.getContext("2d");
          const width = canvas.width;
          const height = canvas.height;

          ctx.clearRect(0, 0, width, height);

          const bars = 180;
          const blockSize = Math.floor(samples.length / bars);

          // Find the maximum amplitude in the recording.
          let maxAmplitude = 0;

          for (let i = 0; i < samples.length; i++) {
            maxAmplitude = Math.max(
              maxAmplitude,
              Math.abs(samples[i])
            );
          }

          if (maxAmplitude === 0) return;

          const centerY = height / 2;

          for (let i = 0; i < bars; i++) {
            const start = i * blockSize;
            let peak = 0;

            for (
              let j = start;
              j < start + blockSize && j < samples.length;
              j++
            ) {
              peak = Math.max(
                peak,
                Math.abs(samples[j])
              );
            }

            // Normalize amplitudes relative to the recording.
            const normalized = peak / maxAmplitude;
            const barHeight = Math.max(
              2,
              normalized * height * 0.85
            );

            const x = (i / bars) * width;

            ctx.fillStyle = "#60a5fa";
            ctx.beginPath();

            ctx.roundRect(
              x,
              centerY - barHeight / 2,
              Math.max(2, width / bars - 2),
              barHeight,
              3
            );

            ctx.fill();
          }
        } finally {
          await audioContext.close();
        }
      } catch (error) {
        if (!cancelled) {
          console.error("Waveform generation failed:", error);
        }
      }
    };

    drawWaveform();

    return () => {
      cancelled = true;
    };
  }, [audioUrl]);

  return (
    <canvas
      ref={canvasRef}
      className="waveform-canvas"
      width={900}
      height={100}
    />
  );
}


function Spectrogram({ audioUrl }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    const generateSpectrogram = async () => {
      try {
        const response = await fetch(audioUrl);
        const arrayBuffer = await response.arrayBuffer();

        const AudioContextClass =
          window.AudioContext || window.webkitAudioContext;

        const audioContext = new AudioContextClass();

        try {
          const audioBuffer =
            await audioContext.decodeAudioData(arrayBuffer);

          if (cancelled) return;

          const samples = audioBuffer.getChannelData(0);
          const canvas = canvasRef.current;

          if (!canvas) return;

          const ctx = canvas.getContext("2d");

          const width = canvas.width;
          const height = canvas.height;

        const fftSize = 2048;
        const hopSize = 512;
        const bins = fftSize / 2;

          const totalFrames = Math.floor(
            (samples.length - fftSize) / hopSize
          );

          ctx.fillStyle = "#0d1625";
          ctx.fillRect(0, 0, width, height);

          if (totalFrames <= 0) return;

          // Hann window reduces spectral leakage.
          const windowFunction = new Float32Array(fftSize);

          for (let i = 0; i < fftSize; i++) {
            windowFunction[i] =
              0.5 -
              0.5 * Math.cos(
                (2 * Math.PI * i) / (fftSize - 1)
              );
          }

          // Fast Fourier Transform.
          const fft = (input) => {
            const real = new Float32Array(input);
            const imag = new Float32Array(fftSize);

            let j = 0;

            for (let i = 1; i < fftSize; i++) {
              let bit = fftSize >> 1;

              while (j & bit) {
                j ^= bit;
                bit >>= 1;
              }

              j ^= bit;

              if (i < j) {
                [real[i], real[j]] = [real[j], real[i]];
              }
            }

            for (let size = 2; size <= fftSize; size *= 2) {
              const half = size / 2;
              const angle = (-2 * Math.PI) / size;

              for (
                let start = 0;
                start < fftSize;
                start += size
              ) {
                for (let k = 0; k < half; k++) {
                  const cos = Math.cos(angle * k);
                  const sin = Math.sin(angle * k);

                  const even = start + k;
                  const odd = even + half;

                  const tr =
                    real[odd] * cos - imag[odd] * sin;

                  const ti =
                    real[odd] * sin + imag[odd] * cos;

                  real[odd] = real[even] - tr;
                  imag[odd] = imag[even] - ti;

                  real[even] += tr;
                  imag[even] += ti;
                }
              }
            }

            const magnitudes = new Float32Array(bins);

            for (let i = 0; i < bins; i++) {
              magnitudes[i] = Math.sqrt(
                real[i] ** 2 + imag[i] ** 2
              );
            }

            return magnitudes;
          };

          // Calculate frequency magnitudes for every time frame.
          const frames = [];
          let maxMagnitude = 0;

          for (let x = 0; x < width; x++) {
            const frame = Math.floor(
              (x / width) * totalFrames
            );

            const start = frame * hopSize;
            const frameData = new Float32Array(fftSize);

            for (let i = 0; i < fftSize; i++) {
              frameData[i] =
                samples[start + i] * windowFunction[i];
            }

            const magnitudes = fft(frameData);

            for (let i = 0; i < bins; i++) {
              maxMagnitude = Math.max(
                maxMagnitude,
                magnitudes[i]
              );
            }

            frames.push(magnitudes);
          }

          if (maxMagnitude === 0) return;

          const image = ctx.createImageData(width, height);

          // Draw normalized frequency intensities.
          for (let x = 0; x < width; x++) {
            const magnitudes = frames[x];

            for (let y = 0; y < height; y++) {
              // Logarithmic frequency scale.
            const minFreq = 80;
            const maxFreq = audioBuffer.sampleRate / 2;

            const normalizedY = (height - 1 - y) / (height - 1);

            const frequency =
              minFreq * Math.pow(
                maxFreq / minFreq,
                normalizedY
              );

            const bin = Math.min(
              bins - 1,
              Math.max(
                1,
                Math.round(
                  (frequency * fftSize) / audioBuffer.sampleRate
                )
              )
            );

              const magnitude = magnitudes[bin];

              // Decibels relative to the strongest frequency.
              const db = 20 * Math.log10(
                magnitude / maxMagnitude + 0.000001
              );

               const intensity = Math.pow(
                Math.max(
                  0,
                  Math.min(1, (db + 50) / 50)
                ),
                0.7
              );

              // Dark blue -> cyan -> yellow.
              let r, g, b;

              if (intensity < 0.5) {
                const t = intensity * 2;

                r = Math.round(8 + t * 10);
                g = Math.round(15 + t * 130);
                b = Math.round(65 + t * 160);
              } else {
                const t = (intensity - 0.5) * 2;

                r = Math.round(18 + t * 237);
                g = Math.round(145 + t * 100);
                b = Math.round(225 - t * 175);
              }

              const pixel = (y * width + x) * 4;

              image.data[pixel] = r;
              image.data[pixel + 1] = g;
              image.data[pixel + 2] = b;
              image.data[pixel + 3] = 255;
            }
          }

          if (!cancelled) {
            ctx.putImageData(image, 0, 0);
          }
        } finally {
          await audioContext.close();
        }
      } catch (error) {
        if (!cancelled) {
          console.error("Spectrogram generation failed:", error);
        }
      }
    };

    generateSpectrogram();

    return () => {
      cancelled = true;
    };
  }, [audioUrl]);

  return (
    <canvas
      ref={canvasRef}
      className="spectrogram-canvas"
      width={900}
      height={260}
    />
  );
}

function App() {
  const inputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [audioInfo, setAudioInfo] = useState(null);
  const [activePage, setActivePage] = useState("analyzer");

      useEffect(() => {
      if (!audioUrl) {
        setAudioInfo(null);
        return;
      }

      let cancelled = false;
      let audioContext;

      const loadAudioInfo = async () => {
        try {
          const response = await fetch(audioUrl);
          const buffer = await response.arrayBuffer();

          const AudioContextClass =
            window.AudioContext || window.webkitAudioContext;

          audioContext = new AudioContextClass();

          const audio = await audioContext.decodeAudioData(buffer);

          if (!cancelled) {
            setAudioInfo({
              duration: audio.duration.toFixed(2),
              sampleRate: audio.sampleRate,
              channels: audio.numberOfChannels
            });
          }
        } catch (error) {
          console.error("Audio information error:", error);
        } finally {
          if (audioContext) await audioContext.close();
        }
      };

      loadAudioInfo();

      return () => {
        cancelled = true;
      };
    }, [audioUrl]);

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

        <nav className="main-nav">
        <button
          className={activePage === "analyzer" ? "active" : ""}
          onClick={() => setActivePage("analyzer")}
        >
          Analyzer
        </button>

        <button
          className={activePage === "about" ? "active" : ""}
          onClick={() => setActivePage("about")}
        >
          About
        </button>
      </nav>

        <div className="system-status">
          <span className="status-dot"></span>
          ML ENGINE
          <span className="status-text">READY</span>
        </div>
      </header>

      <main className="main-content">
        {activePage === "analyzer" && (
         <>
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
              AUDIO WAVEFORM
            </div>

            <Waveform audioUrl={audioUrl} />

            <div className="spectrogram-section">
              <div className="player-label">
                <BarChart3 size={16} />
                FREQUENCY SPECTROGRAM
              </div>

              <Spectrogram audioUrl={audioUrl} />

              <div className="spectrogram-axis">
                <span>0 sec</span>
                <span>TIME →</span>
                <span>End</span>
              </div>
            </div>

            <audio
              src={audioUrl}
              controls
              className="audio-control"
            />
            </div>  
          )}

          {audioInfo && (
          <div className="audio-info">
            <div className="info-item">
              <span>DURATION</span>
              <strong>{audioInfo.duration} sec</strong>
            </div>

            <div className="info-item">
              <span>SAMPLE RATE</span>
              <strong>
                {(audioInfo.sampleRate / 1000).toFixed(1)} kHz
              </strong>
            </div>

            <div className="info-item">
              <span>CHANNELS</span>
              <strong>
                {audioInfo.channels === 1 ? "Mono" : `${audioInfo.channels} Channels`}
              </strong>
            </div>

            <div className="info-item">
              <span>FILE SIZE</span>
              <strong>{formatSize(file.size)}</strong>
            </div>
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
              <div className="prediction-top">
                <div className="prediction-label">
                  DETECTED SOUND
                </div>

                <span className="prediction-badge">
                  <CheckCircle2 size={13} />
                  TOP MATCH
                </span>
              </div>

              <h2>
                {result.predicted_category.replaceAll("_", " ")}
              </h2>

              <p>
                This is the highest-scoring category predicted
                by the machine learning model.
              </p>

              {result.top_predictions?.length > 0 && (
                <div className="primary-score">
                  <div className="primary-score-info">
                    <span>MODEL SCORE</span>
                    <strong>
                      {result.top_predictions[0].probability.toFixed(2)}%
                    </strong>
                  </div>

                  <div className="primary-score-track">
                    <div
                      className="primary-score-fill"
                      style={{
                        width: `${Math.max(
                          0,
                          Math.min(100, result.top_predictions[0].probability)
                        )}%`
                      }}
                    />
                  </div>
                </div>
              )}
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
        </>
        )}
        
{activePage === "about" && (
  <section className="inner-page">

    <div className="page-heading">
      <div className="eyebrow">
        <AudioLines size={15} />
        ABOUT THE PROJECT
      </div>

      <h1>About SoundScape</h1>

      <p>
        Exploring the world of environmental sounds
        through machine learning.
      </p>
    </div>

    <div className="about-panel">

      <div className="about-intro">
        <h2>What is SoundScape?</h2>

        <p>
          SoundScape is a machine learning-based
          environmental sound classification application.
          It analyzes uploaded audio recordings and predicts
          the most likely environmental sound category.
        </p>

        <p>
          The project uses the ESC-50 dataset, which contains
          2,000 audio recordings organized into 50 different
          environmental sound classes.
        </p>
      </div>

      <div className="about-divider" />

      <h2>Technologies used</h2>

      <div className="tech-grid">
        {[
          "Python",
          "Librosa",
          "NumPy",
          "Scikit-learn",
          "SVM",
          "FastAPI",
          "React",
          "Vite"
        ].map((tech) => (
          <div className="tech-item" key={tech}>
            {tech}
          </div>
        ))}
      </div>

      <div className="about-divider" />

      <h2>Classification pipeline</h2>

      <div className="pipeline">

        <div className="pipeline-step">
          <span>01</span>
          <div>
            <strong>Audio input</strong>
            <p>Upload an environmental audio recording in WAV format.</p>
          </div>
        </div>

        <div className="pipeline-step">
          <span>02</span>
          <div>
            <strong>Feature extraction</strong>
            <p>
              Extract 52 audio features using MFCC
              coefficients and their first-order delta features.
            </p>
          </div>
        </div>

        <div className="pipeline-step">
          <span>03</span>
          <div>
            <strong>Classification</strong>
            <p>
              A trained Support Vector Machine (SVM)
              classifies the extracted audio features.
            </p>
          </div>
        </div>

        <div className="pipeline-step">
          <span>04</span>
          <div>
            <strong>Prediction results</strong>
            <p>
              Display the predicted category and
              the top three model probability estimates.
            </p>
          </div>
        </div>

      </div>

      <div className="about-footer">
        SOUND CLASSIFICATION · 50 ENVIRONMENTAL CATEGORIES
      </div>

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
