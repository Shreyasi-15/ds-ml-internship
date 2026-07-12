import { useState } from "react";
import { UploadCloud, ShieldCheck } from "lucide-react";
import "./index.css";

interface AnalysisResult {
  filename: string;
  rt60_estimate_sec: number;
  reverb_ratio: number;
  breathing_band_energy: number;
  mel_spectrogram_shape: number[];
  waveform_summary: {
    duration_sec: number;
    peak_amplitude: number;
    rms: number;
  };
}

const API_BASE_URL = "http://127.0.0.1:8000";
function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
      setError(null);
      setProgress(0);
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
      setError(null);
      setProgress(0);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);
    setProgress(0);

    // Fake progress that creeps up while we wait for the real response.
    // Stops at 90% so it doesn't lie and claim "done" before the server replies.
    const progressTimer = setInterval(() => {
      setProgress((prev) => (prev < 90 ? prev + 10 : prev));
    }, 200);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/extract-features`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let message = `Server responded with status ${response.status}`;
        try {
          const body = await response.json();
          if (body?.detail) message += `: ${body.detail}`;
        } catch {
          // Keep the status-only message when the server did not return JSON.
        }
        throw new Error(message);
      }

      const data: AnalysisResult = await response.json();
      setResult(data);
      setProgress(100);
    } catch (err) {
      setError(
        err instanceof TypeError
          ? "Cannot reach the backend. Start FastAPI on port 8000 and keep that terminal running."
          : err instanceof Error
            ? err.message
            : "Something went wrong",
      );
      setProgress(0);
    } finally {
      clearInterval(progressTimer);
      setIsAnalyzing(false);
    }
  };
  return (
    <div className="app-shell">
      <h1>AcousticSpace</h1>
      <p className="subtitle">Deepfake detection via Room Impulse Response analysis</p>

      <label
        className={`upload-zone ${isDragging ? "dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <UploadCloud size={40} className="upload-icon" />
        <p>Click or drag an audio file here</p>
        <p className="hint">WAV, MP3, FLAC, or M4A</p>
        <input
          type="file"
          accept=".wav,.mp3,.flac,.m4a"
          onChange={handleFileChange}
        />
      </label>

      {selectedFile && (
        <div className="result-card">
          <div className="result-info">
            <p><strong>Selected:</strong> {selectedFile.name}</p>
            <p>{(selectedFile.size / 1024).toFixed(1)} KB</p>

            {(isAnalyzing || progress > 0) && (
              <div className="progress-row">
                <div className="progress-track">
                  <div
                    className="progress-fill"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <span className="progress-label">{progress}%</span>
              </div>
            )}
          </div>
          <div className="result-actions">
            <ShieldCheck size={22} color="#8b5cf6" />
            <button
              className="analyze-btn"
              onClick={handleAnalyze}
              disabled={isAnalyzing}
            >
              {isAnalyzing ? "Analyzing..." : "Start Analysis"}
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className="result-card" style={{ borderColor: "#ff5b6e" }}>
          <p style={{ color: "#ff5b6e" }}>Error: {error}</p>
        </div>
      )}

      {result && (
        <div className="result-card" style={{ display: "block" }}>
          <h3 style={{ marginTop: 0 }}>Analysis Results</h3>
          <p><strong>RT60 (reverb time):</strong> {result.rt60_estimate_sec < 0 ? "N/A — no reliable decay tail" : `${result.rt60_estimate_sec.toFixed(3)}s`}</p>
          <p><strong>Reverb ratio:</strong> {result.reverb_ratio.toFixed(3)}</p>
          <p><strong>Breathing-band energy:</strong> {result.breathing_band_energy.toFixed(3)}</p>
          <p><strong>Duration:</strong> {result.waveform_summary.duration_sec}s</p>
        </div>
      )}
    </div>
  );
}
export default App;