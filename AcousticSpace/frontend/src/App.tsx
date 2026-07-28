import { useState } from "react";
import type { ChangeEvent, DragEvent } from "react";
import { ShieldCheck, UploadCloud } from "lucide-react";

import { analyzeAudio } from "./api/analyze";
import { WaveformViewer } from "./components/WaveformViewer";
import "./index.css";
import "./week3.css";
import type { AnalysisResult } from "./types/analysis";

const MAX_FILE_BYTES = 25 * 1024 * 1024;
const ACCEPTED_EXTENSIONS = [".wav", ".mp3", ".flac", ".m4a"];

function validateFile(file: File): string | null {
  const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!ACCEPTED_EXTENSIONS.includes(extension)) {
    return "Choose a WAV, MP3, FLAC, or M4A audio file.";
  }
  if (file.size === 0) return "The selected file is empty.";
  if (file.size > MAX_FILE_BYTES) {
    return "The selected file exceeds the 25 MB limit.";
  }
  return null;
}

function percentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const selectFile = (file: File) => {
    const validationError = validateFile(file);
    setResult(null);
    setProgress(0);
    if (validationError) {
      setSelectedFile(null);
      setError(validationError);
      return;
    }
    setSelectedFile(file);
    setError(null);
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) selectFile(file);
  };

  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) selectFile(file);
  };

  const handleAnalyze = async () => {
    if (!selectedFile || isAnalyzing) return;
    setIsAnalyzing(true);
    setError(null);
    setProgress(0);

    const progressTimer = window.setInterval(() => {
      setProgress((previous) => (previous < 90 ? previous + 10 : previous));
    }, 200);

    try {
      const data = await analyzeAudio(selectedFile);
      setResult(data);
      setProgress(100);
    } catch (caught) {
      setError(
        caught instanceof TypeError
          ? "Cannot reach FastAPI. Start the backend on port 8000."
          : caught instanceof Error
            ? caught.message
            : "The analysis could not be completed.",
      );
      setProgress(0);
    } finally {
      window.clearInterval(progressTimer);
      setIsAnalyzing(false);
    }
  };

  const suspiciousSegments =
    result?.segments.filter((segment) => segment.suspicious) ?? [];

  return (
    <main className="app-shell">
      <h1>AcousticSpace</h1>
      <p className="subtitle">
        Analyze acoustic evidence and baseline deepfake confidence.
      </p>

      <label
        className={`upload-zone ${isDragging ? "dragging" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <UploadCloud size={40} className="upload-icon" aria-hidden="true" />
        <p>Click or drag an audio file here</p>
        <p className="hint">WAV, MP3, FLAC, or M4A · maximum 25 MB</p>
        <input
          type="file"
          accept=".wav,.mp3,.flac,.m4a,audio/*"
          onChange={handleFileChange}
        />
      </label>

      {selectedFile && (
        <section className="result-card">
          <div className="result-info">
            <p><strong>Selected:</strong> {selectedFile.name}</p>
            <p>{(selectedFile.size / 1024).toFixed(1)} KB</p>
            {(isAnalyzing || progress > 0) && (
              <div className="progress-row">
                <div
                  className="progress-track"
                  role="progressbar"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={progress}
                >
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
            <ShieldCheck size={22} aria-hidden="true" />
            <button
              type="button"
              onClick={handleAnalyze}
              disabled={isAnalyzing}
            >
              {isAnalyzing ? "Analyzing…" : "Start Analysis"}
            </button>
          </div>
        </section>
      )}

      {selectedFile && (
        <WaveformViewer
          key={`${selectedFile.name}-${selectedFile.lastModified}`}
          file={selectedFile}
          segments={result?.segments}
        />
      )}

      {error && <div className="error-card" role="alert">{error}</div>}

      {result && (
        <>
          <section className="prediction-panel" aria-live="polite">
            <div className="prediction-heading">
              <div>
                <span className="eyebrow">Baseline CNN result</span>
                <h2 className={`verdict ${result.predicted_label}`}>
                  {result.predicted_label === "spoof"
                    ? "Potential spoof"
                    : "Likely bonafide"}
                </h2>
              </div>
              <strong className="confidence">
                {percentage(result.confidence)} confidence
              </strong>
            </div>

            <p className="evidence-note">
              This is a research baseline, not a final forensic verdict.
              Review the segment evidence and acoustic proxies together.
            </p>

            <div className="probability-row">
              <span>Bonafide</span>
              <div><i style={{width: percentage(result.bonafide_probability)}} /></div>
              <strong>{percentage(result.bonafide_probability)}</strong>
            </div>
            <div className="probability-row spoof-row">
              <span>Spoof</span>
              <div><i style={{width: percentage(result.spoof_probability)}} /></div>
              <strong>{percentage(result.spoof_probability)}</strong>
            </div>

            <div className="prediction-summary">
              <div>
                <span>Suspicious segments</span>
                <strong>{suspiciousSegments.length} / {result.segments.length}</strong>
              </div>
              <div>
                <span>Breathing-cadence score</span>
                <strong>{percentage(result.breathing_cadence.score)}</strong>
              </div>
              <div>
                <span>Model</span>
                <strong>{result.model_version}</strong>
              </div>
            </div>

            <h3>Segment evidence</h3>
            {suspiciousSegments.length ? (
              <div className="segment-list">
                {suspiciousSegments.map((segment) => (
                  <div key={`${segment.start_sec}-${segment.end_sec}`}>
                    <span>
                      {segment.start_sec.toFixed(1)}–{segment.end_sec.toFixed(1)} s
                    </span>
                    <strong>
                      {percentage(segment.spoof_probability)} spoof
                    </strong>
                  </div>
                ))}
              </div>
            ) : (
              <p className="evidence-note">
                No four-second segment crossed the spoof threshold.
              </p>
            )}
          </section>

          <section className="results-panel" aria-live="polite">
            <h2>Acoustic feature report</h2>
            <p className="evidence-note">
              These values are diagnostic proxies and do not individually
              prove whether a recording is real or fake.
            </p>
            <div className="metrics-grid">
              <div>
                <span>RT60 proxy</span>
                <strong>
                  {result.rt60_estimate_sec < 0
                    ? "N/A"
                    : `${result.rt60_estimate_sec.toFixed(3)} s`}
                </strong>
              </div>
              <div>
                <span>Reverb ratio</span>
                <strong>{result.reverb_ratio.toFixed(4)}</strong>
              </div>
              <div>
                <span>Breathing-band ratio</span>
                <strong>{result.breathing_band_energy.toFixed(4)}</strong>
              </div>
              <div>
                <span>Duration</span>
                <strong>{result.waveform_summary.duration_sec.toFixed(2)} s</strong>
              </div>
              <div>
                <span>Peak amplitude</span>
                <strong>{result.waveform_summary.peak_amplitude.toFixed(4)}</strong>
              </div>
              <div>
                <span>RMS</span>
                <strong>{result.waveform_summary.rms.toFixed(4)}</strong>
              </div>
              <div>
                <span>Mel shape</span>
                <strong>{result.mel_spectrogram_shape.join(" × ")}</strong>
              </div>
            </div>
          </section>
        </>
      )}
    </main>
  );
}

export default App;
