import { useEffect, useState } from "react";
import type {
  ChangeEvent,
  DragEvent,
} from "react";
import {
  Activity,
  AudioLines,
  CheckCircle2,
  Clock3,
  Cpu,
  FileAudio,
  FileText,
  Gauge,
  History as HistoryIcon,
  LayoutDashboard,
  Search,
  ShieldCheck,
  UploadCloud,
} from "lucide-react";

import { analyzeAudio } from "./api/analyze";
import { getCurrentUser } from "./api/auth";
import type { AuthUser } from "./api/auth";
import { AuthModal } from "./components/AuthModal";
import { WaveformViewer } from "./components/WaveformViewer";
import { ModelStatistics } from "./components/ModelStatistics";
import "./index.css";
import "./week3.css";
import type { AnalysisResult } from "./types/analysis";

const MAX_FILE_BYTES = 25 * 1024 * 1024;

const ACCEPTED_EXTENSIONS = [
  ".wav",
  ".mp3",
  ".flac",
  ".m4a",
];

type HistoryEntry = {
  id: string;
  filename: string;
  label: string;
  confidence: number;
  model: string;
  analyzedAt: string;
};

function validateFile(file: File): string | null {
  const extension = file.name
    .slice(file.name.lastIndexOf("."))
    .toLowerCase();

  if (!ACCEPTED_EXTENSIONS.includes(extension)) {
    return "Choose a WAV, MP3, FLAC, or M4A audio file.";
  }

  if (file.size === 0) {
    return "The selected file is empty.";
  }

  if (file.size > MAX_FILE_BYTES) {
    return "The selected file exceeds the 25 MB limit.";
  }

  return null;
}

function percentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function loadHistory(): HistoryEntry[] {
  try {
    const storedHistory = window.localStorage.getItem(
      "acousticspace-history",
    );

    if (!storedHistory) {
      return [];
    }

    const parsedHistory: unknown = JSON.parse(storedHistory);

    return Array.isArray(parsedHistory)
      ? (parsedHistory as HistoryEntry[])
      : [];
  } catch {
    return [];
  }
}

function App() {
  const [selectedFile, setSelectedFile] =
    useState<File | null>(null);

  const [isDragging, setIsDragging] =
    useState(false);

  const [isAnalyzing, setIsAnalyzing] =
    useState(false);

  const [statisticsRefresh, setStatisticsRefresh] =
    useState(0);

  const [progress, setProgress] =
    useState(0);

  const [result, setResult] =
    useState<AnalysisResult | null>(null);

  const [error, setError] =
    useState<string | null>(null);

  const [searchQuery, setSearchQuery] =
    useState("");

  const [history, setHistory] =
    useState<HistoryEntry[]>(loadHistory);
  const [authenticatedUser, setAuthenticatedUser] =
  useState<AuthUser | null>(null);

  const [authModalOpen, setAuthModalOpen] =
    useState(false);

  useEffect(() => {
    void getCurrentUser()
      .then(setAuthenticatedUser)
      .catch(() => setAuthenticatedUser(null));
  }, []);
  const suspiciousSegments =
    result?.segments.filter(
      (segment) => segment.suspicious,
    ) ?? [];

  const filteredHistory = history.filter((entry) =>
    entry.filename
      .toLowerCase()
      .includes(searchQuery.toLowerCase()),
  );

  const currentDate = new Intl.DateTimeFormat(
    "en-IN",
    {
      day: "2-digit",
      month: "short",
      year: "numeric",
    },
  ).format(new Date());

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

  const handleFileChange = (
    event: ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];

    if (file) {
      selectFile(file);
    }
  };

  const handleDrop = (
    event: DragEvent<HTMLLabelElement>,
  ) => {
    event.preventDefault();
    setIsDragging(false);

    const file = event.dataTransfer.files?.[0];

    if (file) {
      selectFile(file);
    }
  };

  const saveHistoryEntry = (
    data: AnalysisResult,
    file: File,
  ) => {
    const entry: HistoryEntry = {
      id: `${Date.now()}-${file.name}`,
      filename: file.name,
      label: data.predicted_label,
      confidence: data.confidence,
      model: data.model_version,
      analyzedAt: new Date().toISOString(),
    };

    setHistory((previousHistory) => {
      const updatedHistory = [
        entry,
        ...previousHistory,
      ].slice(0, 12);

      window.localStorage.setItem(
        "acousticspace-history",
        JSON.stringify(updatedHistory),
      );

      return updatedHistory;
    });
  };

  const clearHistory = () => {
    setHistory([]);
    window.localStorage.removeItem(
      "acousticspace-history",
    );
  };

  const handleAnalyze = async () => {
    if (!selectedFile || isAnalyzing) {
      return;
    }
    if (!authenticatedUser) {
      setAuthModalOpen(true);
      setError("Sign in before analyzing an audio file.");
      return;
    }
    setIsAnalyzing(true);
    setError(null);
    setProgress(0);

    const progressTimer = window.setInterval(() => {
      setProgress((previous) =>
        previous < 90
          ? previous + 10
          : previous,
      );
    }, 200);

    try {
      const data = await analyzeAudio(selectedFile);

      setResult(data);
      setProgress(100);
      saveHistoryEntry(data, selectedFile);
      setStatisticsRefresh(
        (previous) => previous + 1,
      );
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

  return (
    <div className="dashboard-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <ShieldCheck
              size={28}
              aria-hidden="true"
            />
          </div>

          <div>
            <strong>AcousticSpace</strong>
            <span>Audio Detection Platform</span>
          </div>
        </div>

        <nav className="sidebar-navigation">
          <span className="navigation-label">
            Navigation
          </span>

           <a
            className="navigation-item"
            href="#statistics"
          >
            <Gauge size={19} />
            Model statistics
          </a>

          <a
            className="navigation-item active"
            href="#dashboard"
          >
            <LayoutDashboard size={19} />
            Dashboard
          </a>

          <a
            className="navigation-item"
            href="#scanner"
          >
            <AudioLines size={19} />
            Audio analysis
          </a>

          <a
            className="navigation-item"
            href="#history"
          >
            <HistoryIcon size={19} />
            History
          </a>

          <a
            className="navigation-item"
            href="#assessment"
          >
            <FileText size={19} />
            Report
          </a>
        </nav>

        <div className="system-status">
          <span className="navigation-label">
            System status
          </span>

          <div className="system-status-card">
            <div>
              <span>API</span>
              <strong className="online-status">
                <i />
                {error
                  ? "Check connection"
                  : "Available"}
              </strong>
            </div>

            <div>
              <span>Active model</span>
              <strong>
                {result?.model_version ??
                  "wav2vec2-asvspoof-domain-v2"}
              </strong>
            </div>

            <div>
              <span>AST checkpoint</span>
              <strong className="experiment-status">
                Trained and loaded
              </strong>
            </div>

            <div>
              <span>Runtime</span>
              <strong>
                <Cpu size={15} />
                CPU
              </strong>
            </div>
          </div>
        </div>
      </aside>

      <main
        className="dashboard-content"
        id="dashboard"
      >
        <header className="dashboard-header">
          <div>
            <h1>
              AI Audio Intelligence Dashboard
            </h1>

            <p>
              Deepfake detection and acoustic
              evidence analysis
              <span className="header-date">
                <Clock3 size={15} />
                {currentDate}
              </span>
            </p>
          </div>

          <div className="header-actions">
            <label className="search-box">
              <Search
                size={18}
                aria-hidden="true"
              />

              <input
                type="search"
                value={searchQuery}
                placeholder="Search history"
                aria-label="Search analysis history"
                onChange={(event) =>
                  setSearchQuery(event.target.value)
                }
              />
            </label>

              <button
            type="button"
            className="analyst-profile"
            onClick={() => setAuthModalOpen(true)}
          >
            <div className="profile-avatar">
              {authenticatedUser
                ? authenticatedUser.display_name
                    .split(" ")
                    .map((part) => part[0])
                    .join("")
                    .slice(0, 2)
                    .toUpperCase()
                : "?"}
            </div>

            <div>
              <strong>
                {authenticatedUser?.display_name ?? "Sign in"}
              </strong>

              <span>
                {authenticatedUser?.email ?? "Analyst account"}
              </span>
            </div>
            </button>
           </div>
          </header>

        <section className="overview-grid">
          <article className="overview-card">
            <div>
              <span>Detection</span>
              <strong
                className={
                  result?.predicted_label === "spoof"
                    ? "danger-text"
                    : result
                      ? "success-text"
                      : ""
                }
              >
                {result
                  ? result.predicted_label === "spoof"
                    ? "Potential spoof"
                    : "Likely bonafide"
                  : "Waiting"}
              </strong>
            </div>

            <div className="overview-icon blue">
              <ShieldCheck size={24} />
            </div>
          </article>

          <article className="overview-card">
            <div>
              <span>Confidence</span>
              <strong className="purple-text">
                {result
                  ? percentage(result.confidence)
                  : "—"}
              </strong>
            </div>

            <div className="overview-icon purple">
              <Gauge size={24} />
            </div>
          </article>

          <article className="overview-card">
            <div>
              <span>Active model</span>
              <strong>
                {result?.model_version ??
                  "wav2vec2-asvspoof-domain-v2"}
              </strong>
            </div>

            <div className="overview-icon violet">
              <Cpu size={24} />
            </div>
          </article>

          <article className="overview-card">
            <div>
              <span>Status</span>
              <strong
                className={
                  isAnalyzing
                    ? "purple-text"
                    : result
                      ? "success-text"
                      : ""
                }
              >
                {isAnalyzing
                  ? "Analyzing"
                  : result
                    ? "Completed"
                    : "Ready"}
              </strong>
            </div>

            <div className="overview-icon green">
              {isAnalyzing ? (
                <Activity size={24} />
              ) : (
                <CheckCircle2 size={24} />
              )}
            </div>
          </article>
        </section>

        <div className="workspace-grid">
          <section
            className="scanner-panel dashboard-panel"
            id="scanner"
          >
            <div className="panel-heading">
              <div>
                <span className="eyebrow">
                  Audio input
                </span>
                <h2>
                  Audio Intelligence Scanner
                </h2>
                <p>
                  Upload an audio recording for
                  deepfake and acoustic analysis.
                </p>
              </div>

              <div className="panel-heading-icon">
                <AudioLines size={25} />
              </div>
            </div>

            <label
              className={`upload-zone ${
                isDragging ? "dragging" : ""
              }`}
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() =>
                setIsDragging(false)
              }
              onDrop={handleDrop}
            >
              <UploadCloud
                size={48}
                className="upload-icon"
                aria-hidden="true"
              />

              <strong>
                Drag and drop audio here
              </strong>

              <p>
                or click to browse your computer
              </p>

              <div className="format-list">
                <span>WAV</span>
                <span>MP3</span>
                <span>FLAC</span>
                <span>M4A</span>
              </div>

              <small>Maximum file size: 25 MB</small>

              <input
                type="file"
                accept=".wav,.mp3,.flac,.m4a,audio/*"
                onChange={handleFileChange}
              />
            </label>

            {selectedFile && (
              <section className="selected-file-card">
                <div className="selected-file-icon">
                  <FileAudio size={25} />
                </div>

                <div className="selected-file-details">
                  <strong>
                    {selectedFile.name}
                  </strong>

                  <span>
                    {(
                      selectedFile.size /
                      (1024 * 1024)
                    ).toFixed(2)}{" "}
                    MB
                  </span>

                  <span className="ready-text">
                    Ready for analysis
                  </span>

                  {(isAnalyzing ||
                    progress > 0) && (
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
                          style={{
                            width: `${progress}%`,
                          }}
                        />
                      </div>

                      <span className="progress-label">
                        {progress}%
                      </span>
                    </div>
                  )}
                </div>
              </section>
            )}

            <button
              type="button"
              className="analyze-button"
              onClick={handleAnalyze}
              disabled={
                !selectedFile || isAnalyzing
              }
            >
              <Cpu size={19} />

              {isAnalyzing
                ? "Analyzing audio…"
                : "Analyze audio"}
            </button>

            {error && (
              <div
                className="error-card"
                role="alert"
              >
                {error}
              </div>
            )}

            {selectedFile && (
              <WaveformViewer
                key={`${selectedFile.name}-${selectedFile.lastModified}`}
                file={selectedFile}
                segments={result?.segments}
              />
            )}
          </section>

          <section
            className="assessment-panel dashboard-panel"
            id="assessment"
            aria-live="polite"
          >
            <div className="panel-heading">
              <div>
                <span className="eyebrow">
                  Analysis report
                </span>
                <h2>AI Threat Assessment</h2>
                <p>
                  AST classification with acoustic
                  evidence.
                </p>
              </div>

              {result && (
                <span
                  className={`verdict-badge ${result.predicted_label}`}
                >
                  {result.predicted_label === "spoof"
                    ? "Potential spoof"
                    : "Likely bonafide"}
                </span>
              )}
            </div>

            {!result ? (
              <div className="empty-assessment">
                <ShieldCheck size={50} />

                <h3>No analysis available</h3>

                <p>
                  Select an audio file and run the
                  scanner to generate a report.
                </p>
              </div>
            ) : (
              <>
                <div className="assessment-summary">
                  <div>
                    <span>Confidence</span>
                    <strong>
                      {percentage(
                        result.confidence,
                      )}
                    </strong>
                  </div>

                  <div>
                    <span>Risk level</span>
                    <strong
                      className={
                        result.predicted_label ===
                        "spoof"
                          ? "danger-text"
                          : "success-text"
                      }
                    >
                      {result.predicted_label ===
                      "spoof"
                        ? "Review required"
                        : "Lower risk"}
                    </strong>
                  </div>

                  <div>
                    <span>Active model</span>
                    <strong>
                      {result.model_version}
                    </strong>
                  </div>
                </div>

                <p className="evidence-note">
                  This is a research AST detector, not
                  a final forensic verdict. Review all
                  segment and acoustic evidence.
                </p>

                <div className="probability-row">
                  <span>Bonafide</span>

                  <div>
                    <i
                      style={{
                        width: percentage(
                          result.bonafide_probability,
                        ),
                      }}
                    />
                  </div>

                  <strong>
                    {percentage(
                      result.bonafide_probability,
                    )}
                  </strong>
                </div>

                <div className="probability-row spoof-row">
                  <span>Spoof</span>

                  <div>
                    <i
                      style={{
                        width: percentage(
                          result.spoof_probability,
                        ),
                      }}
                    />
                  </div>

                  <strong>
                    {percentage(
                      result.spoof_probability,
                    )}
                  </strong>
                </div>

                <div className="prediction-summary">
                  <div>
                    <span>
                      Suspicious segments
                    </span>

                    <strong>
                      {suspiciousSegments.length} /{" "}
                      {result.segments.length}
                    </strong>
                  </div>

                  <div>
                    <span>
                      Breathing cadence
                    </span>

                    <strong>
                      {percentage(
                        result
                          .breathing_cadence
                          .score,
                      )}
                    </strong>
                  </div>

                  <div>
                    <span>Runtime</span>
                    <strong>CPU inference</strong>
                  </div>
                </div>

                <div className="report-section">
                  <h3>Segment evidence</h3>

                  {suspiciousSegments.length > 0 ? (
                    <div className="segment-list">
                      {suspiciousSegments.map(
                        (segment) => (
                          <div
                            key={`${segment.start_sec}-${segment.end_sec}`}
                          >
                            <span>
                              {segment.start_sec.toFixed(
                                1,
                              )}
                              –
                              {segment.end_sec.toFixed(
                                1,
                              )}{" "}
                              s
                            </span>

                            <strong>
                              {percentage(
                                segment
                                  .spoof_probability,
                              )}{" "}
                              spoof
                            </strong>
                          </div>
                        ),
                      )}
                    </div>
                  ) : (
                    <p className="evidence-note">
                      No four-second segment crossed
                      the spoof threshold.
                    </p>
                  )}
                </div>

                <div className="report-section">
                  <h3>Acoustic features</h3>

                  <div className="metrics-grid">
                    <div>
                      <span>RT60 proxy</span>
                      <strong>
                        {result.rt60_estimate_sec < 0
                          ? "N/A"
                          : `${result.rt60_estimate_sec.toFixed(
                              3,
                            )} s`}
                      </strong>
                    </div>

                    <div>
                      <span>Reverb ratio</span>
                      <strong>
                        {result.reverb_ratio.toFixed(
                          4,
                        )}
                      </strong>
                    </div>

                    <div>
                      <span>
                        Breathing-band ratio
                      </span>
                      <strong>
                        {result.breathing_band_energy.toFixed(
                          4,
                        )}
                      </strong>
                    </div>

                    <div>
                      <span>Duration</span>
                      <strong>
                        {result.waveform_summary.duration_sec.toFixed(
                          2,
                        )}{" "}
                        s
                      </strong>
                    </div>

                    <div>
                      <span>Peak amplitude</span>
                      <strong>
                        {result.waveform_summary.peak_amplitude.toFixed(
                          4,
                        )}
                      </strong>
                    </div>

                    <div>
                      <span>RMS</span>
                      <strong>
                        {result.waveform_summary.rms.toFixed(
                          4,
                        )}
                      </strong>
                    </div>

                    <div>
                      <span>Mel shape</span>
                      <strong>
                        {result.mel_spectrogram_shape.join(
                          " × ",
                        )}
                      </strong>
                    </div>
                  </div>
                </div>
              </>
            )}
          </section>
        </div>

         <ModelStatistics
          enabled={authenticatedUser !== null}
          refreshKey={String(statisticsRefresh)}
          currentResult={result}
        />

        <section
          className="history-panel dashboard-panel"
          id="history"
        >
          <div className="history-heading">
            <div>
              <span className="eyebrow">
                Browser storage
              </span>

              <h2>Recent analysis history</h2>

              <p>
                The latest analyses saved on this
                device.
              </p>
            </div>

            {history.length > 0 && (
              <button
                type="button"
                className="secondary-button"
                onClick={clearHistory}
              >
                Clear history
              </button>
            )}
          </div>

          {filteredHistory.length === 0 ? (
            <div className="empty-history">
              <HistoryIcon size={30} />

              <span>
                {searchQuery
                  ? "No matching analysis found."
                  : "No analysis history yet."}
              </span>
            </div>
          ) : (
            <div className="history-table-wrapper">
              <table className="history-table">
                <thead>
                  <tr>
                    <th>Recording</th>
                    <th>Detection</th>
                    <th>Confidence</th>
                    <th>Model</th>
                    <th>Analyzed</th>
                  </tr>
                </thead>

                <tbody>
                  {filteredHistory.map((entry) => (
                    <tr key={entry.id}>
                      <td>{entry.filename}</td>

                      <td>
                        <span
                          className={`history-label ${entry.label}`}
                        >
                          {entry.label}
                        </span>
                      </td>

                      <td>
                        {percentage(
                          entry.confidence,
                        )}
                      </td>

                      <td>{entry.model}</td>

                      <td>
                        {new Intl.DateTimeFormat(
                          "en-IN",
                          {
                            day: "2-digit",
                            month: "short",
                            hour: "2-digit",
                            minute: "2-digit",
                          },
                        ).format(
                          new Date(
                            entry.analyzedAt,
                          ),
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>
       {authModalOpen && (
        <AuthModal
          user={authenticatedUser}
          onAuthenticated={(user) => {
            setAuthenticatedUser(user);
            setError(null);
          }}
          onLogout={() => {
            setAuthenticatedUser(null);
            setResult(null);
          }}
          onClose={() => setAuthModalOpen(false)}
        />
      )}
    </div>
  );
}
export default App;

