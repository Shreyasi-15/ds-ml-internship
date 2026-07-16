import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";

type WaveformViewerProps = {
  file: File;
};

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) {
    return "0:00";
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);

  return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
}

export function WaveformViewer({ file }: WaveformViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const waveSurferRef = useRef<WaveSurfer | null>(null);

  const [isReady, setIsReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    if (!containerRef.current) {
      return;
    }

    const audioUrl = URL.createObjectURL(file);

    const waveSurfer = WaveSurfer.create({
      container: containerRef.current,
      url: audioUrl,
      height: 96,
      waveColor: "#64748b",
      progressColor: "#7c3aed",
      cursorColor: "#c4b5fd",
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      normalize: true,
    });

    waveSurferRef.current = waveSurfer;

    waveSurfer.on("ready", (audioDuration) => {
      setDuration(audioDuration);
      setIsReady(true);
    });

    waveSurfer.on("timeupdate", (time) => {
      setCurrentTime(time);
    });

    waveSurfer.on("play", () => {
      setIsPlaying(true);
    });

    waveSurfer.on("pause", () => {
      setIsPlaying(false);
    });

    waveSurfer.on("finish", () => {
      setIsPlaying(false);
      setCurrentTime(0);
    });

    return () => {
      waveSurfer.destroy();
      waveSurferRef.current = null;
      URL.revokeObjectURL(audioUrl);
    };
  }, [file]);

  function togglePlayback() {
    if (waveSurferRef.current) {
      void waveSurferRef.current.playPause();
    }
  }

  return (
    <section className="waveform-card">
      <div className="waveform-heading">
        <div>
          <h2>Audio waveform</h2>
          <p>{file.name}</p>
        </div>

        <span>
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>

      <div
        ref={containerRef}
        className="waveform-canvas"
        aria-label={`Waveform for ${file.name}`}
      />

      <div className="waveform-controls">
        <button
          type="button"
          className="waveform-play-button"
          onClick={togglePlayback}
          disabled={!isReady}
        >
          {isPlaying ? "Pause" : "Play"}
        </button>

        {!isReady && <span>Preparing waveform…</span>}
      </div>
    </section>
  );
}