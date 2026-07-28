import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.esm.js";

import type { SegmentPrediction } from "../types/analysis";

type WaveformViewerProps = {
  file: File;
  segments?: SegmentPrediction[];
};

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return "0:00";
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
}

export function WaveformViewer({
  file,
  segments = [],
}: WaveformViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const waveSurferRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<ReturnType<typeof RegionsPlugin.create> | null>(
    null,
  );

  const [isReady, setIsReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    if (!containerRef.current) return;
    const audioUrl = URL.createObjectURL(file);
    const regions = RegionsPlugin.create();
    const waveSurfer = WaveSurfer.create({
      container: containerRef.current,
      url: audioUrl,
      plugins: [regions],
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
    regionsRef.current = regions;
    waveSurfer.on("ready", (audioDuration) => {
      setDuration(audioDuration);
      setIsReady(true);
    });
    waveSurfer.on("timeupdate", setCurrentTime);
    waveSurfer.on("play", () => setIsPlaying(true));
    waveSurfer.on("pause", () => setIsPlaying(false));
    waveSurfer.on("finish", () => {
      setIsPlaying(false);
      setCurrentTime(0);
    });

    return () => {
      waveSurfer.destroy();
      waveSurferRef.current = null;
      regionsRef.current = null;
      URL.revokeObjectURL(audioUrl);
    };
  }, [file]);

  useEffect(() => {
    const regions = regionsRef.current;
    if (!regions || !isReady) return;
    regions.clearRegions();
    segments
      .filter((segment) => segment.suspicious)
      .forEach((segment) => {
        regions.addRegion({
          start: segment.start_sec,
          end: Math.min(segment.end_sec, duration),
          color: "rgba(239, 68, 68, 0.28)",
          drag: false,
          resize: false,
        });
      });
  }, [duration, isReady, segments]);

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
        {segments.some((segment) => segment.suspicious) && (
          <span className="segment-legend">Red = suspicious segment</span>
        )}
      </div>
    </section>
  );
}
