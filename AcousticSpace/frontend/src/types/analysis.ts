export interface SegmentPrediction {
  start_sec: number;
  end_sec: number;
  label: "bonafide" | "spoof";
  spoof_probability: number;
  suspicious: boolean;
}

export interface AnalysisResult {
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
  predicted_label: "bonafide" | "spoof";
  confidence: number;
  bonafide_probability: number;
  spoof_probability: number;
  threshold: number;
  model_version: string;
  segments: SegmentPrediction[];
  breathing_cadence: {
    score: number;
    event_count: number;
    event_times_sec: number[];
  };
}
