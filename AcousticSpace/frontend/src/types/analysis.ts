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
}
