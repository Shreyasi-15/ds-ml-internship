import type { AnalysisResult } from "../types/analysis";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

async function postAudio<T>(endpoint: string, file: File): Promise<T> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    body: formData,
    credentials: "include",
  });

  if (!response.ok) {
    let message = `Server responded with status ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) message += `: ${body.detail}`;
    } catch {
      // Keep the status-only message when the server does not return JSON.
    }
    throw new Error(message);
  }

  return (await response.json()) as T;
}

export async function analyzeAudio(file: File): Promise<AnalysisResult> {
  const [features, prediction] = await Promise.all([
    postAudio<Omit<
      AnalysisResult,
      | "predicted_label"
      | "confidence"
      | "bonafide_probability"
      | "spoof_probability"
      | "threshold"
      | "model_version"
      | "segments"
      | "breathing_cadence"
    >>("/extract-features", file),
    postAudio<Pick<
      AnalysisResult,
      | "filename"
      | "predicted_label"
      | "confidence"
      | "bonafide_probability"
      | "spoof_probability"
      | "threshold"
      | "model_version"
      | "segments"
      | "breathing_cadence"
    >>("/predict", file),
  ]);

  return {...features, ...prediction};
}
