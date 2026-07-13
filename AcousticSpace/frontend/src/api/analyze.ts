import type { AnalysisResult } from "../types/analysis";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

export async function analyzeAudio(file: File): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/extract-features`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let message = `Server responded with status ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) message += `: ${body.detail}`;
    } catch {
      // Retain the status-only message when the server does not return JSON.
    }
    throw new Error(message);
  }

  return (await response.json()) as AnalysisResult;
}
