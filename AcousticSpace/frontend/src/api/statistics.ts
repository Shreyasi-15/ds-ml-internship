const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export type AnalysisHistoryItem = {
  id: number;
  filename: string;
  predicted_label: "bonafide" | "spoof";
  confidence: number;
  model_version: string;
  analyzed_at: string;
  retained: boolean;
};

export type ModelEvaluationMetrics = {
  accuracy: number; precision: number; recall: number;
  f1_score: number; eer: number; evaluated_recordings: number;
};
export type ConfidenceBucket = { range: string; count: number };
export type DailyPrediction = {
  date: string; bonafide: number; spoof: number;
};
export type AnalysisStatistics = {
  total_analyses: number;
  bonafide_detections: number;
  spoof_detections: number;
  average_confidence: number;
  model_evaluation: ModelEvaluationMetrics;
  confidence_distribution: ConfidenceBucket[];
  daily_predictions: DailyPrediction[];
};

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string };
    return body.detail ?? "The request failed.";
  } catch { return "The request failed."; }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    credentials: "include", ...init,
  });
  if (!response.ok) throw new Error(await readError(response));
  return response.json() as Promise<T>;
}

export const getStatistics = () => request<AnalysisStatistics>("/statistics");
export const getRecentHistory = () =>
  request<AnalysisHistoryItem[]>("/history/recent");
export const getSavedHistory = () =>
  request<AnalysisHistoryItem[]>("/history/saved");
export const keepAnalysis = (id: number) =>
  request<{ message: string }>(`/history/${id}/keep`, { method: "PATCH" });
export const deleteAnalysis = (id: number) =>
  request<{ message: string }>(`/history/${id}`, { method: "DELETE" });
