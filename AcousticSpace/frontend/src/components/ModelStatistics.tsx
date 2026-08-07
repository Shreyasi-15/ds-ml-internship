import { useEffect, useState } from "react";
import type { AnalysisResult } from "../types/analysis";
import {
  Activity,
  Bot,
  BrainCircuit,
  Gauge,
  UserRound,
} from "lucide-react";

import {
  getAnalysisHistory,
  getStatistics,
} from "../api/statistics";
import type {
  AnalysisHistoryItem,
  AnalysisStatistics,
} from "../api/statistics";

type ModelStatisticsProps = {
  enabled: boolean;
  refreshKey: string;
  currentResult: AnalysisResult | null;
};

function percentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function ModelStatistics({
  enabled,
  refreshKey,
  currentResult,
}: ModelStatisticsProps) {
  const [statistics, setStatistics] =
    useState<AnalysisStatistics | null>(null);

  const [history, setHistory] =
    useState<AnalysisHistoryItem[]>([]);

  const [error, setError] =
    useState<string | null>(null);

  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    let active = true;

    const loadStatistics = async () => {
      setLoading(true);
      setError(null);

      try {
        const [statisticsResult, historyResult] =
          await Promise.all([
            getStatistics(),
            getAnalysisHistory(),
          ]);

        if (active) {
          setStatistics(statisticsResult);
          setHistory(historyResult);
        }
      } catch (caught) {
        if (active) {
          setError(
            caught instanceof Error
              ? caught.message
              : "Statistics could not be loaded.",
          );
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    void loadStatistics();

    return () => {
      active = false;
    };
  }, [enabled, refreshKey]);

  if (!enabled) {
    return (
      <section
        className="model-statistics dashboard-panel"
        id="statistics"
      >
        <h2>Model statistics</h2>
        <p className="statistics-message">
          Sign in to view your analysis statistics.
        </p>
      </section>
    );
  }

  if (loading && !statistics) {
    return (
      <section
        className="model-statistics dashboard-panel"
        id="statistics"
      >
        <h2>Model statistics</h2>
        <p className="statistics-message">
          Loading statistics…
        </p>
      </section>
    );
  }

  if (error || !statistics) {
    return (
      <section
        className="model-statistics dashboard-panel"
        id="statistics"
      >
        <h2>Model statistics</h2>
        <p className="statistics-error" role="alert">
          {error ?? "Statistics are unavailable."}
        </p>
      </section>
    );
  }

  const total = statistics.total_analyses;

 const currentBonafideProbability =
    currentResult?.bonafide_probability ?? 0;

  const currentSpoofProbability =
    currentResult?.spoof_probability ?? 0;

  const bonafideDegrees =
    currentBonafideProbability * 360;

  const maximumConfidenceCount = Math.max(
    1,
    ...statistics.confidence_distribution.map(
      (bucket) => bucket.count,
    ),
  );

  const maximumDailyCount = Math.max(
    1,
    ...statistics.daily_predictions.flatMap(
      (day) => [day.bonafide, day.spoof],
    ),
  );

  return (
    <section
      className="model-statistics dashboard-panel"
      id="statistics"
    >
      <div className="statistics-heading">
        <div>
          <span className="eyebrow">
            Performance and distribution
          </span>
          <h2>Model Statistics</h2>
          <p>
            Official evaluation results and your
            uploaded-audio predictions.
          </p>
        </div>

        <BrainCircuit size={29} />
      </div>

      <div className="statistics-cards">
        <article>
          <Activity size={21} />
          <span>Total analyses</span>
          <strong>{total}</strong>
        </article>

        <article>
          <Gauge size={21} />
          <span>Evaluation accuracy</span>
          <strong>
            {percentage(
              statistics.model_evaluation.accuracy,
            )}
          </strong>
        </article>

        <article>
          <UserRound size={21} />
          <span>Human detected</span>
          <strong>
            {statistics.bonafide_detections}
          </strong>
        </article>

        <article>
          <Bot size={21} />
          <span>AI detected</span>
          <strong>
            {statistics.spoof_detections}
          </strong>
        </article>
      </div>

      <div className="statistics-content">
          <article className="distribution-card">
          <div>
            <h3>Current audio prediction</h3>
            <p>
              Bonafide versus spoof probability for
              the latest analyzed recording
            </p>
          </div>

          {!currentResult ? (
            <p className="statistics-message">
              Analyze an audio file to view its
              probability distribution.
            </p>
          ) : (
            <div className="donut-layout">
              <div
                className="prediction-donut"
                role="img"
                aria-label={
                  `${percentage(
                    currentBonafideProbability,
                  )} bonafide and ` +
                  `${percentage(
                    currentSpoofProbability,
                  )} spoof`
                }
                style={{
                  background: `conic-gradient(
                    #24c58b 0deg
                    ${bonafideDegrees}deg,
                    #f0525f
                    ${bonafideDegrees}deg
                    360deg
                  )`,
                }}
              >
                <div>
                  <strong>
                    {percentage(
                      currentResult.confidence,
                    )}
                  </strong>
                  <span>confidence</span>
                </div>
              </div>

              <div className="distribution-legend">
                <div>
                  <i className="human-dot" />
                  <span>Bonafide probability</span>
                  <strong>
                    {percentage(
                      currentBonafideProbability,
                    )}
                  </strong>
                </div>

                <div>
                  <i className="ai-dot" />
                  <span>Spoof probability</span>
                  <strong>
                    {percentage(
                      currentSpoofProbability,
                    )}
                  </strong>
                </div>

                <div>
                  <i className="confidence-dot" />
                  <span>Recording</span>
                  <strong>
                    {currentResult.filename}
                  </strong>
                </div>
              </div>
            </div>
          )}
        </article>

        <article className="evaluation-card">
          <h3>Official model evaluation</h3>

          <dl>
            <div>
              <dt>Accuracy</dt>
              <dd>
                {percentage(
                  statistics.model_evaluation.accuracy,
                )}
              </dd>
            </div>

            <div>
              <dt>Precision</dt>
              <dd>
                {percentage(
                  statistics.model_evaluation.precision,
                )}
              </dd>
            </div>

            <div>
              <dt>Recall</dt>
              <dd>
                {percentage(
                  statistics.model_evaluation.recall,
                )}
              </dd>
            </div>

            <div>
              <dt>F1 score</dt>
              <dd>
                {percentage(
                  statistics.model_evaluation.f1_score,
                )}
              </dd>
            </div>

            <div>
              <dt>EER</dt>
              <dd>
                {percentage(
                  statistics.model_evaluation.eer,
                )}
              </dd>
            </div>
          </dl>
        </article>
      </div>
      <div className="statistics-trends">
        <article className="statistics-chart-card">
          <div className="chart-heading">
            <div>
              <h3>Confidence distribution</h3>
              <p>
                Completed analyses grouped by model
                confidence
              </p>
            </div>
          </div>

          <div className="confidence-chart">
            {statistics.confidence_distribution.map(
              (bucket) => (
                <div
                  className="confidence-chart-row"
                  key={bucket.range}
                >
                  <span>{bucket.range}</span>

                  <div className="confidence-chart-track">
                    <i
                      style={{
                        width:
                          `${(
                            bucket.count /
                            maximumConfidenceCount
                          ) * 100}%`,
                      }}
                    />
                  </div>

                  <strong>{bucket.count}</strong>
                </div>
              ),
            )}
          </div>
        </article>

        <article className="statistics-chart-card">
          <div className="chart-heading">
            <div>
              <h3>Daily predictions</h3>
              <p>
                Human and AI predictions during the
                last seven days
              </p>
            </div>

            <div className="daily-chart-legend">
              <span>
                <i className="human-dot" />
                Human
              </span>

              <span>
                <i className="ai-dot" />
                AI
              </span>
            </div>
          </div>

          <div className="daily-chart">
            {statistics.daily_predictions.map(
              (day) => (
                <div
                  className="daily-chart-column"
                  key={day.date}
                >
                  <div className="daily-bars">
                    <i
                      className="daily-human-bar"
                      title={`${day.bonafide} human`}
                      style={{
                        height:
                          `${(
                            day.bonafide /
                            maximumDailyCount
                          ) * 100}%`,
                      }}
                    />

                    <i
                      className="daily-ai-bar"
                      title={`${day.spoof} AI`}
                      style={{
                        height:
                          `${(
                            day.spoof /
                            maximumDailyCount
                          ) * 100}%`,
                      }}
                    />
                  </div>

                  <strong>
                    {day.bonafide + day.spoof}
                  </strong>

                  <span>
                    {new Intl.DateTimeFormat(
                      "en-IN",
                      {
                        weekday: "short",
                      },
                    ).format(new Date(day.date))}
                  </span>
                </div>
              ),
            )}
          </div>
        </article>
      </div>

      <div className="statistics-history">
        <h3>Recent server-saved analyses</h3>

        {history.length === 0 ? (
          <p className="statistics-message">
            No completed analyses yet.
          </p>
        ) : (
          <div className="statistics-table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Recording</th>
                  <th>Detection</th>
                  <th>Confidence</th>
                  <th>Date</th>
                </tr>
              </thead>

              <tbody>
                {history.slice(0, 8).map((entry) => (
                  <tr key={entry.id}>
                    <td>{entry.filename}</td>
                    <td>
                      <span
                        className={
                          `statistics-label ` +
                          entry.predicted_label
                        }
                      >
                        {entry.predicted_label ===
                        "bonafide"
                          ? "Human"
                          : "AI-generated"}
                      </span>
                    </td>
                    <td>
                      {percentage(entry.confidence)}
                    </td>
                    <td>
                      {new Intl.DateTimeFormat(
                        "en-IN",
                        {
                          day: "2-digit",
                          month: "short",
                          year: "numeric",
                        },
                      ).format(
                        new Date(entry.analyzed_at),
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <p className="statistics-disclaimer">
        Evaluation accuracy uses labelled test data.
        Uploaded-audio confidence is not the same as
        measured accuracy.
      </p>
    </section>
  );
}