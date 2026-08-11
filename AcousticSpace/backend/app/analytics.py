"""Persistent per-user analysis history and statistics."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from app.auth import (
    DATABASE_PATH,
    initialize_auth_database,
)


MODEL_EVALUATION = {
    "accuracy": 0.99,
    "precision": 0.9939516129032258,
    "recall": 0.986,
    "f1_score": 0.9899598393574297,
    "eer": 0.009000000000000005,
    "evaluated_recordings": 1000,
}


def _connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=10,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def initialize_analytics_database() -> None:
    """Create the analysis-history table."""

    initialize_auth_database()

    with _connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                predicted_label TEXT NOT NULL
                    CHECK (
                        predicted_label IN (
                            'bonafide',
                            'spoof'
                        )
                    ),
                confidence REAL NOT NULL
                    CHECK (
                        confidence >= 0
                        AND confidence <= 1
                    ),
                bonafide_probability REAL NOT NULL,
                spoof_probability REAL NOT NULL,
                model_version TEXT NOT NULL,
                analyzed_at TEXT NOT NULL,
                FOREIGN KEY (user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            index_analyses_user_date
            ON analyses(user_id, analyzed_at DESC)
            """
        )


def save_analysis(
    user_id: int,
    filename: str,
    prediction: dict,
) -> dict:
    """Save one completed prediction."""

    initialize_analytics_database()

    analyzed_at = datetime.now(timezone.utc).isoformat()

    with _connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO analyses (
                user_id,
                filename,
                predicted_label,
                confidence,
                bonafide_probability,
                spoof_probability,
                model_version,
                analyzed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                filename,
                prediction["predicted_label"],
                float(prediction["confidence"]),
                float(prediction["bonafide_probability"]),
                float(prediction["spoof_probability"]),
                prediction["model_version"],
                analyzed_at,
            ),
        )

        analysis_id = cursor.lastrowid

    return {
        "id": analysis_id,
        "filename": filename,
        "predicted_label": prediction["predicted_label"],
        "confidence": float(prediction["confidence"]),
        "model_version": prediction["model_version"],
        "analyzed_at": analyzed_at,
    }


def get_analysis_history(
    user_id: int,
    limit: int = 50,
) -> list[dict]:
    """Return recent analyses belonging to one user."""

    initialize_analytics_database()

    safe_limit = max(1, min(limit, 100))

    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                filename,
                predicted_label,
                confidence,
                model_version,
                analyzed_at
            FROM analyses
            WHERE user_id = ?
            ORDER BY analyzed_at DESC
            LIMIT ?
            """,
            (user_id, safe_limit),
        ).fetchall()

    return [dict(row) for row in rows]


def get_user_statistics(user_id: int) -> dict:
    """Calculate aggregate prediction statistics."""

    initialize_analytics_database()

    with _connection() as connection:
        totals = connection.execute(
            """
            SELECT
                COUNT(*) AS total_analyses,
                SUM(
                    CASE
                        WHEN predicted_label = 'bonafide'
                        THEN 1
                        ELSE 0
                    END
                ) AS bonafide_detections,
                SUM(
                    CASE
                        WHEN predicted_label = 'spoof'
                        THEN 1
                        ELSE 0
                    END
                ) AS spoof_detections,
                AVG(confidence) AS average_confidence
            FROM analyses
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()

        rows = connection.execute(
            """
            SELECT
                predicted_label,
                confidence,
                analyzed_at
            FROM analyses
            WHERE user_id = ?
            ORDER BY analyzed_at ASC
            """,
            (user_id,),
        ).fetchall()

    confidence_distribution = [
        {"range": "50–60%", "count": 0},
        {"range": "60–70%", "count": 0},
        {"range": "70–80%", "count": 0},
        {"range": "80–90%", "count": 0},
        {"range": "90–100%", "count": 0},
    ]

    today = datetime.now(timezone.utc).date()

    dates = [
        today - timedelta(days=offset)
        for offset in range(6, -1, -1)
    ]

    daily_map = {
        day.isoformat(): {
            "date": day.isoformat(),
            "bonafide": 0,
            "spoof": 0,
        }
        for day in dates
    }

    for row in rows:
        confidence = float(row["confidence"])

        if confidence < 0.6:
            bucket_index = 0
        elif confidence < 0.7:
            bucket_index = 1
        elif confidence < 0.8:
            bucket_index = 2
        elif confidence < 0.9:
            bucket_index = 3
        else:
            bucket_index = 4

        confidence_distribution[
            bucket_index
        ]["count"] += 1

        analyzed_date = datetime.fromisoformat(
            row["analyzed_at"]
        ).date().isoformat()

        if analyzed_date in daily_map:
            daily_map[analyzed_date][
                row["predicted_label"]
            ] += 1

    return {
        "total_analyses": totals["total_analyses"],
        "bonafide_detections": (
            totals["bonafide_detections"] or 0
        ),
        "spoof_detections": (
            totals["spoof_detections"] or 0
        ),
        "average_confidence": float(
            totals["average_confidence"] or 0
        ),
        "model_evaluation": MODEL_EVALUATION,
        "confidence_distribution": (
            confidence_distribution
        ),
        "daily_predictions": [
            daily_map[day.isoformat()]
            for day in dates
        ],
    }
