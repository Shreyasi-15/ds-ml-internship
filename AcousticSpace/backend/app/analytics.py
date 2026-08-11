"""Persistent per-user analysis history and statistics."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from app.auth import DATABASE_PATH, initialize_auth_database


HISTORY_RETENTION_DAYS = 30

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
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_analytics_database() -> None:
    """Create and safely migrate the analysis-history table."""
    initialize_auth_database()
    with _connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                predicted_label TEXT NOT NULL
                    CHECK (predicted_label IN ('bonafide', 'spoof')),
                confidence REAL NOT NULL
                    CHECK (confidence >= 0 AND confidence <= 1),
                bonafide_probability REAL NOT NULL,
                spoof_probability REAL NOT NULL,
                model_version TEXT NOT NULL,
                analyzed_at TEXT NOT NULL,
                retained INTEGER NOT NULL DEFAULT 0
                    CHECK (retained IN (0, 1)),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(analyses)")
        }
        if "retained" not in columns:
            connection.execute(
                "ALTER TABLE analyses ADD COLUMN retained INTEGER NOT NULL DEFAULT 0"
            )
        connection.execute(
            """CREATE INDEX IF NOT EXISTS index_analyses_user_date
            ON analyses(user_id, analyzed_at DESC)"""
        )


def purge_expired_analyses(user_id: int | None = None) -> int:
    """Delete unkept records older than the 30-day retention window."""
    initialize_analytics_database()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=HISTORY_RETENTION_DAYS)).isoformat()
    with _connection() as connection:
        if user_id is None:
            cursor = connection.execute(
                "DELETE FROM analyses WHERE retained = 0 AND analyzed_at < ?", (cutoff,)
            )
        else:
            cursor = connection.execute(
                """DELETE FROM analyses
                WHERE user_id = ? AND retained = 0 AND analyzed_at < ?""",
                (user_id, cutoff),
            )
    return cursor.rowcount


def save_analysis(user_id: int, filename: str, prediction: dict) -> dict:
    initialize_analytics_database()
    analyzed_at = datetime.now(timezone.utc).isoformat()
    with _connection() as connection:
        cursor = connection.execute(
            """INSERT INTO analyses (
                user_id, filename, predicted_label, confidence,
                bonafide_probability, spoof_probability, model_version,
                analyzed_at, retained
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)""",
            (
                user_id, filename, prediction["predicted_label"],
                float(prediction["confidence"]),
                float(prediction["bonafide_probability"]),
                float(prediction["spoof_probability"]),
                prediction["model_version"], analyzed_at,
            ),
        )
    return {
        "id": cursor.lastrowid, "filename": filename,
        "predicted_label": prediction["predicted_label"],
        "confidence": float(prediction["confidence"]),
        "model_version": prediction["model_version"],
        "analyzed_at": analyzed_at, "retained": False,
    }


def _history(user_id: int, retained: bool, limit: int = 50) -> list[dict]:
    purge_expired_analyses(user_id)
    safe_limit = max(1, min(limit, 100))
    with _connection() as connection:
        rows = connection.execute(
            """SELECT id, filename, predicted_label, confidence,
                model_version, analyzed_at, retained
            FROM analyses WHERE user_id = ? AND retained = ?
            ORDER BY analyzed_at DESC LIMIT ?""",
            (user_id, int(retained), safe_limit),
        ).fetchall()
    return [{**dict(row), "retained": bool(row["retained"])} for row in rows]


def get_recent_analyses(user_id: int, limit: int = 50) -> list[dict]:
    return _history(user_id, False, limit)


def get_saved_analyses(user_id: int, limit: int = 100) -> list[dict]:
    return _history(user_id, True, limit)


def retain_analysis(user_id: int, analysis_id: int) -> bool:
    purge_expired_analyses(user_id)
    with _connection() as connection:
        cursor = connection.execute(
            "UPDATE analyses SET retained = 1 WHERE id = ? AND user_id = ?",
            (analysis_id, user_id),
        )
    return cursor.rowcount == 1


def delete_analysis(user_id: int, analysis_id: int) -> bool:
    with _connection() as connection:
        cursor = connection.execute(
            "DELETE FROM analyses WHERE id = ? AND user_id = ?",
            (analysis_id, user_id),
        )
    return cursor.rowcount == 1


def get_analysis_history(user_id: int, limit: int = 50) -> list[dict]:
    """Compatibility alias for callers that need recent unkept records."""
    return get_recent_analyses(user_id, limit)


def get_user_statistics(user_id: int) -> dict:
    purge_expired_analyses(user_id)
    with _connection() as connection:
        totals = connection.execute(
            """SELECT COUNT(*) AS total_analyses,
            SUM(CASE WHEN predicted_label = 'bonafide' THEN 1 ELSE 0 END)
                AS bonafide_detections,
            SUM(CASE WHEN predicted_label = 'spoof' THEN 1 ELSE 0 END)
                AS spoof_detections,
            AVG(confidence) AS average_confidence
            FROM analyses WHERE user_id = ?""", (user_id,),
        ).fetchone()
        rows = connection.execute(
            """SELECT predicted_label, confidence, analyzed_at FROM analyses
            WHERE user_id = ? ORDER BY analyzed_at ASC""", (user_id,),
        ).fetchall()
    distribution = [
        {"range": label, "count": 0}
        for label in ("50–60%", "60–70%", "70–80%", "80–90%", "90–100%")
    ]
    today = datetime.now(timezone.utc).date()
    dates = [today - timedelta(days=offset) for offset in range(6, -1, -1)]
    daily = {
        day.isoformat(): {"date": day.isoformat(), "bonafide": 0, "spoof": 0}
        for day in dates
    }
    for row in rows:
        confidence = float(row["confidence"])
        index = 0 if confidence < .6 else 1 if confidence < .7 else 2 if confidence < .8 else 3 if confidence < .9 else 4
        distribution[index]["count"] += 1
        date = datetime.fromisoformat(row["analyzed_at"]).date().isoformat()
        if date in daily:
            daily[date][row["predicted_label"]] += 1
    return {
        "total_analyses": totals["total_analyses"],
        "bonafide_detections": totals["bonafide_detections"] or 0,
        "spoof_detections": totals["spoof_detections"] or 0,
        "average_confidence": float(totals["average_confidence"] or 0),
        "model_evaluation": MODEL_EVALUATION,
        "confidence_distribution": distribution,
        "daily_predictions": [daily[day.isoformat()] for day in dates],
    }
