"""Local user authentication for AcousticSpace."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Cookie, HTTPException, Response, status


BACKEND_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = Path(
    os.getenv(
        "ACOUSTICSPACE_DATABASE_PATH",
        str(BACKEND_ROOT / "data" / "acousticspace.db"),
    )
).expanduser()

SESSION_COOKIE_NAME = "acousticspace_session"
SESSION_LIFETIME_HOURS = 12
PASSWORD_ITERATIONS = 600_000

EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)


def _connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=10,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def initialize_auth_database() -> None:
    """Create authentication tables when they do not exist."""

    with _connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            index_sessions_user_id
            ON sessions(user_id)
            """
        )


def normalize_email(email: str) -> str:
    normalized = email.strip().lower()

    if len(normalized) > 254 or not EMAIL_PATTERN.fullmatch(normalized):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Enter a valid email address.",
        )

    return normalized


def validate_display_name(display_name: str) -> str:
    normalized = " ".join(display_name.strip().split())

    if len(normalized) < 2 or len(normalized) > 60:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Display name must contain 2 to 60 characters.",
        )

    return normalized


def validate_password(password: str) -> None:
    if len(password) < 8:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must contain at least 8 characters.",
        )

    if len(password) > 128:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password cannot exceed 128 characters.",
        )

    if not any(character.isalpha() for character in password):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must contain at least one letter.",
        )

    if not any(character.isdigit() for character in password):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must contain at least one number.",
        )


def _hash_password(password: str, salt: bytes) -> str:
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )

    return password_hash.hex()


def _public_user(user: sqlite3.Row) -> dict:
    return {
        "id": user["id"],
        "email": user["email"],
        "display_name": user["display_name"],
        "created_at": user["created_at"],
    }


def register_user(
    email: str,
    display_name: str,
    password: str,
) -> dict:
    initialize_auth_database()

    normalized_email = normalize_email(email)
    normalized_name = validate_display_name(display_name)
    validate_password(password)

    salt = secrets.token_bytes(32)
    password_hash = _hash_password(password, salt)
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        with _connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (
                    email,
                    display_name,
                    password_hash,
                    password_salt,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    normalized_email,
                    normalized_name,
                    password_hash,
                    salt.hex(),
                    created_at,
                ),
            )

            user = connection.execute(
                """
                SELECT id, email, display_name, created_at
                FROM users
                WHERE id = ?
                """,
                (cursor.lastrowid,),
            ).fetchone()
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        ) from exc

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The account could not be created.",
        )

    return _public_user(user)


def verify_credentials(email: str, password: str) -> dict:
    initialize_auth_database()

    normalized_email = normalize_email(email)

    with _connection() as connection:
        user = connection.execute(
            """
            SELECT
                id,
                email,
                display_name,
                password_hash,
                password_salt,
                created_at
            FROM users
            WHERE email = ?
            """,
            (normalized_email,),
        ).fetchone()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    candidate_hash = _hash_password(
        password,
        bytes.fromhex(user["password_salt"]),
    )

    if not hmac.compare_digest(
        candidate_hash,
        user["password_hash"],
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    return _public_user(user)


def create_session(user_id: int) -> str:
    initialize_auth_database()

    token = secrets.token_urlsafe(48)
    token_hash = hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(
        hours=SESSION_LIFETIME_HOURS
    )

    with _connection() as connection:
        connection.execute(
            """
            DELETE FROM sessions
            WHERE expires_at <= ?
            """,
            (now.isoformat(),),
        )

        connection.execute(
            """
            INSERT INTO sessions (
                token_hash,
                user_id,
                expires_at,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                token_hash,
                user_id,
                expires_at.isoformat(),
                now.isoformat(),
            ),
        )

    return token


def set_session_cookie(response: Response, token: str) -> None:
    secure_cookie = (
        os.getenv(
            "ACOUSTICSPACE_SECURE_COOKIES",
            "false",
        ).lower()
        == "true"
    )

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_LIFETIME_HOURS * 60 * 60,
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        path="/",
    )


def delete_session(token: str | None) -> None:
    if not token:
        return

    token_hash = hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()

    initialize_auth_database()

    with _connection() as connection:
        connection.execute(
            """
            DELETE FROM sessions
            WHERE token_hash = ?
            """,
            (token_hash,),
        )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
    )


def require_authenticated_user(
    acousticspace_session: str | None = Cookie(
        default=None,
        alias=SESSION_COOKIE_NAME,
    ),
) -> dict:
    if not acousticspace_session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required.",
        )

    token_hash = hashlib.sha256(
        acousticspace_session.encode("utf-8")
    ).hexdigest()

    now = datetime.now(timezone.utc)

    initialize_auth_database()

    with _connection() as connection:
        user = connection.execute(
            """
            SELECT
                users.id,
                users.email,
                users.display_name,
                users.created_at,
                sessions.expires_at
            FROM sessions
            JOIN users
                ON users.id = sessions.user_id
            WHERE sessions.token_hash = ?
            """,
            (token_hash,),
        ).fetchone()

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="The session is invalid.",
            )

        expires_at = datetime.fromisoformat(
            user["expires_at"]
        )

        if expires_at <= now:
            connection.execute(
                """
                DELETE FROM sessions
                WHERE token_hash = ?
                """,
                (token_hash,),
            )

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="The session has expired.",
            )

    return _public_user(user)