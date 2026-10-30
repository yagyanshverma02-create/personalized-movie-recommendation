"""Link verified Gmail accounts to MovieMind's existing JSON profiles."""

from __future__ import annotations

import json
import hashlib
import hmac
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


PROFILE_PATH = Path(__file__).resolve().parent / "data" / "user_profiles.json"
ACCOUNT_DB_PATH = Path(__file__).resolve().parent / "data" / "accounts.sqlite3"
_GMAIL_PATTERN = r"^[a-zA-Z0-9._%+\-]+@gmail\.com$"
_PASSWORD_ITERATIONS = 310_000


def normalize_username(username: str) -> str | None:
    """Trim and validate a simple display name suitable for a demo profile."""
    if not isinstance(username, str):
        return None
    normalized = username.strip()
    if not normalized or len(normalized) > 40 or not normalized[0].isalnum():
        return None
    if any(not (character.isalnum() or character in " ._'-") for character in normalized):
        return None
    return normalized


def is_valid_gmail(email: str) -> bool:
    """Keep the demo login's existing @gmail.com validation behavior."""
    import re

    return isinstance(email, str) and bool(re.match(_GMAIL_PATTERN, email.strip().lower()))


def clear_auth_session(session_state) -> None:
    """Clear authentication and temporary OTP state without touching profiles."""
    for key in list(session_state.keys()):
        if key.startswith("_auth") or key.startswith("_otp"):
            del session_state[key]


def load_profiles(profile_path: Path = PROFILE_PATH) -> list[dict]:
    try:
        with open(profile_path, "r", encoding="utf-8") as profile_file:
            profiles = json.load(profile_file)
        return profiles if isinstance(profiles, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_profiles(profiles: list[dict], profile_path: Path = PROFILE_PATH) -> None:
    with open(profile_path, "w", encoding="utf-8") as profile_file:
        json.dump(profiles, profile_file, indent=2, ensure_ascii=False)


def _database_path(profile_path: Path) -> Path:
    """Keep test databases beside temporary profile files; production uses data/accounts.sqlite3."""
    profile_path = Path(profile_path)
    if profile_path == PROFILE_PATH:
        return ACCOUNT_DB_PATH
    return profile_path.parent / "accounts.sqlite3"


def _connect_account_db(profile_path: Path):
    database_path = _database_path(profile_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=10)
    connection.execute(
        """CREATE TABLE IF NOT EXISTS account_credentials (
            email TEXT PRIMARY KEY COLLATE NOCASE,
            password_salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            password_iterations INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        )"""
    )
    return connection


def _store_password_record(
    email: str,
    salt: str,
    password_hash: str,
    iterations: int,
    updated_at: str,
    profile_path: Path,
) -> None:
    with closing(_connect_account_db(profile_path)) as connection:
        with connection:
            connection.execute(
                """INSERT INTO account_credentials
                   (email, password_salt, password_hash, password_iterations, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(email) DO UPDATE SET
                     password_salt=excluded.password_salt,
                     password_hash=excluded.password_hash,
                     password_iterations=excluded.password_iterations,
                     updated_at=excluded.updated_at""",
                (email.strip().lower(), salt, password_hash, iterations, updated_at),
            )


def _migrate_legacy_passwords(profiles: list[dict], profile_path: Path) -> bool:
    """Move any earlier JSON password hashes into SQLite and remove them from JSON."""
    changed = False
    for profile in profiles:
        salt = profile.get("password_salt")
        password_hash = profile.get("password_hash")
        if not salt or not password_hash:
            continue
        email = str(profile.get("email", "")).strip().lower()
        if not email:
            continue
        updated_at = str(profile.get("updated_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"))
        try:
            iterations = int(profile.get("password_iterations", _PASSWORD_ITERATIONS))
        except (TypeError, ValueError):
            iterations = _PASSWORD_ITERATIONS
        _store_password_record(email, str(salt), str(password_hash), iterations, updated_at, profile_path)
        for key in ("password_salt", "password_hash", "password_iterations"):
            profile.pop(key, None)
        changed = True
    if changed:
        save_profiles(profiles, profile_path)
    return changed


def set_account_password(email: str, password: str, profile_path: Path = PROFILE_PATH) -> bool:
    """Store an email-keyed, salted PBKDF2 password hash in SQLite."""
    normalized_email = email.strip().lower()
    profiles = load_profiles(profile_path)
    try:
        _migrate_legacy_passwords(profiles, profile_path)
    except sqlite3.Error:
        return False
    profile = next((item for item in profiles if str(item.get("email", "")).strip().lower() == normalized_email), None)
    if profile is None or not isinstance(password, str) or not 8 <= len(password) <= 128:
        return False
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PASSWORD_ITERATIONS)
    updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        _store_password_record(
            normalized_email,
            salt.hex(),
            digest.hex(),
            _PASSWORD_ITERATIONS,
            updated_at,
            profile_path,
        )
    except sqlite3.Error:
        return False
    profile["updated_at"] = updated_at
    save_profiles(profiles, profile_path)
    return True


def verify_account_password(email: str, password: str, profile_path: Path = PROFILE_PATH) -> dict | None:
    """Return the matching profile when a password is correct, otherwise None."""
    normalized_email = email.strip().lower()
    profiles = load_profiles(profile_path)
    try:
        _migrate_legacy_passwords(profiles, profile_path)
    except sqlite3.Error:
        return None
    profile = next((item for item in profiles if str(item.get("email", "")).strip().lower() == normalized_email), None)
    if profile is None or not isinstance(password, str):
        return None
    try:
        with closing(_connect_account_db(profile_path)) as connection:
            record = connection.execute(
                """SELECT password_salt, password_hash, password_iterations
                   FROM account_credentials WHERE email = ? COLLATE NOCASE""",
                (normalized_email,),
            ).fetchone()
        if not record:
            return None
        salt = bytes.fromhex(record[0])
        expected = bytes.fromhex(record[1])
        iterations = int(record[2])
        if iterations < 100_000 or iterations > 2_000_000:
            return None
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    except (sqlite3.Error, TypeError, ValueError):
        return None
    return profile if hmac.compare_digest(actual, expected) else None


def get_or_create_account_profile(
    email: str,
    profile_path: Path = PROFILE_PATH,
    username: str | None = None,
) -> tuple[dict, list[dict]]:
    """Return the persistent profile for a verified email, creating it once."""
    normalized_email = email.strip().lower()
    profiles = load_profiles(profile_path)
    existing = next(
        (
            profile for profile in profiles
            if str(profile.get("email", "")).strip().lower() == normalized_email
        ),
        None,
    )
    if existing is not None:
        display_name = normalize_username(username) if username is not None else None
        if display_name and existing.get("name") != display_name:
            existing["name"] = display_name
            existing["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            save_profiles(profiles, profile_path)
        return existing, profiles

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    display_name = normalize_username(username) if username is not None else None
    display_name = display_name or normalized_email.split("@", 1)[0].replace(".", " ").title()
    profile_id = f"MM-{uuid.uuid4().hex[:6].upper()}"
    existing_ids = {profile.get("profile_id") for profile in profiles}
    while profile_id in existing_ids:
        profile_id = f"MM-{uuid.uuid4().hex[:6].upper()}"

    profile = {
        "profile_id": profile_id,
        "email": normalized_email,
        "name": display_name,
        "movies": [],
        "watch_later": [],
        "created_at": now,
        "updated_at": now,
    }
    profiles.append(profile)
    save_profiles(profiles, profile_path)
    return profile, profiles
