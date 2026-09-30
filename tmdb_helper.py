"""Small, cached TMDB metadata client used only for presentation."""

from __future__ import annotations

from io import BytesIO
import re
from typing import Any

import requests
import streamlit as st
from PIL import Image

TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"
POSTER_PLACEHOLDER = """<svg xmlns="http://www.w3.org/2000/svg" width="300" height="450" viewBox="0 0 300 450">
<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#171b22"/><stop offset="1" stop-color="#0b0d10"/></linearGradient></defs>
<rect width="300" height="450" fill="url(#bg)"/><rect x="1" y="1" width="298" height="448" rx="10" fill="none" stroke="#343a46" stroke-width="2"/>
<g fill="none" stroke="#8e98a8" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" opacity=".8"><path d="M111 182h78a13 13 0 0 1 13 13v59a13 13 0 0 1-13 13h-78a13 13 0 0 1-13-13v-59a13 13 0 0 1 13-13Z"/><path d="m130 207 39 24-39 24z"/></g>
<text x="150" y="320" text-anchor="middle" fill="#f3f4f6" font-family="Arial,sans-serif" font-size="20" font-weight="700">MOVIEMIND</text><text x="150" y="349" text-anchor="middle" fill="#9ca3af" font-family="Arial,sans-serif" font-size="14">Poster unavailable</text></svg>"""


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _get_movie_details_cached(tmdb_id: int) -> dict[str, Any] | None:
    """Fetch metadata once per TMDB ID; failures safely leave the app usable."""
    try:
        api_key = st.secrets.get("TMDB_API_KEY")
        if not api_key:
            return None
        response = requests.get(
            f"{TMDB_API_BASE}/movie/{tmdb_id}",
            params={"api_key": api_key, "language": "en-US"},
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            return None
        # Keep only presentation fields; local MovieMind data remains authoritative.
        return {
            "overview": data.get("overview") or "",
            "poster_path": data.get("poster_path"),
            "backdrop_path": data.get("backdrop_path"),
            "release_date": data.get("release_date") or "",
        }
    except Exception:
        return None


def get_movie_details(tmdb_id: Any) -> dict[str, Any] | None:
    """Return cached TMDB presentation metadata for a positive numeric ID."""
    try:
        if tmdb_id is None:
            return None
        movie_id = int(tmdb_id)
        if movie_id <= 0:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return _get_movie_details_cached(movie_id)


def get_movie_poster_url(poster_path: Any, size: str = "w500") -> str | None:
    """Build an image URL only when TMDB returned a poster path."""
    if not isinstance(poster_path, str) or not re.fullmatch(r"/[A-Za-z0-9_-]+\.(?:jpe?g|png|webp)", poster_path, re.IGNORECASE):
        return None
    return f"{TMDB_IMAGE_BASE}/{size}{poster_path}"


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _get_movie_poster_cached(poster_path: str, size: str) -> bytes | None:
    """Fetch and validate image bytes so clients never render a broken URL."""
    poster_url = get_movie_poster_url(poster_path, size)
    if not poster_url:
        return None
    try:
        response = requests.get(poster_url, timeout=8)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
        image_bytes = response.content
        if not content_type.startswith("image/") or not image_bytes or len(image_bytes) > 10 * 1024 * 1024:
            return None
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
        return image_bytes
    except Exception:
        return None


def get_movie_poster_image(poster_path: Any, size: str = "w500") -> bytes | None:
    """Return cached, validated TMDB image bytes; otherwise let the UI fall back."""
    if not get_movie_poster_url(poster_path, size):
        return None
    return _get_movie_poster_cached(poster_path, size)


def get_movie_backdrop_image(backdrop_path: Any, size: str = "w1280") -> bytes | None:
    """Return a cached, validated TMDB backdrop using the shared image fetcher."""
    if not get_movie_poster_url(backdrop_path, size):
        return None
    return _get_movie_poster_cached(backdrop_path, size)
