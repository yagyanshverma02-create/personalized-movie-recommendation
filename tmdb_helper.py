"""Small, cached TMDB metadata client used only for presentation."""

from __future__ import annotations

from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
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
            timeout=1.8,
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


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _get_movie_trailer_cached(tmdb_id: int) -> dict[str, str] | None:
    """Choose a safe YouTube trailer listed for this movie in TMDB."""
    try:
        api_key = st.secrets.get("TMDB_API_KEY")
        if not api_key:
            return None
        response = requests.get(
            f"{TMDB_API_BASE}/movie/{tmdb_id}/videos",
            params={"api_key": api_key, "language": "en-US"},
            timeout=1.8,
        )
        response.raise_for_status()
        videos = response.json().get("results", [])
        if not isinstance(videos, list):
            return None
        youtube_trailers = [
            video for video in videos
            if isinstance(video, dict)
            and video.get("site") == "YouTube"
            and video.get("type") == "Trailer"
            and isinstance(video.get("key"), str)
            and re.fullmatch(r"[A-Za-z0-9_-]{6,20}", video["key"])
        ]
        if not youtube_trailers:
            return None
        youtube_trailers.sort(
            key=lambda video: (
                not bool(video.get("official")),
                video.get("iso_639_1") != "en",
                video.get("iso_3166_1") != "US",
            )
        )
        trailer = youtube_trailers[0]
        return {"key": trailer["key"], "name": str(trailer.get("name") or "Official trailer")}
    except Exception:
        return None


def get_movie_trailer(tmdb_id: Any) -> dict[str, str] | None:
    """Return the preferred YouTube trailer for a positive TMDB movie ID."""
    try:
        movie_id = int(tmdb_id)
        if movie_id <= 0:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return _get_movie_trailer_cached(movie_id)


def preload_movie_media(tmdb_ids: list[Any], include_trailers: bool = True) -> dict[int, dict[str, Any]]:
    """Warm cached metadata and trailer lookups concurrently for visible cards."""
    ids = set()
    for value in tmdb_ids or []:
        try:
            movie_id = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if movie_id > 0:
            ids.add(movie_id)
    results: dict[int, dict[str, Any]] = {movie_id: {} for movie_id in ids}
    if not ids:
        return results

    jobs = {}
    max_workers = min(16, len(ids) * (2 if include_trailers else 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for movie_id in ids:
            jobs[executor.submit(get_movie_details, movie_id)] = (movie_id, "details")
            if include_trailers:
                jobs[executor.submit(get_movie_trailer, movie_id)] = (movie_id, "trailer")
        for future in as_completed(jobs):
            movie_id, kind = jobs[future]
            try:
                results[movie_id][kind] = future.result()
            except Exception:
                results[movie_id][kind] = None
    return results


@st.cache_data(ttl=15 * 60, show_spinner=False)
def _get_tmdb_similar_movies_cached(tmdb_id: int) -> list[dict[str, Any]]:
    """Fetch TMDB's similar-movie list for an external catalog title."""
    try:
        api_key = st.secrets.get("TMDB_API_KEY")
        if not api_key:
            return []
        response = requests.get(
            f"{TMDB_API_BASE}/movie/{tmdb_id}/similar",
            params={"api_key": api_key, "language": "en-US", "page": 1},
            timeout=8,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not isinstance(results, list):
            return []
        similar = []
        for movie in results[:10]:
            if not isinstance(movie, dict):
                continue
            try:
                similar_id = int(movie.get("id"))
            except (TypeError, ValueError, OverflowError):
                continue
            if similar_id <= 0 or not movie.get("title"):
                continue
            similar.append({
                "tmdbId": similar_id,
                "title": str(movie.get("title", "")),
                "overview": str(movie.get("overview", "")),
                "poster_path": movie.get("poster_path"),
                "backdrop_path": movie.get("backdrop_path"),
                "release_date": str(movie.get("release_date", "")),
                "genre_ids": movie.get("genre_ids", []),
                "vote_average": movie.get("vote_average"),
            })
        return similar
    except Exception:
        return []


def get_tmdb_similar_movies(tmdb_id: Any) -> list[dict[str, Any]]:
    """Return cached TMDB similar movies for a positive movie ID."""
    try:
        movie_id = int(tmdb_id)
        if movie_id <= 0:
            return []
    except (TypeError, ValueError, OverflowError):
        return []
    return _get_tmdb_similar_movies_cached(movie_id)


@st.cache_data(ttl=15 * 60, show_spinner=False)
def _get_discover_movies_cached(category: str, page: int) -> list[dict[str, Any]]:
    """Fetch one page of a supported TMDB movie discovery category safely."""
    endpoints = {
        "Trending": ("trending/movie/week", {}),
        "Popular": ("movie/popular", {}),
        "Now Playing": ("movie/now_playing", {}),
        "Latest": (
            "discover/movie",
            {
                "sort_by": "primary_release_date.desc",
                "release_date.lte": date.today().isoformat(),
                "include_adult": "false",
            },
        ),
        "Upcoming": ("movie/upcoming", {}),
        "Highest Rated": ("movie/top_rated", {}),
    }
    if category not in endpoints:
        return []
    try:
        api_key = st.secrets.get("TMDB_API_KEY")
        if not api_key:
            return []
        endpoint, category_params = endpoints[category]
        response = requests.get(
            f"{TMDB_API_BASE}/{endpoint}",
            params={
                "api_key": api_key,
                "language": "en-US",
                "page": max(1, int(page)),
                **category_params,
            },
            timeout=8,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not isinstance(results, list):
            return []
        movies = []
        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                tmdb_id = int(item.get("id"))
            except (TypeError, ValueError, OverflowError):
                continue
            if tmdb_id <= 0 or not item.get("title"):
                continue
            genre_ids = item.get("genre_ids", [])
            if not isinstance(genre_ids, list):
                genre_ids = []
            movies.append({
                "tmdbId": tmdb_id,
                "title": str(item.get("title") or ""),
                "overview": str(item.get("overview") or ""),
                "poster_path": item.get("poster_path"),
                "backdrop_path": item.get("backdrop_path"),
                "release_date": str(item.get("release_date") or ""),
                "genre_ids": genre_ids,
                "vote_average": item.get("vote_average"),
            })
        return movies
    except Exception:
        return []


def get_discover_movies(category: str, page: int = 1) -> list[dict[str, Any]]:
    """Return cached TMDB-backed movies for one of MovieMind's supported categories."""
    try:
        page_number = int(page)
    except (TypeError, ValueError, OverflowError):
        page_number = 1
    return _get_discover_movies_cached(str(category), max(1, page_number))


TMDB_MOVIE_GENRES = {
    "Action": 28, "Comedy": 35, "Drama": 18, "Romance": 10749,
    "Science Fiction": 878, "Thriller": 53, "Horror": 27,
    "Animation": 16, "Crime": 80,
}


@st.cache_data(ttl=15 * 60, show_spinner=False)
def _get_tmdb_genre_movies_cached(genre_id: int, page: int) -> tuple[list[dict[str, Any]], int]:
    """Fetch a page of TMDB movies for one supported genre."""
    try:
        api_key = st.secrets.get("TMDB_API_KEY")
        if not api_key:
            return [], 1
        response = requests.get(
            f"{TMDB_API_BASE}/discover/movie",
            params={
                "api_key": api_key,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "with_genres": genre_id,
                "include_adult": "false",
                "page": max(1, min(6, int(page))),
            },
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        if not isinstance(results, list):
            return [], 1
        movies = []
        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                tmdb_id = int(item.get("id"))
            except (TypeError, ValueError, OverflowError):
                continue
            if tmdb_id <= 0 or not item.get("title"):
                continue
            genre_ids = item.get("genre_ids", [])
            movies.append({
                "tmdbId": tmdb_id,
                "title": str(item.get("title") or ""),
                "overview": str(item.get("overview") or ""),
                "poster_path": item.get("poster_path"),
                "backdrop_path": item.get("backdrop_path"),
                "release_date": str(item.get("release_date") or ""),
                "genre_ids": genre_ids if isinstance(genre_ids, list) else [],
                "vote_average": item.get("vote_average"),
            })
        total_pages = max(1, min(6, int(payload.get("total_pages", 1) or 1)))
        return movies, total_pages
    except Exception:
        return [], 1


def get_tmdb_genre_movies(genre: str, page: int = 1) -> tuple[list[dict[str, Any]], int]:
    """Return popular TMDB movies for a supported genre and page number."""
    genre_id = TMDB_MOVIE_GENRES.get(str(genre))
    if not genre_id:
        return [], 1
    try:
        page_number = max(1, min(6, int(page)))
    except (TypeError, ValueError, OverflowError):
        page_number = 1
    return _get_tmdb_genre_movies_cached(genre_id, page_number)


def get_movie_poster_url(poster_path: Any, size: str = "w500") -> str | None:
    """Build an image URL only when TMDB returned a poster path."""
    if not isinstance(poster_path, str) or not re.match(r"^/[A-Za-z0-9_.\-]+\.(?:jpe?g|png|webp)$", poster_path, re.IGNORECASE):
        return None
    return f"{TMDB_IMAGE_BASE}/{size}{poster_path}"


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _get_movie_poster_cached(poster_path: str, size: str) -> bytes | None:
    """Fetch and validate image bytes so clients never render a broken URL."""
    poster_url = get_movie_poster_url(poster_path, size)
    if not poster_url:
        return None
    try:
        response = requests.get(poster_url, timeout=12, allow_redirects=True)
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
