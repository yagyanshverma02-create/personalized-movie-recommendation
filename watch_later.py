"""Small profile-local Watch Later entry helpers."""

from __future__ import annotations


DEFAULT_WATCH_LATER_PROFILE_ID = "MM-WATCH-LATER"


def watch_later_key(entry):
    """Return a stable key for either a MovieLens catalog movie or a TMDB movie."""
    if not isinstance(entry, dict):
        return None
    if entry.get("movieId") is not None:
        try:
            movie_id = int(entry["movieId"])
            return f"movie:{movie_id}" if movie_id > 0 else None
        except (TypeError, ValueError, OverflowError):
            return None
    try:
        tmdb_id = int(entry.get("tmdbId"))
        return f"tmdb:{tmdb_id}" if tmdb_id > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def watch_later_movie_entry(movie):
    """Create a compact local-profile JSON entry from a rendered movie row."""
    if not isinstance(movie, dict):
        movie = movie.to_dict()
    if movie.get("_external_tmdb"):
        entry = {
            "tmdbId": int(movie["tmdbId"]),
            "title": str(movie.get("title", "")),
            "genres": str(movie.get("genres", "")),
            "release_date": str(movie.get("release_date", "")),
            "poster_path": movie.get("poster_path"),
            "backdrop_path": movie.get("backdrop_path"),
            "vote_average": movie.get("vote_average"),
            "external_tmdb": True,
        }
        return entry if watch_later_key(entry) else None
    try:
        movie_id = int(movie.get("movieId"))
    except (TypeError, ValueError, OverflowError):
        return None
    return {"movieId": movie_id} if movie_id > 0 else None


def toggle_watch_later_entry(entries, movie):
    """Toggle one movie, repairing duplicates while preserving list order."""
    entries = list(entries or [])
    new_entry = watch_later_movie_entry(movie)
    new_key = watch_later_key(new_entry)
    if not new_key:
        return entries, False

    saved = any(watch_later_key(entry) == new_key for entry in entries)
    if saved:
        result = [entry for entry in entries if watch_later_key(entry) != new_key]
        return result, False
    return entries + [new_entry], True
