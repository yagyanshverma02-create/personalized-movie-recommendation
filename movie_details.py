"""Shared normalization for MovieMind's selected movie details state."""

from __future__ import annotations


def normalize_movie_for_details(movie):
    """Normalize recommendation, catalog, and TMDB rows for the shared details view."""
    if hasattr(movie, "to_dict"):
        movie = movie.to_dict()
    movie = dict(movie or {})
    tmdb_id = movie.get("tmdbId")
    movie_id = movie.get("movieId")
    try:
        tmdb_id = int(tmdb_id) if tmdb_id is not None else None
    except (TypeError, ValueError, OverflowError):
        tmdb_id = None
    try:
        movie_id = int(movie_id) if movie_id is not None else None
    except (TypeError, ValueError, OverflowError):
        movie_id = None
    return {
        "movieId": movie_id,
        "tmdbId": tmdb_id,
        "imdbId": movie.get("imdbId"),
        "title": str(movie.get("title") or "Movie details"),
        "genres": str(movie.get("genres") or ""),
        "release_date": str(movie.get("release_date") or ""),
        "poster_path": movie.get("poster_path"),
        "backdrop_path": movie.get("backdrop_path"),
        "_external_tmdb": bool(movie.get("_external_tmdb")) or (movie_id is not None and movie_id < 0),
        "_recommendation": movie if "hybrid_score" in movie else None,
    }


def select_movie_details(session_state, movie):
    """Set the one canonical selected movie state consumed by MovieMind details."""
    normalized = normalize_movie_for_details(movie)
    session_state["selected_movie_details"] = normalized
    return normalized
