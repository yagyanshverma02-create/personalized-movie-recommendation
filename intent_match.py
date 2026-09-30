"""Movie-specific presentation score for verified current-viewing signals."""

from __future__ import annotations

import math


def get_intent_match_details(
    movie_genres,
    selected_genres=None,
    selected_mood=None,
    mood_genres_by_name=None,
    reference_similarity=None,
    reference_title=None,
):
    """Return an integer score and evidence-based explanation, or ``(None, None)``.

    Genre overlap uses F1 (harmonic mean of requested-genre recall and the share
    of catalog tags that match). Verified components use fixed 60/25/15
    genre/mood/reference weights, normalized over whichever signals are active.
    This score is for display only.
    """
    selected_genres = list(dict.fromkeys(selected_genres or []))
    requested_genres = set()
    for genre in selected_genres:
        if genre == "Rom-Com":
            requested_genres.update(("Comedy", "Romance"))
        else:
            requested_genres.add(str(genre))

    movie_genres = "" if movie_genres is None else movie_genres
    if isinstance(movie_genres, float) and math.isnan(movie_genres):
        movie_genres = ""
    actual_genres = {
        genre.strip()
        for genre in str(movie_genres or "").split("|")
        if genre.strip() and genre.strip() != "(no genres listed)"
    }
    signals = []
    explanations = []

    if requested_genres:
        matched_genres = requested_genres & actual_genres
        recall = len(matched_genres) / len(requested_genres)
        precision = len(matched_genres) / len(actual_genres) if actual_genres else 0.0
        genre_f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        signals.append((genre_f1, 0.60))
        display_genres = " + ".join(selected_genres)
        if matched_genres == requested_genres:
            explanations.append(f"Matches your {display_genres} intent")
        elif matched_genres:
            explanations.append(
                f"Matches {', '.join(sorted(matched_genres))} ({len(matched_genres)} of {len(requested_genres)} requested genres)"
            )
        else:
            explanations.append("No requested genre appears in its catalog metadata")

    mood_genres = set((mood_genres_by_name or {}).get(selected_mood, set())) if selected_mood else set()
    if mood_genres:
        matched_mood_genres = mood_genres & actual_genres
        mood_recall = len(matched_mood_genres) / len(mood_genres)
        mood_precision = len(matched_mood_genres) / len(actual_genres) if actual_genres else 0.0
        mood_f1 = (
            2 * mood_precision * mood_recall / (mood_precision + mood_recall)
            if mood_precision + mood_recall
            else 0.0
        )
        signals.append((mood_f1, 0.25))
        if matched_mood_genres:
            explanations.append(f"Local genre metadata supports {selected_mood}")
        else:
            explanations.append(f"No local genre support for {selected_mood}")

    try:
        reference_similarity = float(reference_similarity)
    except (TypeError, ValueError, OverflowError):
        reference_similarity = None
    if reference_similarity is not None and math.isfinite(reference_similarity):
        reference_similarity = min(1.0, max(0.0, reference_similarity))
        signals.append((reference_similarity, 0.15))
        if reference_title:
            explanations.append(
                f"{round(reference_similarity * 100)}% similar to reference movie {reference_title}"
            )
        else:
            explanations.append(
                f"{round(reference_similarity * 100)}% existing reference-movie similarity"
            )

    if not signals:
        return None, None
    active_weight = sum(weight for _, weight in signals)
    score = round(100 * sum(value * weight for value, weight in signals) / active_weight)
    return score, "; ".join(explanations)
