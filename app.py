import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from scipy.sparse import csr_matrix, lil_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from surprise import Dataset, Reader, SVD

PROFILE_PATH = Path(__file__).resolve().parent / "data" / "user_profiles.json"
CONTEXT_WEIGHT = 0.15


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MovieMind",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main background */

[data-testid="stAppViewContainer"] {
    background-color: #0b0d10;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #101216;
}

/* Hero */

.hero-box {
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 28px;
    background: linear-gradient(
        135deg,
        rgba(185, 28, 28, 0.18),
        rgba(17, 20, 25, 0.98)
    );
    border: 1px solid rgba(255,255,255,0.08);
}

.hero-title {
    font-size: 44px;
    font-weight: 750;
    letter-spacing: -1.5px;
}

.hero-subtitle {
    color: #9ca3af;
    font-size: 16px;
    margin-top: 5px;
}

.hero-description {
    color: #d1d5db;
    font-size: 14px;
    max-width: 780px;
    line-height: 1.7;
    margin-top: 15px;
}

/* Section titles */

.section-title {
    font-size: 26px;
    font-weight: 700;
}

.section-description {
    color: #9ca3af;
    font-size: 14px;
    margin-bottom: 18px;
}

/* Stat cards */

.stat-box {
    padding: 18px;
    border-radius: 14px;
    background-color: #111419;
    border: 1px solid rgba(255,255,255,0.07);
}

.stat-label {
    color: #8b929e;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.stat-value {
    font-size: 25px;
    font-weight: 700;
    margin-top: 7px;
}

/* Recommendation cards */

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 15px;
    border-color: rgba(255,255,255,0.08);
    background-color: #111419;
}

/* Buttons */

.stButton > button {
    border-radius: 9px;
    min-height: 42px;
    font-weight: 600;
}

/* Tabs */

button[data-baseweb="tab"] {
    font-weight: 600;
}

/* Progress */

div[data-testid="stProgress"] {
    margin-top: 4px;
    margin-bottom: 8px;
}

/* Metrics */

div[data-testid="stMetric"] {
    background-color: #111419;
    border-radius: 12px;
    padding: 12px;
}

/* Expanders */

div[data-testid="stExpander"] {
    border-radius: 10px;
    border-color: rgba(255,255,255,0.07);
}

/* Footer */

.footer-text {
    color: #6b7280;
    font-size: 12px;
    text-align: center;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():

    ratings = pd.read_csv(
        "data/ratings_clean.csv"
    )

    movies = pd.read_csv(
        "data/movies_clean.csv"
    )

    links = pd.read_csv(
        "data/links.csv"
    )

    return ratings, movies, links


ratings, movies, links = load_data()


# ============================================================
# MOVIE LOOKUP
# ============================================================

movie_lookup = (
    movies
    .merge(
        links,
        on="movieId",
        how="left"
    )
    .set_index("movieId")
)


# ============================================================
# MOVIE OPTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def create_movie_options(movies):

    title_counts = (
        movies["title"]
        .value_counts()
    )

    movie_label_to_id = {}

    for _, row in movies.sort_values(
        "title"
    ).iterrows():

        movie_id = int(
            row["movieId"]
        )

        title = str(
            row["title"]
        )

        if title_counts[title] > 1:
            label = (
                f"{title} | Movie ID {movie_id}"
            )
        else:
            label = title

        movie_label_to_id[
            label
        ] = movie_id

    return movie_label_to_id


movie_label_to_id = (
    create_movie_options(movies)
)


# ============================================================
# PREPARE SPARSE MATRIX
# ============================================================

@st.cache_resource(show_spinner=False)
def prepare_matrices(_ratings):

    user_ids = np.sort(
        _ratings["userId"].unique()
    )

    movie_ids = np.sort(
        _ratings["movieId"].unique()
    )

    user_to_index = {
        int(user_id): index
        for index, user_id
        in enumerate(user_ids)
    }

    movie_to_index = {
        int(movie_id): index
        for index, movie_id
        in enumerate(movie_ids)
    }

    index_to_movie = {
        index: int(movie_id)
        for movie_id, index
        in movie_to_index.items()
    }

    user_indices = (
        _ratings["userId"]
        .map(user_to_index)
        .to_numpy(dtype=np.int32)
    )

    movie_indices = (
        _ratings["movieId"]
        .map(movie_to_index)
        .to_numpy(dtype=np.int32)
    )

    rating_values = (
        _ratings["rating"]
        .to_numpy(dtype=np.float32)
    )

    matrix = csr_matrix(
        (
            rating_values,
            (
                user_indices,
                movie_indices
            )
        ),
        shape=(
            len(user_ids),
            len(movie_ids)
        ),
        dtype=np.float32
    )

    return (
        matrix,
        user_ids,
        movie_ids,
        user_to_index,
        movie_to_index,
        index_to_movie
    )


(
    user_movie_matrix,
    user_ids,
    movie_ids,
    user_to_index,
    movie_to_index,
    index_to_movie
) = prepare_matrices(ratings)


# ============================================================
# USER MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def train_user_model(_matrix):

    n_neighbors = min(
        21,
        _matrix.shape[0]
    )

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=n_neighbors
    )

    model.fit(_matrix)

    return model


# ============================================================
# ITEM MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def train_item_model(_matrix):

    movie_user_matrix = (
        _matrix.T.tocsr()
    )

    normalized_matrix = normalize(
        movie_user_matrix,
        axis=1
    )

    n_neighbors = min(
        21,
        normalized_matrix.shape[0]
    )

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=n_neighbors
    )

    model.fit(
        normalized_matrix
    )

    return (
        model,
        normalized_matrix
    )


# ============================================================
# SVD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def train_svd_model(_ratings):

    sample_size = min(
        1_000_000,
        len(_ratings)
    )

    sample = _ratings.sample(
        n=sample_size,
        random_state=42
    )

    reader = Reader(
        rating_scale=(0.5, 5.0)
    )

    data = Dataset.load_from_df(
        sample[
            [
                "userId",
                "movieId",
                "rating"
            ]
        ],
        reader
    )

    trainset = (
        data.build_full_trainset()
    )

    model = SVD(
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42
    )

    model.fit(trainset)

    return model


# ============================================================
# TRAIN MODELS
# ============================================================

with st.spinner(
    "Preparing recommendation models..."
):

    user_model = train_user_model(
        user_movie_matrix
    )

    (
        item_model,
        normalized_item_matrix
    ) = train_item_model(
        user_movie_matrix
    )

    svd_model = train_svd_model(
        ratings
    )


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def normalize_scores(scores):

    if not scores:
        return {}

    scores = {
        key: float(value)
        for key, value in scores.items()
        if np.isfinite(value)
    }
    if not scores:
        return {}

    values = np.array(list(scores.values()), dtype=float)

    minimum = values.min()
    maximum = values.max()

    if np.isclose(
        minimum,
        maximum
    ):

        return {
            key: 1.0
            for key in scores
        }

    return {
        key: float(
            (value - minimum)
            / (maximum - minimum)
        )
        for key, value in scores.items()
    }


# ============================================================
# USER-BASED COLLABORATIVE FILTERING
# ============================================================

def get_user_based_scores(
    profile_ratings
):

    profile_vector = lil_matrix(
        (
            1,
            user_movie_matrix.shape[1]
        ),
        dtype=np.float32
    )

    rated_movie_ids = set()

    for movie_id, rating in profile_ratings:

        if movie_id not in movie_to_index:
            continue

        column = movie_to_index[
            movie_id
        ]

        profile_vector[
            0,
            column
        ] = rating

        rated_movie_ids.add(
            movie_id
        )

    profile_vector = (
        profile_vector.tocsr()
    )

    neighbor_count = min(
        20,
        user_movie_matrix.shape[0]
    )

    distances, indices = (
        user_model.kneighbors(
            profile_vector,
            n_neighbors=neighbor_count
        )
    )

    similarities = (
        1.0 - distances[0]
    )

    neighbors = indices[0]

    profile_mean = np.mean(
        [
            rating
            for _, rating
            in profile_ratings
        ]
    )

    weighted_scores = {}
    similarity_weights = {}

    for similarity, neighbor_index in zip(
        similarities,
        neighbors
    ):

        if similarity <= 0:
            continue

        neighbor_row = (
            user_movie_matrix
            .getrow(neighbor_index)
        )

        if neighbor_row.nnz == 0:
            continue

        neighbor_mean = (
            neighbor_row.data.mean()
        )

        for column, rating in zip(
            neighbor_row.indices,
            neighbor_row.data
        ):

            movie_id = (
                index_to_movie[column]
            )

            if movie_id in rated_movie_ids:
                continue

            deviation = (
                float(rating)
                - float(neighbor_mean)
            )

            weighted_scores[
                movie_id
            ] = (
                weighted_scores.get(
                    movie_id,
                    0.0
                )
                + similarity * deviation
            )

            similarity_weights[
                movie_id
            ] = (
                similarity_weights.get(
                    movie_id,
                    0.0
                )
                + similarity
            )

    final_scores = {}

    for movie_id, value in (
        weighted_scores.items()
    ):

        weight = (
            similarity_weights.get(
                movie_id,
                0.0
            )
        )

        if weight <= 0:
            continue

        prediction = (
            profile_mean
            + value / weight
        )

        final_scores[movie_id] = float(
            np.clip(
                prediction,
                0.5,
                5.0
            )
        )

    return final_scores


# ============================================================
# ITEM-BASED COLLABORATIVE FILTERING
# ============================================================

def get_item_based_scores(
    profile_ratings
):

    rated_movie_ids = {
        movie_id
        for movie_id, _
        in profile_ratings
    }

    weighted_scores = {}
    similarity_weights = {}

    neighbor_count = min(
        21,
        normalized_item_matrix.shape[0]
    )

    for movie_id, rating in profile_ratings:

        if movie_id not in movie_to_index:
            continue

        movie_index = (
            movie_to_index[movie_id]
        )

        query_vector = (
            normalized_item_matrix[
                movie_index
            ]
        )

        distances, indices = (
            item_model.kneighbors(
                query_vector,
                n_neighbors=neighbor_count
            )
        )

        similarities = (
            1.0 - distances[0]
        )

        neighbors = indices[0]

        preference = (
            float(rating) - 3.0
        )

        for similarity, neighbor_index in zip(
            similarities,
            neighbors
        ):

            if similarity <= 0:
                continue

            candidate_movie_id = (
                index_to_movie[
                    neighbor_index
                ]
            )

            if (
                candidate_movie_id
                in rated_movie_ids
            ):
                continue

            weighted_scores[
                candidate_movie_id
            ] = (
                weighted_scores.get(
                    candidate_movie_id,
                    0.0
                )
                + similarity * preference
            )

            similarity_weights[
                candidate_movie_id
            ] = (
                similarity_weights.get(
                    candidate_movie_id,
                    0.0
                )
                + similarity
            )

    final_scores = {}
    final_support = {}

    for movie_id, value in (
        weighted_scores.items()
    ):

        weight = (
            similarity_weights.get(
                movie_id,
                0.0
            )
        )

        if weight <= 0:
            continue

        # Keep how much similarity evidence supports each candidate.
        # This prevents equal profile ratings from making every item
        # score identical after the weighted average cancels similarity.
        confidence = weight / (weight + 1.0)
        final_scores[movie_id] = (value / weight) * confidence
        final_support[movie_id] = float(weight)

    return final_scores, final_support


# ============================================================
# SVD COLD-START PROFILE
# ============================================================

def get_svd_cold_start_scores(
    profile_ratings
):

    trainset = svd_model.trainset

    raw_to_inner = (
        trainset._raw2inner_id_items
    )

    known_items = []
    known_ratings = []

    for movie_id, rating in profile_ratings:

        movie_id = int(movie_id)

        if movie_id not in raw_to_inner:
            continue

        known_items.append(
            raw_to_inner[movie_id]
        )

        known_ratings.append(
            float(rating)
        )

    if not known_items:
        return {}

    known_items = np.array(
        known_items,
        dtype=np.int32
    )

    known_ratings = np.array(
        known_ratings,
        dtype=float
    )

    global_mean = float(
        trainset.global_mean
    )

    item_biases = np.asarray(
        svd_model.bi,
        dtype=float
    )

    item_factors = np.asarray(
        svd_model.qi,
        dtype=float
    )

    X = np.column_stack(
        [
            np.ones(
                len(known_items)
            ),
            item_factors[
                known_items
            ]
        ]
    )

    y = (
        known_ratings
        - global_mean
        - item_biases[
            known_items
        ]
    )

    regularization = 1.0

    A = (
        X.T @ X
        + regularization
        * np.eye(
            X.shape[1]
        )
    )

    A[0, 0] -= regularization

    b = X.T @ y

    try:

        beta = np.linalg.solve(
            A,
            b
        )

    except np.linalg.LinAlgError:

        beta = (
            np.linalg.pinv(A)
            @ b
        )

    user_bias = float(
        beta[0]
    )

    user_factors = beta[1:]

    rated_movie_ids = {
        int(movie_id)
        for movie_id, _
        in profile_ratings
    }

    scores = {}

    for movie_id in movie_ids:

        movie_id = int(movie_id)

        if movie_id in rated_movie_ids:
            continue

        if movie_id not in raw_to_inner:
            continue

        inner_id = (
            raw_to_inner[movie_id]
        )

        prediction = (
            global_mean
            + user_bias
            + item_biases[inner_id]
            + np.dot(
                item_factors[inner_id],
                user_factors
            )
        )

        scores[movie_id] = float(
            np.clip(
                prediction,
                0.5,
                5.0
            )
        )

    return scores


# ============================================================
# HYBRID RECOMMENDATION ENGINE
# ============================================================

def get_recommendations(
    profile_ratings,
    selected_genres=None,
    selected_mood=None
):

    user_raw = (
        get_user_based_scores(
            profile_ratings
        )
    )

    item_raw, item_support = (
        get_item_based_scores(
            profile_ratings
        )
    )

    svd_raw = (
        get_svd_cold_start_scores(
            profile_ratings
        )
    )

    user_scores = normalize_scores(
        user_raw
    )

    item_scores = normalize_scores(
        item_raw
    )

    svd_scores = normalize_scores(
        svd_raw
    )

    rated_movies = {
        movie_id
        for movie_id, _
        in profile_ratings
    }

    candidate_movies = (
        set(user_scores)
        | set(item_scores)
        | set(svd_scores)
    )

    candidate_movies -= rated_movies

    results = []

    for movie_id in candidate_movies:

        user_score = (
            user_scores.get(
                movie_id,
                0.0
            )
        )

        item_score = (
            item_scores.get(
                movie_id,
                0.0
            )
        )

        svd_score = (
            svd_scores.get(
                movie_id,
                0.0
            )
        )

        base_hybrid_score = (
            0.30 * user_score
            + 0.30 * item_score
            + 0.40 * svd_score
        )

        if movie_id not in movie_lookup.index:
            continue

        imdb_id = movie_lookup.loc[movie_id, "imdbId"]
        tmdb_id = movie_lookup.loc[movie_id, "tmdbId"]
        raw_genres = movie_lookup.loc[movie_id, "genres"]
        genre_value = "" if pd.isna(raw_genres) else str(raw_genres)
        context_score = get_context_score(
            genre_value,
            selected_genres or [],
            selected_mood
        )
        results.append(
            {
                "movieId": int(movie_id),
                "title": str(
                    movie_lookup.loc[
                        movie_id,
                        "title"
                    ]
                ),
                "genres": str(
                    movie_lookup.loc[
                        movie_id,
                        "genres"
                    ]
                ),
                "imdbId": imdb_id,
                "tmdbId": tmdb_id,
                "user_score": user_score,
                "item_score": item_score,
                "svd_score": svd_score,
                "item_support": item_support.get(movie_id, 0.0),
                "base_hybrid_score": float(base_hybrid_score),
                "context_score": context_score,
                "hybrid_score": float(base_hybrid_score)
            }
        )

    result_df = pd.DataFrame(
        results
    )

    if result_df.empty:
        return result_df

    has_context = bool(selected_genres) or bool(selected_mood)
    context_available = (
        has_context
        and result_df["context_score"].notna().any()
        and result_df["context_score"].fillna(0).max() > 0
    )
    result_df["context_applied"] = bool(context_available)
    if context_available:
        result_df["final_score"] = (
            (1.0 - CONTEXT_WEIGHT) * result_df["base_hybrid_score"]
            + CONTEXT_WEIGHT * result_df["context_score"]
        )
    else:
        # Preserve the original ranking exactly when no context is selected
        # or no candidate has any meaningful contextual match.
        result_df["final_score"] = result_df["base_hybrid_score"]
    result_df["hybrid_score"] = result_df["final_score"]

    result_df = (
        result_df
        .sort_values(
            "final_score",
            ascending=False
        )
        .head(10)
        .reset_index(drop=True)
    )

    result_df.insert(
        0,
        "rank",
        range(
            1,
            len(result_df) + 1
        )
    )

    return result_df


def get_current_viewing_recommendations(selected_genres=None, selected_mood=None):
    """Recommend from metadata for a current-viewing request without a taste profile."""
    selected_genres = set(selected_genres or [])
    mood_genres = MOOD_GENRES.get(selected_mood, set()) if selected_mood else set()
    results = []

    for movie_id, movie in movie_lookup.iterrows():
        raw_genres = movie.get("genres")
        actual = set() if pd.isna(raw_genres) else {
            genre.strip() for genre in str(raw_genres).split("|") if genre.strip()
        }

        genre_score = None
        if selected_genres:
            genre_matches = [
                {"Romance", "Comedy"}.issubset(actual)
                if genre == "Rom-Com"
                else genre in actual
                for genre in selected_genres
            ]
            genre_score = sum(genre_matches) / len(genre_matches)
            # In current-viewing mode, selected genres define the candidate set.
            if genre_score == 0:
                continue

        mood_score = (
            len(actual & mood_genres) / len(mood_genres)
            if mood_genres else None
        )
        if genre_score is not None and mood_score is not None:
            context_score = 0.70 * genre_score + 0.30 * mood_score
        else:
            context_score = genre_score if genre_score is not None else mood_score

        results.append({
            "movieId": int(movie_id),
            "title": str(movie.get("title", "")),
            "genres": "" if pd.isna(raw_genres) else str(raw_genres),
            "imdbId": movie.get("imdbId"),
            "tmdbId": movie.get("tmdbId"),
            "user_score": 0.0,
            "item_score": 0.0,
            "svd_score": 0.0,
            "item_support": 0.0,
            "base_hybrid_score": None,
            "context_score": float(context_score or 0.0),
            "hybrid_score": float(context_score or 0.0),
            "final_score": float(context_score or 0.0),
            "context_applied": True,
        })

    result_df = pd.DataFrame(results)
    if result_df.empty:
        return result_df

    result_df = (
        result_df.sort_values(["final_score", "title"], ascending=[False, True])
        .head(10)
        .reset_index(drop=True)
    )
    result_df.insert(0, "rank", range(1, len(result_df) + 1))
    return result_df


# ============================================================
# GENRE PROFILE
# ============================================================

def get_genre_profile(
    profile_ratings
):

    genre_data = {}

    for movie_id, rating in profile_ratings:

        if movie_id not in movie_lookup.index:
            continue

        genres = str(
            movie_lookup.loc[
                movie_id,
                "genres"
            ]
        )

        if genres == "(no genres listed)":
            continue

        for genre in genres.split("|"):

            if genre not in genre_data:
                genre_data[genre] = []

            genre_data[genre].append(
                float(rating)
            )

    rows = []

    for genre, values in genre_data.items():

        rows.append(
            {
                "Genre": genre,
                "Movies": len(values),
                "Average Rating": round(
                    np.mean(values),
                    2
                )
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            "Average Rating",
            ascending=False
        )
        .reset_index(drop=True)
    )


# Mood is not a MovieLens field; these associations interpret mood through
# genres that exist in the current MovieMind dataset.
MOOD_GENRES = {
    "Feel-Good": {"Comedy", "Children", "Animation"},
    "Funny": {"Comedy"},
    "Romantic": {"Romance"},
    "Emotional": {"Drama", "Romance"},
    "Relaxing": {"Animation", "Comedy", "Documentary"},
    "Exciting": {"Action", "Adventure", "Thriller"},
    "Dark": {"Horror", "Crime", "Thriller"},
    "Suspenseful": {"Thriller", "Mystery", "Crime"},
}

PRACTICAL_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Horror", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
DATASET_GENRES = {
    genre
    for genres in movies["genres"].fillna("").astype(str)
    for genre in genres.split("|")
    if genre and genre != "(no genres listed)"
}
GENRE_OPTIONS = [genre for genre in PRACTICAL_GENRES if genre in DATASET_GENRES]
if {"Romance", "Comedy"}.issubset(DATASET_GENRES):
    # Convenience context for a combined romance/comedy request; the dataset
    # stores these as separate genre tags.
    GENRE_OPTIONS.append("Rom-Com")


def get_context_score(movie_genres, selected_genres=None, selected_mood=None):
    """Return a semantic 0–1 genre/mood relevance score, or None without context."""
    selected_genres = set(selected_genres or [])
    mood_genres = MOOD_GENRES.get(selected_mood, set()) if selected_mood else set()
    if not selected_genres and not mood_genres:
        return None

    if not isinstance(movie_genres, str) or not movie_genres.strip():
        actual = set()
    else:
        actual = {
            genre.strip()
            for genre in movie_genres.split("|")
            if genre.strip() and genre.strip() != "(no genres listed)"
        }

    component_scores = []
    if selected_genres:
        genre_matches = []
        for genre in selected_genres:
            if genre == "Rom-Com":
                genre_matches.append({"Romance", "Comedy"}.issubset(actual))
            else:
                genre_matches.append(genre in actual)
        component_scores.append(sum(genre_matches) / len(genre_matches))
    if mood_genres:
        component_scores.append(len(actual & mood_genres) / len(mood_genres))
    return float(np.mean(component_scores)) if component_scores else 0.0


# ============================================================
# ============================================================
# LOCAL PROFILE STORAGE
# ============================================================

def load_profiles():
    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as profile_file:
            profiles = json.load(profile_file)
        return profiles if isinstance(profiles, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_profiles(profiles):
    with open(PROFILE_PATH, "w", encoding="utf-8") as profile_file:
        json.dump(profiles, profile_file, indent=2, ensure_ascii=False)


def profile_rows_to_pairs(profile):
    unique = {}
    for row in profile.get("movies", []):
        movie_id = int(row["movieId"])
        rating = float(row["rating"])
        if movie_id in movie_lookup.index and np.isfinite(rating):
            unique[movie_id] = float(np.clip(rating, 0.5, 5.0))
    return list(unique.items())


def apply_profile(profile):
    pairs = profile_rows_to_pairs(profile)
    id_to_label = {movie_id: label for label, movie_id in movie_label_to_id.items()}
    st.session_state["profile_name"] = profile.get("name", "")
    st.session_state["profile_id"] = profile.get("profile_id")
    st.session_state["profile"] = pairs
    st.session_state["selected_movies"] = [id_to_label[mid] for mid, _ in pairs if mid in id_to_label]
    for movie_id, rating in pairs:
        st.session_state[f"taste_rating_{movie_id}"] = rating
    st.session_state.pop("recommendations", None)


def reset_profile_state():
    for key in list(st.session_state):
        if key.startswith("taste_rating_"):
            del st.session_state[key]
    st.session_state["selected_movies"] = []
    st.session_state["profile_name"] = ""
    st.session_state.pop("profile_id", None)
    st.session_state.pop("profile", None)
    st.session_state.pop("recommendations", None)


# ============================================================
# APPLICATION
# ============================================================

profiles = load_profiles()
st.markdown("""<div class="hero-box"><div class="hero-title">MovieMind</div>
<div class="hero-subtitle">Personalized Hybrid Movie Recommendation System</div>
<div class="hero-description">Recommend by what you want to watch, personalize using your taste, or combine current intent with your long-term preferences.</div></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## MovieMind")
    st.caption("Personalized recommendation engine")
    st.divider()
    st.markdown("**Recommendation Signals**")
    st.write("Taste Match — 30%")
    st.write("Movie Similarity — 30%")
    st.write("Preference Learning — 40%")
    st.divider()
    st.markdown("**Dataset**")
    st.write(f"{len(user_ids):,} users")
    st.write(f"{len(movies):,} movies")
    st.write(f"{len(ratings):,} ratings")
    st.divider()
    st.caption("Saved profiles are stored locally on this computer.")
    st.markdown("**Saved profiles**")
    st.caption(f"{len(profiles)} profile(s) saved locally.")
    profile_labels = {p.get("profile_id"): p.get("name", "Unnamed") for p in profiles if p.get("profile_id")}
    chosen_profile_id = st.selectbox("Load a profile", [""] + list(profile_labels), format_func=lambda value: "Choose a saved profile" if not value else profile_labels.get(value, value), label_visibility="collapsed")
    if st.button("Load Profile", disabled=not chosen_profile_id, use_container_width=True):
        match = next((p for p in profiles if p.get("profile_id") == chosen_profile_id), None)
        if match:
            apply_profile(match)
            st.rerun()

st.markdown("## Recommend by what I want to watch")
st.caption("Choose a genre and/or current mood to get recommendations without rating any movies. When a valid taste profile is also provided, MovieMind combines both preferences.")
context_genre_col, context_mood_col = st.columns(2)
with context_genre_col:
    selected_genres = st.multiselect(
        "Genre Preference",
        options=GENRE_OPTIONS,
        placeholder="Any",
        key="context_genres"
    )
with context_mood_col:
    selected_mood_option = st.selectbox(
        "What are you in the mood for?",
        options=["Any", *MOOD_GENRES],
        key="context_mood"
    )
selected_mood = None if selected_mood_option == "Any" else selected_mood_option

st.markdown("## Personalize using my taste")
st.caption("Choose and rate 5 to 10 movies to use MovieMind's personalized hybrid system: Taste Match 30%, Movie Similarity 30%, and Preference Learning 40%.")
st.text_input("Profile name", key="profile_name", placeholder="Enter your name")
selected_movies = st.multiselect("Search and select movies", options=list(movie_label_to_id), max_selections=10, placeholder="Search by title...", key="selected_movies")

if selected_movies:
    st.markdown("### Rate your movies")
    st.caption("Each movie can be rated in half-star increments.")
    for start in range(0, len(selected_movies), 2):
        columns = st.columns(2)
        for pos, column in enumerate(columns):
            idx = start + pos
            if idx >= len(selected_movies):
                continue
            label = selected_movies[idx]
            movie_id = movie_label_to_id[label]
            movie_info = movie_lookup.loc[movie_id]
            with column:
                with st.container(border=True):
                    st.markdown(f"**{movie_info['title']}**")
                    st.caption(str(movie_info["genres"]).replace("|", " · "))
                    st.slider("Your rating", 0.5, 5.0, 4.0, 0.5, key=f"taste_rating_{movie_id}")

profile_preview = [(movie_label_to_id[label], float(st.session_state.get(f"taste_rating_{movie_label_to_id[label]}", 4.0))) for label in selected_movies]
if profile_preview:
    genre_preview = get_genre_profile(profile_preview)
    leading_genre = genre_preview.iloc[0]["Genre"] if not genre_preview.empty else "Not available"
    m1, m2, m3 = st.columns(3)
    m1.metric("Movies Selected", len(profile_preview))
    m2.metric("Average Rating", f"{np.mean([r for _, r in profile_preview]):.1f} / 5")
    m3.metric("Leading Genre", leading_genre)

b1, b2, b3 = st.columns([2, 1, 1])
generate = b1.button("Generate Recommendations", type="primary", use_container_width=True)
save_profile = b2.button("Save Profile", use_container_width=True)
reset = b3.button("Reset Profile", use_container_width=True, on_click=reset_profile_state)

if save_profile:
    if not st.session_state.get("profile_name", "").strip():
        st.warning("Enter a profile name before saving.")
    elif not 5 <= len(profile_preview) <= 10:
        st.warning("Select between 5 and 10 movies before saving.")
    else:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        profile_id = st.session_state.get("profile_id") or f"MM-{uuid.uuid4().hex[:6].upper()}"
        record = {"profile_id": profile_id, "name": st.session_state["profile_name"].strip(), "movies": [{"movieId": mid, "rating": rating} for mid, rating in profile_preview]}
        previous = next((p for p in profiles if p.get("profile_id") == profile_id), None)
        record["created_at"] = previous.get("created_at", now) if previous else now
        record["updated_at"] = now
        profiles = [p for p in profiles if p.get("profile_id") != profile_id] + [record]
        save_profiles(profiles)
        st.session_state["profile_id"] = profile_id
        st.session_state["profile"] = profile_preview
        st.success("Profile saved locally.")

if generate:
    valid_profile = 5 <= len(profile_preview) <= 10
    has_context = bool(selected_genres or selected_mood)
    if not valid_profile and not has_context:
        st.warning("Choose a genre or mood, or rate 5 to 10 movies, before generating recommendations.")
    else:
        if valid_profile:
            mode = "combined" if has_context else "personalized"
            with st.spinner("Building your recommendations..."):
                st.session_state["recommendations"] = get_recommendations(
                    profile_preview,
                    selected_genres=selected_genres,
                    selected_mood=selected_mood
                )
            st.session_state["profile"] = profile_preview
        else:
            mode = "current"
            with st.spinner("Finding movies for your current viewing intent..."):
                st.session_state["recommendations"] = get_current_viewing_recommendations(
                    selected_genres=selected_genres,
                    selected_mood=selected_mood
                )
            st.session_state["profile"] = []
        st.session_state["recommendation_context"] = {
            "genres": list(selected_genres),
            "mood": selected_mood,
        }
        st.session_state["recommendation_mode"] = mode
        st.session_state["recommendation_profile"] = list(profile_preview) if valid_profile else []
        st.success("Your recommendations are ready.")

active_profile = profile_preview
stored_profile = st.session_state.get("profile", [])
recommendations = st.session_state.get("recommendations", pd.DataFrame())
generated_context = st.session_state.get("recommendation_context", {"genres": [], "mood": None})
generated_mode = st.session_state.get("recommendation_mode", "personalized")
if (
    (generated_mode != "current" and stored_profile != profile_preview)
    or generated_context.get("genres", []) != list(selected_genres)
    or generated_context.get("mood") != selected_mood
    or (generated_mode != "current" and st.session_state.get("recommendation_profile", []) != list(profile_preview))
):
    recommendations = pd.DataFrame()
recommendation_tab, taste_tab, history_tab, evaluation_tab, about_tab = st.tabs(["Recommendations", "My Taste", "Rating History", "Model Performance", "About MovieMind"])

with recommendation_tab:
    current_mode = st.session_state.get("recommendation_mode", "personalized")
    if current_mode == "current":
        st.markdown("### Recommendations for your viewing intent")
        st.caption("Viewing Intent Match reflects movie genre metadata and your current genre and mood preferences; it is not a probability.")
    else:
        st.markdown("### MovieMind Match")
        st.caption("The Match percentage is a normalized hybrid recommendation score, not a probability that you will like the movie.")
    if recommendations.empty:
        st.info("Choose a genre or mood, or rate 5 to 10 movies. You can use either path or provide both.")
    else:
        current_context = st.session_state.get("recommendation_context", {"genres": [], "mood": None})
        context_genres = current_context.get("genres", [])
        context_mood = current_context.get("mood")
        has_context = bool(context_genres or context_mood)
        st.markdown("#### Your current viewing intent")
        st.caption(f"Genre: {', '.join(context_genres) if context_genres else 'Any'}")
        st.caption(f"What are you in the mood for?: {context_mood or 'Any'}")
        mode_label = {
            "current": "Recommend by what I want to watch",
            "personalized": "Personalize using my taste",
            "combined": "Personalized taste + current viewing intent",
        }.get(current_mode, "Personalize using my taste")
        st.caption(f"Recommendation mode: {mode_label}")
        if current_mode == "current":
            st.caption("Ranked from movie genre metadata and current viewing intent. No rated movies are required.")
        elif current_mode == "combined":
            st.caption("The 30/30/40 personalized hybrid is adjusted by your current viewing context.")
        for _, movie in recommendations.iterrows():
            user_score, item_score, svd_score = (float(movie[k]) for k in ("user_score", "item_score", "svd_score"))
            score = float(movie["hybrid_score"])
            contributions = {"Taste Match": .30 * user_score, "Movie Similarity": .30 * item_score, "Preference Learning": .40 * svd_score}
            strongest = max(contributions, key=contributions.get)
            strength = "Very strong match" if score >= .8 else "Strong match" if score >= .6 else "Good match" if score >= .4 else "Potential match"
            with st.container(border=True):
                left, right = st.columns([5, 1])
                left.caption(f"RANK {int(movie['rank'])}")
                left.markdown(f"### {movie['title']}")
                left.caption(str(movie["genres"]).replace("|", " · "))
                right.metric("Viewing Intent Match" if current_mode == "current" else "MovieMind Match", f"{score:.0%}")
                st.progress(float(np.clip(score, 0, 1)))
                if current_mode == "current":
                    st.caption(f"{strength} for your selected genre and mood preferences.")
                else:
                    st.caption(f"{strength} · Main signal: {strongest}")
                if has_context:
                    st.caption(
                        f"Current viewing intent: {' + '.join(context_genres) if context_genres else ''}"
                        f"{' · ' if context_genres and context_mood else ''}"
                        f"{f'What are you in the mood for?: {context_mood}' if context_mood else ''}"
                    )
                    if bool(movie.get("context_applied", False)):
                        relevance = float(movie["context_score"])
                        relevance_label = "Strong" if relevance >= 0.75 else "Good" if relevance >= 0.4 else "Low"
                        st.caption(f"Context: {relevance_label} match for your current viewing preference.")
                    else:
                        st.caption("No candidates matched this context; the personalized ranking was retained.")
                if current_mode != "current":
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Taste Match", f"{contributions['Taste Match']:.1%}")
                    c2.metric("Movie Similarity", f"{contributions['Movie Similarity']:.1%}")
                    c3.metric("Preference Learning", f"{contributions['Preference Learning']:.1%}")
                with st.expander("View recommendation details"):
                    if current_mode == "current":
                        st.write("This recommendation uses movie genre metadata and your current viewing intent. Mood is mapped to related genres for ranking; it is not a movie attribute or a separate model.")
                        st.caption(f"Viewing intent relevance score: {float(movie['context_score']):.4f}.")
                    else:
                        st.write("Taste Match — User-Based Collaborative Filtering compares the profile with nearby existing users using centered rating deviations.")
                        st.write("Movie Similarity — Item-Based Collaborative Filtering combines similarities to the movies you rated.")
                        st.write("Preference Learning — SVD Matrix Factorization infers a temporary latent profile from your ratings.")
                        st.caption(f"Normalized model scores: Taste Match {user_score:.4f}; Movie Similarity {item_score:.4f}; Preference Learning {svd_score:.4f}. Hybrid = 0.30 × Taste + 0.30 × Similarity + 0.40 × Preference = {float(movie['base_hybrid_score']):.4f}.")
                    if has_context and current_mode != "current":
                        if bool(movie.get("context_applied", False)):
                            st.caption(f"Base hybrid: {float(movie['base_hybrid_score']):.4f}; context relevance: {float(movie['context_score']):.4f}; final = 0.85 × base + 0.15 × context = {score:.4f}.")
                        else:
                            st.caption(f"No meaningful contextual matches were available; final score remains the base hybrid score of {score:.4f}.")
                    st.caption(f"Movie Similarity support: {float(movie['item_support']):.3f} cumulative cosine similarity.")
                links_out = []
                imdb_id, tmdb_id = movie.get("imdbId"), movie.get("tmdbId")
                if pd.notna(imdb_id) and int(imdb_id) > 0:
                    links_out.append(f"[IMDb](https://www.imdb.com/title/tt{int(imdb_id):07d}/)")
                if pd.notna(tmdb_id) and int(tmdb_id) > 0:
                    links_out.append(f"[TMDB](https://www.themoviedb.org/movie/{int(tmdb_id)})")
                if links_out:
                    st.markdown(" · ".join(links_out))

with taste_tab:
    st.markdown("### Selected movies and ratings")
    taste_rows = [{"Movie": movie_lookup.loc[mid, "title"], "Rating": rating, "Genres": str(movie_lookup.loc[mid, "genres"]).replace("|", " · ")} for mid, rating in active_profile if mid in movie_lookup.index]
    taste_df = pd.DataFrame(taste_rows)
    if taste_df.empty:
        st.info("Your taste profile will appear here after you select movies.")
    else:
        st.metric("Average Rating", f"{taste_df['Rating'].mean():.2f} / 5")
        st.dataframe(taste_df, use_container_width=True, hide_index=True)
        genre_df = get_genre_profile(active_profile)
        st.markdown("### Genre profile")
        if not genre_df.empty:
            st.dataframe(genre_df, use_container_width=True, hide_index=True)
    current_context = st.session_state.get("recommendation_context", {"genres": [], "mood": None})
    st.markdown("### Current viewing intent")
    st.caption("This temporary context is separate from your long-term taste profile.")
    st.write(f"Genre: {', '.join(current_context.get('genres', [])) or 'Any'}")
    st.write(f"What are you in the mood for?: {current_context.get('mood') or 'Any'}")

with history_tab:
    st.markdown("### Rating History")
    if taste_df.empty:
        st.info("Ratings for the current profile will appear here.")
    else:
        st.dataframe(taste_df, use_container_width=True, hide_index=True)
        st.download_button("Download rating history as CSV", taste_df.to_csv(index=False).encode("utf-8"), "moviemind_rating_history.csv", "text/csv")

with evaluation_tab:
    st.markdown("### How MovieMind Recommends")
    st.caption("These are the live weights used in Personalize using my taste and combined recommendation modes. Current Viewing mode ranks from genre metadata and mood mapping without a rated profile.")
    signal_columns = st.columns(3)
    signal_cards = [
        ("Taste Match", "30%", "Finds patterns from users with similar rating behavior."),
        ("Movie Similarity", "30%", "Finds relationships between movies based on rating patterns."),
        ("Preference Learning", "40%", "Uses SVD matrix factorization to learn latent user and movie preferences."),
    ]
    for column, (name, weight, description) in zip(signal_columns, signal_cards):
        with column:
            with st.container(border=True):
                st.metric(name, weight)
                st.caption(description)
    st.info("The personalized hybrid combines these three signals. Current Viewing mode uses genre metadata and mood mapping without a rated profile. The evaluation metrics below come from a separate experiment.")

    st.markdown("### Evaluation Results")
    st.caption("Separate SVD evaluation experiment; these are not live accuracy measurements for the current profile.")
    metric_columns = st.columns(4)
    evaluation_metrics = [
        ("RMSE", "0.8352", "Measures the magnitude of prediction error. Lower values indicate smaller errors."),
        ("MAE", "0.6372", "Average absolute difference between predicted and actual ratings. Lower values indicate smaller errors."),
        ("Precision@10", "53.37%", "Percentage of top-10 recommendations that were relevant in the evaluation experiment."),
        ("Recall@10", "74.77%", "Percentage of relevant movies retrieved within the top-10 recommendations."),
    ]
    for column, (name, value, description) in zip(metric_columns, evaluation_metrics):
        with column:
            with st.container(border=True):
                st.metric(name, value)
                st.caption(description)

    st.markdown("### Evaluation Context")
    st.caption("These results come from a separate SVD evaluation experiment and are not live accuracy measurements for the current profile.")
    context_columns = st.columns(5)
    for column, label, value in zip(
        context_columns,
        ["Evaluation sample", "Training", "Testing", "Users evaluated", "Relevant threshold"],
        ["1,000,000 ratings", "800,000 ratings", "200,000 ratings", "9,804", "4.0 or higher"],
    ):
        with column:
            st.metric(label, value)
    with st.expander("View technical evaluation details"):
        st.write("The evaluation used Surprise SVD with 100 latent factors, 20 training epochs, learning rate 0.005, regularization 0.02, and random_state 42.")
        st.write("A reproducible sample of 1,000,000 ratings was split 80/20 into training and held-out test data. Ratings of 4.0 or higher were treated as relevant for Precision@10 and Recall@10.")

    st.markdown("### Current Viewing Context")
    st.caption("Current Viewing mode ranks movies from selected genre metadata and current mood intent; no rated movies are required. When a taste profile is also provided, context adjusts the existing personalized hybrid ranking. Mood is not a movie attribute or separate model.")
    if selected_genres or selected_mood:
        st.write(f"Genre: {' + '.join(selected_genres) if selected_genres else 'Any'}")
        st.write(f"What are you in the mood for?: {selected_mood or 'Any'}")
    else:
        st.caption("No current genre or mood context is selected. Personalized scores remain unchanged.")

with about_tab:
    st.markdown("### What MovieMind does")
    st.write("MovieMind supports current viewing recommendations from genre and mood preferences without requiring ratings, personalized recommendations from a 5 to 10 movie taste profile, or a combination of both.")
    st.markdown("### Recommendation approach")
    st.write("MovieMind combines long-term taste preferences, collaborative filtering, matrix factorization, and current viewing intent. The core hybrid remains 30% Taste Match + 30% Movie Similarity + 40% Preference Learning. A ridge-regression step infers a temporary SVD profile for new visitors from their selected ratings.")
    st.write("Genre comes from movie metadata. Mood is derived from an editable mood-to-genre mapping; it is not a native dataset field. Context is an optional ranking adjustment, not a fourth ML model.")
    st.write("The internal hybrid score ranges from 0 to 1 and is displayed as a percentage. It is a normalized recommendation score, not a probability.")
    st.markdown("### Data, storage, and technology")
    st.write(f"The development dataset contains {len(user_ids):,} users, {len(movies):,} movies, and {len(ratings):,} ratings. Ratings remain in a sparse matrix to avoid a large dense allocation. Saved profile names and movie ratings are stored locally in data/user_profiles.json; no passwords or sensitive information are requested.")
    st.write("Technology stack: Python, Streamlit, Pandas, NumPy, SciPy, scikit-learn, and Surprise.")

st.divider()
st.markdown('<div class="footer-text">MovieMind | Personalized Hybrid Movie Recommendation System</div>', unsafe_allow_html=True)
