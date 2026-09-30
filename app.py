import base64
import json
import html
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
try:
    from gemini_helper import parse_movie_intent
except Exception:
    parse_movie_intent = None
from tmdb_helper import (
    POSTER_PLACEHOLDER,
    get_movie_backdrop_image,
    get_movie_details,
    get_movie_poster_image,
)

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
.block-container { max-width: 1500px; padding-top: 1.3rem; padding-bottom: 3rem; }
[data-testid="stAppViewContainer"] { background: radial-gradient(ellipse at 76% -18%, #202a3b 0%, #0b0e13 45%, #090b0f 100%); }
section[data-testid="stSidebar"] { width: 242px !important; min-width: 242px !important; background: linear-gradient(180deg,#111722 0%,#0b0e13 58%,#090b0f 100%); border-right: 1px solid rgba(255,255,255,.07); }
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 1rem .85rem 1.2rem; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #aab3c1; }
.sidebar-brand { display:flex;align-items:center;gap:10px;color:#f6f7fa;padding:5px 4px 14px;border-bottom:1px solid #ffffff12;margin-bottom:17px; }
.sidebar-brand-mark { width:30px;height:30px;display:grid;place-items:center;border-radius:9px;background:linear-gradient(145deg,#e0a66d,#b76575);color:#111;font-weight:900;font-size:16px; }
.sidebar-brand-name { font-size:17px;font-weight:760;letter-spacing:-.45px;line-height:1.1; }
.sidebar-brand-caption { color:#8994a4;font-size:9px;letter-spacing:1.15px;text-transform:uppercase;margin-top:4px; }
.sidebar-section-label { color:#778394;font-size:9px;font-weight:800;letter-spacing:1.5px;margin:15px 8px 5px; }
section[data-testid="stSidebar"] .stButton > button { width:100%;justify-content:flex-start;text-align:left;border:1px solid transparent;background:transparent;padding:0 10px;min-height:35px;color:#aab4c1;font-size:12px; }
section[data-testid="stSidebar"] .stButton > button:hover { background:#ffffff0b;color:#f4f6f9;border-color:transparent; }
section[data-testid="stSidebar"] .stButton > button[kind="primary"] { background:linear-gradient(90deg,#d6a36b1d,#d6a36b08)!important;box-shadow:inset 2px 0 #d6a36b;color:#fff!important;border:1px solid transparent!important; }
.sidebar-profile { margin-top:17px;padding:13px 12px 10px;border:1px solid #ffffff12;border-radius:13px;background:linear-gradient(145deg,#ffffff08,#ffffff03); }
.sidebar-profile-title { color:#7e8a9a;font-size:9px;font-weight:800;letter-spacing:1.4px;margin-bottom:8px; }
.sidebar-profile-name { color:#f4f6f9;font-size:14px;font-weight:700; }
.sidebar-profile-meta { color:#9aa5b4;font-size:11px;margin:3px 0 10px; }
.brand-bar { display:flex;align-items:center;gap:11px;margin:0 0 17px;color:#f5f7fb; }
.brand-mark { width:32px;height:32px;border-radius:10px;display:grid;place-items:center;background:linear-gradient(145deg,#dca66d,#b86479);color:#161216;font-size:16px;font-weight:900; }
.brand-name { font-size:20px;font-weight:760;letter-spacing:-.65px;line-height:1.1; }
.brand-sub { color:#929eaf;font-size:11px;margin-top:3px; }
.brand-ai { margin-left:auto;color:#c9d2dc;font-size:10px;letter-spacing:.25px;padding:6px 9px;border:1px solid #ffffff17;border-radius:999px;background:#ffffff08;white-space:nowrap; }
.brand-ai-dot { color:#84c8ad;margin-right:5px; }
.ai-panel { padding:20px 22px 10px;border:1px solid #ffffff17;border-bottom:0;border-radius:18px 18px 0 0;background:linear-gradient(120deg,rgba(24,31,43,.96),rgba(16,20,28,.95));box-shadow:0 14px 40px #0004;margin:0; }
.ai-kicker { color:#d9a978;text-transform:uppercase;letter-spacing:1.55px;font-size:10px;font-weight:800; }
.ai-heading { color:#f7f8fb;font-size:24px;font-weight:720;letter-spacing:-.55px;margin:5px 0; }
.ai-sub { color:#aeb7c4;font-size:12px;margin-bottom:2px; }
div[data-testid="stForm"] { padding:5px 20px 4px;border:1px solid #ffffff17;border-top:0;border-bottom:0;border-radius:0;background:linear-gradient(120deg,rgba(24,31,43,.96),rgba(16,20,28,.95)); }
div[data-testid="stForm"] input { min-height:52px!important;border:1px solid #ffffff20!important;border-radius:13px!important;background:#0e131b!important;color:#f4f6f9!important;padding:0 16px!important;font-size:14px!important;transition:border-color .16s ease,box-shadow .16s ease!important; }
div[data-testid="stForm"] input:focus { border-color:#c9956a!important;box-shadow:0 0 0 3px #d6a36b1a!important; }
div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button { min-height:50px;border-radius:12px;background:linear-gradient(110deg,#d3a06c,#bd7182);border:0;color:#171316;font-weight:750;transition:filter .16s ease,transform .16s ease; }
div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover { filter:brightness(1.08);transform:translateY(-1px); }
.ai-examples { color:#8f9aab;font-size:10px;line-height:1.8;padding:2px 22px 13px;border:1px solid #ffffff17;border-top:0;border-radius:0 0 18px 18px;background:linear-gradient(120deg,rgba(24,31,43,.96),rgba(16,20,28,.95));margin:0 0 12px; }
.ai-example-label { color:#c5ced8;font-weight:700;letter-spacing:.4px;margin-right:5px; }
.ai-status { margin:8px 0 15px;padding:11px 14px;border:1px solid #ffffff14;border-radius:12px;background:#111720;color:#cbd3dd;font-size:12px; }
.ai-status-label { display:block;color:#8190a1;font-size:9px;font-weight:800;letter-spacing:1.3px;margin-bottom:5px; }
.ai-error-state { margin:8px 0 15px;padding:11px 14px;border:1px solid #c5756c50;border-radius:12px;background:#492b2a2e;color:#edc9c4;font-size:12px; }
.ai-429-state { border-color:#d4a15e55;background:#57402438;color:#f2d7b3; }
.manual-refine-label { color:#909bac;font-size:9px;font-weight:800;letter-spacing:1.4px;text-transform:uppercase; }
.sidebar-profile div[data-testid="stSelectbox"] label { display:none; }
.cinema-hero { min-height:350px;position:relative;overflow:hidden;border-radius:24px;margin:4px 0 28px;padding:40px 42px;display:flex;align-items:flex-end;background:linear-gradient(90deg,rgba(5,7,11,.97) 0%,rgba(5,7,11,.82) 39%,rgba(5,7,11,.16) 100%),linear-gradient(0deg,rgba(5,7,11,.82),transparent 65%),var(--hero-bg,linear-gradient(125deg,#18283d,#20152a 56%,#0e121a));background-position:center;background-size:cover;border:1px solid #ffffff18;box-shadow:0 22px 65px #0007; }
.cinema-copy { max-width:650px;position:relative;z-index:1; }
.cinema-kicker { font-size:11px;text-transform:uppercase;letter-spacing:1.5px;font-weight:800;color:#ffbd79; }
.cinema-title { color:#fff;font-size:clamp(32px,4.2vw,56px);line-height:1.02;letter-spacing:-1.8px;font-weight:820;margin:10px 0; }
.cinema-meta { color:#e5e8ed;font-size:13px;font-weight:650;margin:10px 0; }
.cinema-overview { color:#d2d6df;font-size:14px;line-height:1.55;max-width:580px; }
.cinema-why { color:#f0c7a1;font-size:13px;margin-top:13px; }
.match-pill { display:inline-block;padding:7px 11px;border-radius:999px;background:#ffffff1c;border:1px solid #ffffff2b;color:#fff;font-size:13px;font-weight:750;backdrop-filter:blur(8px); }
.movie-card { background:linear-gradient(160deg,#171c25,#101319);border:1px solid #ffffff0e;border-radius:13px;overflow:hidden;transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease;margin-bottom:8px; }
.movie-card:hover { transform:translateY(-4px);border-color:#eab5738c;box-shadow:0 13px 28px #0008; }
.movie-poster { width:100%;aspect-ratio:2/3;object-fit:cover;display:block;background:#171b22; }
.movie-card-info { padding:10px 11px 11px; }
.movie-card-title { color:#f3f5f8;font-size:14px;font-weight:730;line-height:1.25;min-height:35px; }
.movie-card-meta { color:#9ca6b5;font-size:11px;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }
.movie-card-match { color:#ffc27e;font-size:12px;font-weight:750;margin-top:8px; }
.movie-card-intent { color:#aeb9c9;font-size:10px;line-height:1.35;margin-top:4px; }
.row-heading { color:#f4f5f8;font-size:21px;font-weight:760;letter-spacing:-.35px;margin:28px 0 2px; }
.row-caption { color:#8994a4;font-size:12px;margin-bottom:10px; }
.intent-chip { display:inline-block;background:#ffffff10;border:1px solid #ffffff18;color:#dce2eb;border-radius:999px;padding:5px 10px;margin:3px 5px 3px 0;font-size:11px; }
.section-gap {height:10px;}
.stButton > button { border-radius:10px;min-height:38px;font-weight:650;border-color:#ffffff1c;background:#161b24;color:#e9edf3;transition:all .16s ease; }
.stButton > button:hover { border-color:#e4ac6d;background:#222733;color:#fff; }
.stButton > button[kind="primary"] { background:linear-gradient(110deg,#d94b63,#9b4e9b);border:0;color:white; }
button[data-baseweb="tab"] { font-weight:650;color:#aab3c0; }
button[data-baseweb="tab"][aria-selected="true"] { color:#fff; }
div[data-testid="stExpander"] { border-radius:13px;border-color:#ffffff15;background:#10141b; }
div[data-testid="stMetric"] { background:#121720;border-radius:12px;padding:10px; }
.footer-text { color:#687385;font-size:12px;text-align:center;margin-top:22px; }
@media (max-width: 760px) { .block-container{padding-top:1rem}.cinema-hero{min-height:300px;padding:25px 22px}.ai-panel{padding:17px 16px 8px}.ai-heading{font-size:21px}.ai-examples{padding-left:16px;padding-right:16px}.brand-sub{display:none}.row-heading{font-size:18px}section[data-testid="stSidebar"]{width:min(250px,82vw)!important;min-width:min(250px,82vw)!important} }

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

        display_title = title
        if title_counts[title] > 1:
            label = f"{display_title} | Movie ID {movie_id}"
        else:
            label = display_title

        movie_label_to_id[
            label
        ] = movie_id

    return movie_label_to_id


movie_label_to_id = (
    create_movie_options(movies)
)


def movie_title_year(title):
    """Return a local dataset title and its year, when present."""
    title = str(title)
    match = re.search(r"\((\d{4})\)\s*$", title)
    return (title[:match.start()].strip(), match.group(1)) if match else (title, "Year unavailable")


def get_similar_movies(movie_id, limit=8):
    """Use the fitted item-CF model to return cosine-nearest local movies."""
    movie_id = int(movie_id)
    if movie_id not in movie_to_index:
        return pd.DataFrame()
    movie_index = movie_to_index[movie_id]
    neighbor_count = min(21, normalized_item_matrix.shape[0])
    distances, indices = item_model.kneighbors(
        normalized_item_matrix[movie_index], n_neighbors=neighbor_count
    )
    rows = []
    for distance, neighbor_index in zip(distances[0], indices[0]):
        similar_id = index_to_movie[int(neighbor_index)]
        similarity = float(1.0 - distance)
        if similar_id == movie_id or similarity <= 0 or similar_id not in movie_lookup.index:
            continue
        movie = movie_lookup.loc[similar_id]
        rows.append({
            "movieId": int(similar_id), "title": str(movie["title"]),
            "genres": "" if pd.isna(movie["genres"]) else str(movie["genres"]),
            "imdbId": movie.get("imdbId"), "tmdbId": movie.get("tmdbId"),
            "similarity": similarity,
        })
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows)


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


def get_intent_match_details(movie_genres, selected_genres=None, selected_mood=None, reference_similarity=None):
    """Return a rounded, explanatory score from explicit, verifiable intent signals.

    This is presentation-only. It does not feed context_score or recommendation ranking.
    """
    selected_genres = list(dict.fromkeys(selected_genres or []))
    mood_genres = MOOD_GENRES.get(selected_mood, set()) if selected_mood else set()
    actual = set()
    if isinstance(movie_genres, str) and movie_genres.strip():
        actual = {
            genre.strip()
            for genre in movie_genres.split("|")
            if genre.strip() and genre.strip() != "(no genres listed)"
        }

    signals = []
    explanations = []
    if selected_genres:
        matched = [
            genre for genre in selected_genres
            if ({"Romance", "Comedy"}.issubset(actual) if genre == "Rom-Com" else genre in actual)
        ]
        coverage = len(matched) / len(selected_genres)
        has_reference_signal = reference_similarity is not None and pd.notna(reference_similarity)
        genre_weight = 0.60 if selected_mood and has_reference_signal else 0.65 if selected_mood or has_reference_signal else 0.85
        signals.append((coverage, genre_weight))
        if coverage == 1:
            explanations.append(f"Strong {', '.join(selected_genres)} genre match")
        elif matched:
            explanations.append(f"Matches {len(matched)} of {len(selected_genres)} requested genres ({', '.join(matched)})")
        else:
            explanations.append("No requested genre appears in its local metadata")

    if selected_mood and mood_genres:
        mood_coverage = len(actual & mood_genres) / len(mood_genres)
        has_reference_signal = reference_similarity is not None and pd.notna(reference_similarity)
        mood_weight = 0.18 if selected_genres and has_reference_signal else 0.20 if selected_genres else 0.65 if has_reference_signal else 0.85
        signals.append((mood_coverage, mood_weight))
        if mood_coverage > 0:
            explanations.append(f"Local genre metadata supports {selected_mood}")

    if reference_similarity is not None and pd.notna(reference_similarity):
        similarity = float(np.clip(reference_similarity, 0.0, 1.0))
        reference_weight = (
            0.12 if selected_genres or selected_mood
            else 0.75
        )
        signals.append((similarity, reference_weight))
        if similarity > 0:
            explanations.append("Related to your reference movie in MovieMind’s item-similarity model")

    if not signals:
        return None, None
    score = round(100 * sum(value * weight for value, weight in signals))
    return int(np.clip(score, 0, 100)), "; ".join(explanations) or "No verified metadata match"


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
    clear_ai_request_state()


def clear_ai_request_state(clear_recommendations=False):
    """Remove only submitted AI state; manual genre and mood widgets remain untouched."""
    for key in (
        "ai_intent_summary",
        "ai_reference_movie_id",
        "ai_reference_results",
        "ai_reference_missing",
        "ai_request_succeeded",
        "ai_request_error",
    ):
        st.session_state.pop(key, None)
    if st.session_state.get("recommendation_source") == "ai":
        st.session_state["recommendation_source"] = "manual"
        clear_recommendations = True
    if clear_recommendations:
        st.session_state["recommendations"] = pd.DataFrame()
        st.session_state["surprise_result"] = False


def clear_ai_state_for_manual_context():
    """Keep AI summaries from being presented as the current manual request."""
    clear_ai_request_state(clear_recommendations=True)


def set_active_view(view):
    st.session_state["active_view"] = view


def is_gemini_rate_limit_error(error):
    """Recognize SDK and HTTP 429 errors without exposing the exception text."""
    response = getattr(error, "response", None)
    return (
        getattr(error, "code", None) == 429
        or getattr(error, "status_code", None) == 429
        or getattr(response, "status_code", None) == 429
        or "429" in str(error)
        or "rate limit" in str(error).casefold()
    )


def normalize_ai_genres(genres):
    """Map Gemini's genre labels onto values understood by local MovieMind context."""
    aliases = {
        "science fiction": "Sci-Fi", "sci fi": "Sci-Fi", "sci-fi": "Sci-Fi",
        "family": "Children", "children's": "Children", "film noir": "Film-Noir",
        "rom com": "Rom-Com", "rom-com": "Rom-Com",
    }
    by_casefold = {genre.casefold(): genre for genre in DATASET_GENRES}
    normalized = []
    for value in genres or []:
        candidate = str(value).strip()
        if not candidate:
            continue
        canonical = aliases.get(candidate.casefold(), by_casefold.get(candidate.casefold(), candidate))
        if canonical == "Rom-Com" and "Rom-Com" not in GENRE_OPTIONS:
            canonical = "Comedy"
        if canonical in DATASET_GENRES and canonical not in normalized:
            normalized.append(canonical)
    return normalized


def normalize_ai_mood(mood):
    """Map supported intent words to the existing mood-to-genre mapping."""
    if not mood:
        return None
    key = re.sub(r"[^a-z]+", " ", str(mood).casefold()).strip()
    aliases = {
        "funny": "Funny", "humorous": "Funny", "comedic": "Funny",
        "romantic": "Romantic", "love": "Romantic",
        "feel good": "Feel-Good", "uplifting": "Feel-Good",
        "emotional": "Emotional", "moving": "Emotional",
        "relaxing": "Relaxing", "calm": "Relaxing", "cozy": "Relaxing",
        "exciting": "Exciting", "adventurous": "Exciting",
        "dark": "Dark", "suspenseful": "Suspenseful", "tense": "Suspenseful",
    }
    return aliases.get(key)


def find_local_movie_id(title):
    """Resolve a reference title against the local catalog without external search."""
    if not title:
        return None

    def normalize_title(value):
        value = re.sub(r"\s*\(\d{4}\)\s*$", "", str(value))
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    requested = normalize_title(title)
    if not requested:
        return None
    matches = movies.loc[movies["title"].map(normalize_title) == requested, "movieId"]
    return int(matches.iloc[0]) if not matches.empty else None


def signal_explanation(movie, mode, context):
    """Build a restrained explanation from the recommendation's stored signals."""
    if mode == "current":
        labels = list(context.get("genres", []))
        mood = context.get("mood")
        intent = " + ".join(labels) if labels else "your selected mood"
        if mood:
            intent = f"{intent} + {mood}" if labels else mood
        return f"Its local genre metadata aligns with your current {intent} viewing intent."

    weighted = {
        "Taste Match": .30 * float(movie.get("user_score", 0) or 0),
        "Movie Similarity": .30 * float(movie.get("item_score", 0) or 0),
        "Preference Learning": .40 * float(movie.get("svd_score", 0) or 0),
    }
    strongest = max(weighted, key=weighted.get)
    explanations = {
        "Taste Match": "User-based collaborative filtering found rating patterns similar to your profile.",
        "Movie Similarity": "Item-based collaborative filtering found similarity to movies in your profile.",
        "Preference Learning": "Preference Learning found a strong signal from the ratings in your profile.",
    }
    explanation = explanations[strongest]
    if mode == "combined" and bool(movie.get("context_applied", False)) and float(movie.get("context_score", 0) or 0) > 0:
        labels = list(context.get("genres", []))
        mood = context.get("mood")
        intent = " + ".join(labels) if labels else "your selected mood"
        if mood:
            intent = f"{intent} + {mood}" if labels else mood
        explanation = f"Its genre metadata aligns with your current {intent} viewing intent; {explanation[0].lower() + explanation[1:]}"
    return f"{explanation} Main signal: {strongest}."


def image_data_uri(image, fallback_svg=POSTER_PLACEHOLDER):
    """Create a self-contained image URI for compact, styled movie cards."""
    if image:
        raw = image if isinstance(image, bytes) else str(image).encode("utf-8")
        mime = "image/jpeg" if isinstance(image, bytes) else "image/svg+xml"
    else:
        raw = fallback_svg.encode("utf-8")
        mime = "image/svg+xml"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def render_movie_card(movie, mode, context, row_key):
    movie_id = int(movie["movieId"])
    title, year = movie_title_year(movie["title"])
    metadata = get_movie_details(movie.get("tmdbId"))
    poster = get_movie_poster_image((metadata or {}).get("poster_path"))
    is_intent = mode == "current" and movie.get("intent_match") is not None and pd.notna(movie.get("intent_match"))
    score = (float(movie.get("intent_match")) / 100) if is_intent else float(movie.get("hybrid_score", movie.get("similarity", 0)) or 0)
    score_label = "Intent Match" if is_intent else ("Similarity" if "similarity" in movie else "Match")
    genres = str(movie.get("genres", "")).replace("|", " · ")
    markup = (
        '<div class="movie-card">'
        f'<img class="movie-poster" src="{image_data_uri(poster)}" alt="Movie poster">'
        '<div class="movie-card-info">'
        f'<div class="movie-card-title">{html.escape(title)}</div>'
        f'<div class="movie-card-meta">{html.escape(year if year != "Year unavailable" else genres)}</div>'
        f'<div class="movie-card-match">{html.escape(score_label)} · {score:.0%}</div>'
        + (f'<div class="movie-card-intent">{html.escape(str(movie.get("intent_explanation", "")))}</div>' if is_intent and movie.get("intent_explanation") else "")
        + '</div></div>'
    )
    st.markdown(markup, unsafe_allow_html=True)
    details_col, similar_col = st.columns(2)
    if details_col.button("Details", key=f"details_{row_key}_{movie_id}", use_container_width=True):
        st.session_state["selected_hero_movie_id"] = movie_id
        st.session_state["details_movie_id"] = movie_id
    if similar_col.button("More Like", key=f"similar_{row_key}_{movie_id}", use_container_width=True):
        st.session_state["more_like_movie_id"] = movie_id
        st.session_state["more_like_results"] = get_similar_movies(movie_id)
        st.session_state["more_like_source_title"] = str(movie["title"])


def render_movie_row(title, frame, mode, context, row_key, caption=None):
    if frame is None or frame.empty:
        return
    st.markdown(f'<div class="row-heading">{html.escape(title)}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="row-caption">{html.escape(caption)}</div>', unsafe_allow_html=True)
    rows = [row for _, row in frame.iterrows()]
    for start in range(0, len(rows), 5):
        columns = st.columns(5)
        for column, movie in zip(columns, rows[start:start + 5]):
            with column:
                render_movie_card(movie, mode, context, row_key)


def render_hero(movie, mode, context):
    movie_id = int(movie["movieId"])
    metadata = get_movie_details(movie.get("tmdbId"))
    backdrop = get_movie_backdrop_image((metadata or {}).get("backdrop_path"))
    backdrop_uri = image_data_uri(backdrop) if backdrop else "linear-gradient(125deg,#19283b,#21172b 58%,#11151d)"
    title, year = movie_title_year(movie["title"])
    genres = str(movie.get("genres", "")).replace("|", " · ")
    is_intent = mode == "current" and movie.get("intent_match") is not None and pd.notna(movie.get("intent_match"))
    score = (float(movie.get("intent_match")) / 100) if is_intent else float(movie.get("hybrid_score", 0) or 0)
    score_title = "VIEWING INTENT" if is_intent else "MOVIEMIND MATCH"
    release_year = year if year != "Year unavailable" else "Release year unavailable"
    overview = (metadata or {}).get("overview", "").strip()
    why = signal_explanation(movie, mode, context)
    hero_html = f"""
    <section class="cinema-hero" style='--hero-bg:url("{backdrop_uri}")'>
      <div class="cinema-copy">
        <div class="cinema-kicker">{score_title} · FEATURED PICK</div>
        <div style="margin:10px 0"><span class="match-pill">{score:.0%} {"Intent Match" if is_intent else "Match"}</span></div>
        <div class="cinema-title">{html.escape(title)}</div>
        <div class="cinema-meta">{html.escape(release_year)} &nbsp;·&nbsp; {html.escape(genres)}</div>
        <div class="cinema-overview">{html.escape(overview) if overview else "Overview unavailable from TMDB."}</div>
        <div class="cinema-why"><b>Why MovieMind picked this</b><br>{html.escape(why)}</div>
      </div>
    </section>
    """
    st.markdown(hero_html, unsafe_allow_html=True)
    view_col, similar_col, spacer = st.columns([1.2, 1.3, 5])
    if view_col.button("▶  View Details", type="primary", key=f"hero_details_{movie_id}"):
        st.session_state["details_movie_id"] = movie_id
    if similar_col.button("＋  More Like This", key=f"hero_similar_{movie_id}"):
        st.session_state["more_like_movie_id"] = movie_id
        st.session_state["more_like_results"] = get_similar_movies(movie_id)
        st.session_state["more_like_source_title"] = str(movie["title"])


# ============================================================
# APPLICATION
# ============================================================

profiles = load_profiles()
view_names = {
    "recommendations": "Recommendations",
    "my-taste": "My Taste",
    "rating-history": "Rating History",
    "model-performance": "Model Performance",
    "about": "About MovieMind",
}
active_view = st.session_state.get("active_view", "Recommendations")
if active_view not in view_names.values():
    active_view = "Recommendations"
    st.session_state["active_view"] = active_view
discover_views = ["Recommendations", "My Taste", "Rating History"]

st.markdown(
    '<div class="brand-bar"><div class="brand-mark">M</div><div><div class="brand-name">MovieMind</div><div class="brand-sub">Personalized Movie Discovery</div></div><div class="brand-ai"><span class="brand-ai-dot">●</span>AI-powered discovery</div></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand"><div class="sidebar-brand-mark">M</div><div><div class="sidebar-brand-name">MovieMind</div><div class="sidebar-brand-caption">AI movie discovery</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-section-label">DISCOVER</div>', unsafe_allow_html=True)
    nav_icon = {
        "Recommendations": "⌂",
        "My Taste": "♡",
        "Rating History": "◷",
        "Model Performance": "▤",
        "About MovieMind": "ⓘ",
    }

    def sidebar_nav_item(view, key):
        marker = "▏" if active_view == view else " "
        button_type = "primary" if active_view == view else "secondary"
        st.button(
            f"{marker}  {nav_icon[view]}   {view}",
            key=key,
            type=button_type,
            use_container_width=True,
            on_click=set_active_view,
            args=(view,),
        )

    sidebar_nav_item("Recommendations", "nav_recommendations")
    sidebar_nav_item("My Taste", "nav_my_taste")
    sidebar_nav_item("Rating History", "nav_rating_history")
    st.markdown('<div class="sidebar-section-label">INSIGHTS</div>', unsafe_allow_html=True)
    sidebar_nav_item("Model Performance", "nav_model_performance")
    st.markdown('<div class="sidebar-section-label">ABOUT</div>', unsafe_allow_html=True)
    sidebar_nav_item("About MovieMind", "nav_about")
    current_profile_pairs = st.session_state.get("profile", [])
    current_profile_record = next(
        (profile for profile in profiles if profile.get("profile_id") == st.session_state.get("profile_id")),
        None,
    )
    current_profile_name = (
        current_profile_record.get("name", "Personal taste")
        if current_profile_pairs and current_profile_record
        else "Personal taste" if current_profile_pairs else "No active profile"
    )
    profile_meta = f"Taste profile active · {len(current_profile_pairs)} rated movies" if current_profile_pairs else "Rate movies to personalize recommendations"
    st.markdown(
        f'<div class="sidebar-profile"><div class="sidebar-profile-title">PROFILE</div><div class="sidebar-profile-name">{html.escape(current_profile_name)}</div><div class="sidebar-profile-meta">{html.escape(profile_meta)}</div></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"{len(profiles)} saved profile(s) on this computer")
    profile_labels = {p.get("profile_id"): p.get("name", "Unnamed") for p in profiles if p.get("profile_id")}
    chosen_profile_id = st.selectbox(
        "Switch Profile",
        [""] + list(profile_labels),
        format_func=lambda value: "Switch profile…" if not value else profile_labels.get(value, value),
        label_visibility="collapsed",
        key="sidebar_profile_selector",
    )
    if st.button("Switch Profile", disabled=not chosen_profile_id, use_container_width=True):
        match = next((p for p in profiles if p.get("profile_id") == chosen_profile_id), None)
        if match:
            apply_profile(match)
            st.rerun()

st.markdown('<div class="ai-panel"><div class="ai-kicker">Ask MovieMind AI</div><div class="ai-heading">What are you in the mood to watch?</div><div class="ai-sub">Describe the kind of movie you want. MovieMind will translate your request into its existing genre and mood inputs.</div></div>', unsafe_allow_html=True)
with st.form("ai_intent_form", clear_on_submit=False):
    ai_prompt_col, ai_submit_col = st.columns([5, 1])
    with ai_prompt_col:
        ai_prompt = st.text_input("What do you want to watch?", placeholder="Try: An interesting thriller movie", label_visibility="collapsed", key="ai_prompt")
    with ai_submit_col:
        ai_submit = st.form_submit_button("✨ Find Movies", type="primary", use_container_width=True)
st.markdown('<div class="ai-examples"><span class="ai-example-label">TRY</span> “Something funny for tonight” &nbsp;·&nbsp; “Like Inception but more action” &nbsp;·&nbsp; “A dark psychological thriller” &nbsp;·&nbsp; “Feel-good movies for the weekend”</div>', unsafe_allow_html=True)

ai_context = None
if ai_submit:
    clear_ai_request_state(clear_recommendations=True)
    st.session_state["recommendation_source"] = "manual"
    st.session_state["ai_request_succeeded"] = False
    if not ai_prompt.strip():
        st.markdown('<div class="ai-error-state">Enter a movie request to ask MovieMind AI.</div>', unsafe_allow_html=True)
    elif parse_movie_intent is None:
        st.session_state["ai_request_error"] = "unavailable"
        st.markdown('<div class="ai-error-state">MovieMind AI is temporarily unavailable. You can continue using manual preferences below.</div>', unsafe_allow_html=True)
    else:
        try:
            with st.spinner("Understanding your movie request…"):
                parsed_intent = parse_movie_intent(ai_prompt.strip())
            raw_genres = list(parsed_intent.genres or [])
            mapped_genres = normalize_ai_genres(raw_genres)
            mapped_mood = normalize_ai_mood(parsed_intent.mood)
            unsupported_genres = [genre for genre in raw_genres if not normalize_ai_genres([genre])]
            reference_movie = (parsed_intent.reference_movie or "").strip() or None
            reference_id = find_local_movie_id(reference_movie)
            ai_context = {"genres": mapped_genres, "mood": mapped_mood}
            summary = {
                **ai_context,
                "reference_movie": reference_movie,
                "keywords": [str(word) for word in (parsed_intent.keywords or []) if str(word).strip()],
                "unsupported_genres": unsupported_genres,
                "unsupported_mood": bool(parsed_intent.mood and not mapped_mood),
            }
            st.session_state["ai_intent_summary"] = summary
            st.session_state["ai_request_succeeded"] = True
            if reference_id is not None:
                st.session_state["ai_reference_movie_id"] = reference_id
                st.session_state["ai_reference_results"] = get_similar_movies(reference_id)
                st.session_state["ai_reference_missing"] = False
            else:
                st.session_state.pop("ai_reference_movie_id", None)
                st.session_state.pop("ai_reference_results", None)
                st.session_state["ai_reference_missing"] = bool(reference_movie)
            intent_chips = [*mapped_genres]
            if mapped_mood:
                intent_chips.append(mapped_mood)
            intent_chips.extend(summary["keywords"])
            if reference_movie:
                intent_chips.append(f"Reference: {reference_movie}")
            status_markup = "".join(
                f'<span class="intent-chip">{html.escape(str(chip))}</span>'
                for chip in intent_chips
            ) or '<span class="intent-chip">No supported genre or mood signal</span>'
            st.markdown(
                f'<div class="ai-status"><span class="ai-status-label">MOVIEMIND UNDERSTANDS</span>{status_markup}</div>',
                unsafe_allow_html=True,
            )
            if summary["unsupported_genres"] or summary["unsupported_mood"]:
                st.caption("Some intent terms were not available in MovieMind’s context and were skipped.")
        except Exception as exc:
            ai_context = None
            clear_ai_request_state(clear_recommendations=True)
            st.session_state["recommendation_source"] = "manual"
            st.session_state["ai_request_succeeded"] = False
            error_markup = (
                "MovieMind AI has temporarily reached its API request limit. You can continue using manual preferences below."
                if is_gemini_rate_limit_error(exc)
                else "MovieMind AI is temporarily unavailable. You can continue using manual preferences below."
            )
            st.session_state["ai_request_error"] = "rate_limit" if is_gemini_rate_limit_error(exc) else "unavailable"
            error_class = "ai-error-state ai-429-state" if is_gemini_rate_limit_error(exc) else "ai-error-state"
            st.markdown(f'<div class="{error_class}">{html.escape(error_markup)}</div>', unsafe_allow_html=True)
            diagnostic_message = f"{type(exc).__name__}: {exc}"
            for secret_name in ("GEMINI_API_KEY", "TMDB_API_KEY"):
                try:
                    secret_value = st.secrets.get(secret_name)
                except Exception:
                    secret_value = None
                if secret_value:
                    diagnostic_message = diagnostic_message.replace(str(secret_value), "[REDACTED]")
            diagnostic_message = re.sub(
                r"(?i)(api[_ -]?key|token|secret|password|authorization)(\s*[:=]\s*|\s+)[^\s,;]+",
                r"\1\2[REDACTED]",
                diagnostic_message,
            )
            with st.expander("Temporary Gemini diagnostic"):
                st.caption(f"Exception type: {type(exc).__name__}")
                st.code(diagnostic_message)
elif st.session_state.get("ai_request_succeeded") and st.session_state.get("recommendation_source") == "ai":
    summary = st.session_state.get("ai_intent_summary", {})
    intent_chips = [*summary.get("genres", [])]
    if summary.get("mood"):
        intent_chips.append(summary["mood"])
    intent_chips.extend(summary.get("keywords", []))
    if summary.get("reference_movie"):
        intent_chips.append(f"Reference: {summary['reference_movie']}")
    status_markup = "".join(
        f'<span class="intent-chip">{html.escape(str(chip))}</span>'
        for chip in intent_chips
    ) or '<span class="intent-chip">No supported genre or mood signal</span>'
    st.markdown(
        f'<div class="ai-status"><span class="ai-status-label">MOVIEMIND UNDERSTANDS</span>{status_markup}</div>',
        unsafe_allow_html=True,
    )

with st.expander("Refine your search", expanded=False):
    st.markdown('<div class="manual-refine-label">REFINE YOUR SEARCH · MANUAL PREFERENCES</div>', unsafe_allow_html=True)
    context_genre_col, context_mood_col = st.columns(2)
    with context_genre_col:
        selected_genres = st.multiselect("Genre Preference", options=GENRE_OPTIONS, placeholder="Any", key="context_genres", on_change=clear_ai_state_for_manual_context)
    with context_mood_col:
        selected_mood_option = st.selectbox("Mood", options=["Any", *MOOD_GENRES], key="context_mood", on_change=clear_ai_state_for_manual_context)
selected_mood = None if selected_mood_option == "Any" else selected_mood_option

with st.expander("Personalize using my taste", expanded=False):
    st.caption("Rate 5 to 10 movies. Your profile uses the existing 30/30/40 hybrid recommender.")
    st.text_input("Profile name", key="profile_name", placeholder="Enter your name")
    selected_movies = st.multiselect("Search the MovieMind catalog", options=list(movie_label_to_id), max_selections=10, placeholder="Type a title, for example Interstellar (2014)...", key="selected_movies", help="Searches the local MovieMind movie dataset. Movie IDs distinguish duplicate titles.")
    if selected_movies:
        st.markdown("#### Rate your movies")
        st.caption("Each rating uses half-star steps from 0.5 to 5.0.")
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
                    poster_col, details_col = st.columns([1, 2])
                    metadata = get_movie_details(movie_info.get("tmdbId"))
                    poster = get_movie_poster_image((metadata or {}).get("poster_path"))
                    poster_col.image(poster or POSTER_PLACEHOLDER, use_container_width=True)
                    title, year = movie_title_year(movie_info["title"])
                    details_col.markdown(f"**{title}**")
                    details_col.caption(year)
                    details_col.caption(str(movie_info["genres"]).replace("|", " · "))
                    st.slider("Your rating (0.5–5.0)", 0.5, 5.0, 4.0, 0.5, key=f"taste_rating_{movie_id}")

profile_preview = [(movie_label_to_id[label], float(st.session_state.get(f"taste_rating_{movie_label_to_id[label]}", 4.0))) for label in selected_movies]
if profile_preview:
    genre_preview = get_genre_profile(profile_preview)
    leading_genre = genre_preview.iloc[0]["Genre"] if not genre_preview.empty else "Not available"
    m1, m2, m3 = st.columns(3)
    m1.metric("Movies Selected", len(profile_preview))
    m2.metric("Average Rating", f"{np.mean([r for _, r in profile_preview]):.1f} / 5")
    m3.metric("Leading Genre", leading_genre)

action_cols = st.columns([2, 1, 1, 1])
generate = action_cols[0].button("Show Recommendations", type="primary", use_container_width=True, on_click=clear_ai_state_for_manual_context)
surprise = action_cols[1].button("Surprise Me", use_container_width=True, on_click=clear_ai_state_for_manual_context)
save_profile = action_cols[2].button("Save Profile", use_container_width=True)
reset = action_cols[3].button("Reset Profile", use_container_width=True, on_click=reset_profile_state)

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

effective_genres = ai_context["genres"] if ai_context is not None else list(selected_genres)
effective_mood = ai_context["mood"] if ai_context is not None else selected_mood

if generate or surprise or ai_context is not None:
    active_saved_profile = st.session_state.get("profile", [])
    use_profile = profile_preview if 5 <= len(profile_preview) <= 10 else active_saved_profile
    valid_profile = 5 <= len(use_profile) <= 10
    has_context = bool(effective_genres or effective_mood)
    source = "ai" if ai_context is not None else "manual"
    if not valid_profile and not has_context:
        st.session_state["recommendations"] = pd.DataFrame()
        st.session_state["recommendation_source"] = source
        st.session_state["recommendation_context"] = {"genres": [], "mood": None}
        if st.session_state.get("ai_reference_movie_id") is None:
            st.warning("Choose a genre or mood, rate 5 to 10 movies, or describe a movie intent for MovieMind AI.")
        else:
            st.info("No local genre or mood signal was found. MovieMind can still show catalog similarity for a recognized reference movie.")
    else:
        if valid_profile:
            mode = "combined" if has_context else "personalized"
            with st.spinner("Finding movies with your MovieMind profile..."):
                result_frame = get_recommendations(use_profile, selected_genres=effective_genres, selected_mood=effective_mood)
            st.session_state["profile"] = use_profile
        else:
            mode = "current"
            with st.spinner("Finding movies for your viewing intent..."):
                result_frame = get_current_viewing_recommendations(selected_genres=effective_genres, selected_mood=effective_mood)
            st.session_state["profile"] = []
        st.session_state["recommendations"] = result_frame.head(1) if surprise else result_frame
        st.session_state["recommendation_context"] = {"genres": list(effective_genres), "mood": effective_mood}
        st.session_state["recommendation_mode"] = mode
        st.session_state["recommendation_source"] = source
        st.session_state["recommendation_profile"] = list(use_profile) if valid_profile else []
        st.session_state["surprise_result"] = bool(surprise)
        st.session_state.pop("more_like_movie_id", None)
        st.session_state.pop("more_like_results", None)
        st.session_state.pop("details_movie_id", None)
        st.session_state.pop("selected_hero_movie_id", None)
        if surprise:
            st.success("Here’s a MovieMind recommendation based on your current profile and viewing intent.")
        elif source == "ai":
            st.success("MovieMind translated your request into its existing viewing context and recommendation system.")
        else:
            st.success("Your recommendations are ready.")

active_profile = profile_preview
stored_profile = st.session_state.get("profile", [])
recommendations = st.session_state.get("recommendations", pd.DataFrame())
generated_context = st.session_state.get("recommendation_context", {"genres": [], "mood": None})
generated_mode = st.session_state.get("recommendation_mode", "personalized")
if (
    (generated_mode != "current" and stored_profile != profile_preview)
    or (st.session_state.get("recommendation_source") != "ai" and (
        generated_context.get("genres", []) != list(selected_genres)
        or generated_context.get("mood") != selected_mood
    ))
    or (generated_mode != "current" and st.session_state.get("recommendation_profile", []) != list(profile_preview))
):
    recommendations = pd.DataFrame()
taste_rows = [
    {"Movie": movie_lookup.loc[mid, "title"], "Rating": rating, "Genres": str(movie_lookup.loc[mid, "genres"]).replace("|", " · ")}
    for mid, rating in active_profile if mid in movie_lookup.index
]
taste_df = pd.DataFrame(taste_rows)

if active_view == "Recommendations":
    current_mode = st.session_state.get("recommendation_mode", "personalized")
    recommendation_source = st.session_state.get("recommendation_source", "manual")
    current_context = st.session_state.get("recommendation_context", {"genres": [], "mood": None})
    context_genres = current_context.get("genres", [])
    context_mood = current_context.get("mood")
    has_context = bool(context_genres or context_mood)
    reference_results = st.session_state.get("ai_reference_results", pd.DataFrame())
    reference_similarities = (
        dict(zip(reference_results["movieId"].astype(int), reference_results["similarity"].astype(float)))
        if isinstance(reference_results, pd.DataFrame)
        and not reference_results.empty
        and {"movieId", "similarity"}.issubset(reference_results.columns)
        else {}
    )
    if not recommendations.empty and (has_context or reference_similarities):
        recommendations = recommendations.copy()
        intent_details = [
            get_intent_match_details(
                movie.get("genres", ""),
                context_genres,
                context_mood,
                reference_similarities.get(int(movie["movieId"])),
            )
            for _, movie in recommendations.iterrows()
        ]
        recommendations["intent_match"] = [item[0] for item in intent_details]
        recommendations["intent_explanation"] = [item[1] for item in intent_details]
    if recommendation_source == "ai":
        st.markdown("### MovieMind understood")
        summary = st.session_state.get("ai_intent_summary", {})
        chips = [*summary.get("genres", [])]
        if summary.get("mood"):
            chips.append(summary["mood"])
        if summary.get("reference_movie"):
            chips.append(f"Reference: {summary['reference_movie']}")
        chips.extend(summary.get("keywords", []))
        st.markdown("".join(f'<span class="intent-chip">{html.escape(str(chip))}</span>' for chip in chips) or '<span class="intent-chip">No supported genre or mood signal</span>', unsafe_allow_html=True)
        if summary.get("unsupported_genres") or summary.get("unsupported_mood"):
            st.caption("Some intent terms were not available in the local MovieMind context and were skipped.")
        if st.session_state.get("ai_reference_missing"):
            st.info("Reference movie not found in the MovieMind catalog. The rest of your request is still active.")
    if recommendations.empty:
        st.markdown('<div class="row-heading">Your movie night, your way</div><div class="row-caption">Start with a natural-language request, choose a genre or mood, or build a taste profile.</div>', unsafe_allow_html=True)
        if recommendation_source == "ai" and st.session_state.get("ai_reference_movie_id") is not None:
            reference_id = int(st.session_state["ai_reference_movie_id"])
            reference_title = movie_lookup.loc[reference_id, "title"]
            reference_results = st.session_state.get("ai_reference_results", pd.DataFrame())
            render_movie_row(f"Because you mentioned {reference_title}", reference_results, "similarity", current_context, "ai_reference_empty", "Neighbors from MovieMind’s existing item-similarity model.")
    else:
        st.caption("MovieMind Match is a normalized recommendation score, not a probability." if current_mode != "current" else "Viewing Intent Match summarizes verified catalog signals. It is separate from recommendation ranking.")
        if current_context.get("genres") or current_context.get("mood"):
            st.markdown("".join(f'<span class="intent-chip">{html.escape(str(chip))}</span>' for chip in [*context_genres, *([context_mood] if context_mood else [])]), unsafe_allow_html=True)
        hero_id = st.session_state.get("selected_hero_movie_id")
        hero_rows = recommendations.loc[recommendations["movieId"].astype(int) == int(hero_id)] if hero_id is not None else pd.DataFrame()
        hero_movie = hero_rows.iloc[0] if not hero_rows.empty else recommendations.iloc[0]
        render_hero(hero_movie, current_mode, current_context)

        details_id = st.session_state.get("details_movie_id")
        if details_id is not None:
            detail_rows = recommendations.loc[recommendations["movieId"].astype(int) == int(details_id)]
            if not detail_rows.empty:
                detail_movie = detail_rows.iloc[0]
                detail_metadata = get_movie_details(detail_movie.get("tmdbId"))
                detail_poster = get_movie_poster_image((detail_metadata or {}).get("poster_path"))
                detail_backdrop = get_movie_backdrop_image((detail_metadata or {}).get("backdrop_path"))
                detail_left, detail_right = st.columns([1, 3])
                with detail_left:
                    st.image(detail_poster or POSTER_PLACEHOLDER, use_container_width=True)
                with detail_right:
                    title, year = movie_title_year(detail_movie["title"])
                    st.markdown(f"### {title}")
                    st.caption(f"{year} · {str(detail_movie['genres']).replace('|', ' · ')}")
                    if detail_backdrop:
                        st.image(detail_backdrop, use_container_width=True)
                    overview = (detail_metadata or {}).get("overview", "").strip()
                    if overview:
                        st.write(overview)
                    if current_mode == "current" and detail_movie.get("intent_match") is not None and pd.notna(detail_movie.get("intent_match")):
                        st.metric("Viewing Intent Match", f"{int(detail_movie['intent_match'])}%")
                        if detail_movie.get("intent_explanation"):
                            st.caption(str(detail_movie["intent_explanation"]))
                    else:
                        st.metric("MovieMind Match", f"{float(detail_movie['hybrid_score']):.0%}")
                    st.markdown(f"**Why MovieMind picked this**  \n{signal_explanation(detail_movie, current_mode, current_context)}")
                    if current_mode != "current":
                        details = st.expander("View model details")
                        with details:
                            st.write(f"Taste Match contribution: {0.30 * float(detail_movie['user_score']):.1%}")
                            st.write(f"Movie Similarity contribution: {0.30 * float(detail_movie['item_score']):.1%}")
                            st.write(f"Preference Learning contribution: {0.40 * float(detail_movie['svd_score']):.1%}")
                    detail_links = []
                    if pd.notna(detail_movie.get("imdbId")) and int(detail_movie["imdbId"]) > 0:
                        detail_links.append(f"[IMDb](https://www.imdb.com/title/tt{int(detail_movie['imdbId']):07d}/)")
                    if pd.notna(detail_movie.get("tmdbId")) and int(detail_movie["tmdbId"]) > 0:
                        detail_links.append(f"[TMDB](https://www.themoviedb.org/movie/{int(detail_movie['tmdbId'])})")
                    if detail_links:
                        st.markdown(" · ".join(detail_links))
                    close_col, _ = st.columns([1, 4])
                    if close_col.button("Close details", key=f"close_details_{int(detail_movie['movieId'])}"):
                        st.session_state.pop("details_movie_id", None)

        if recommendation_source == "ai" and st.session_state.get("ai_reference_movie_id") is not None:
            reference_id = int(st.session_state["ai_reference_movie_id"])
            reference_title = movie_lookup.loc[reference_id, "title"]
            reference_results = st.session_state.get("ai_reference_results", pd.DataFrame())
            render_movie_row(f"Because you mentioned {reference_title}", reference_results, "similarity", current_context, "ai_reference", "Neighbors from MovieMind’s existing item-similarity model.")

        if st.session_state.get("surprise_result"):
            main_title = "Your Surprise Pick"
        elif recommendation_source == "ai":
            main_title = "AI-Powered Viewing Results"
        elif current_mode == "current":
            main_title = "Current Viewing"
        else:
            main_title = "Highly Matched For You"
        render_movie_row(main_title, recommendations, current_mode, current_context, "primary", "Ranked by the existing MovieMind recommendation system.")

        if current_mode != "current" and len(recommendations) >= 3:
            profile_strength = np.maximum(.30 * recommendations["user_score"].astype(float), .30 * recommendations["item_score"].astype(float))
            because = recommendations.loc[profile_strength > 0].copy()
            if len(because) >= 3:
                because["_profile_signal"] = np.maximum(.30 * because["user_score"].astype(float), .30 * because["item_score"].astype(float))
                because = because.sort_values("_profile_signal", ascending=False).head(5).drop(columns="_profile_signal")
                render_movie_row("Because You Like…", because, current_mode, current_context, "profile", "Movies with existing Taste Match or Movie Similarity contribution in this profile run.")

        if has_context and current_mode != "current" and len(recommendations) >= 3:
            current = recommendations.loc[recommendations["context_score"].fillna(0).astype(float) > 0].head(5)
            if len(current) >= 3:
                render_movie_row("Current Viewing", current, "current", current_context, "context", "Intent match summarizes verified genre and mood signals; recommendation ranking is unchanged.")

        more_like_id = st.session_state.get("more_like_movie_id")
        if more_like_id is not None:
            more_like_results = st.session_state.get("more_like_results", pd.DataFrame())
            source_title = st.session_state.get("more_like_source_title") or movie_lookup.loc[more_like_id, "title"]
            render_movie_row(f"More Like {source_title}", more_like_results, "similarity", current_context, "more_like", "Cosine similarity from the existing Item-Based CF model; separate from MovieMind Match.")

if active_view == "My Taste":
    st.markdown("### Selected movies and ratings")
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

if active_view == "Rating History":
    st.markdown("### Rating History")
    if taste_df.empty:
        st.info("Ratings for the current profile will appear here.")
    else:
        st.dataframe(taste_df, use_container_width=True, hide_index=True)
        st.download_button("Download rating history as CSV", taste_df.to_csv(index=False).encode("utf-8"), "moviemind_rating_history.csv", "text/csv")

if active_view == "Model Performance":
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

if active_view == "About MovieMind":
    st.markdown("### What MovieMind does")
    st.write("MovieMind supports current viewing recommendations from genre and mood preferences without requiring ratings, personalized recommendations from a 5 to 10 movie taste profile, or a combination of both.")
    st.markdown("### Recommendation approach")
    st.write("MovieMind combines long-term taste preferences, collaborative filtering, matrix factorization, and current viewing intent. The core hybrid remains 30% Taste Match + 30% Movie Similarity + 40% Preference Learning. A ridge-regression step infers a temporary SVD profile for new visitors from their selected ratings.")
    st.write("Genre comes from movie metadata. Mood is derived from an editable mood-to-genre mapping; it is not a native dataset field. Context is an optional ranking adjustment, not a fourth ML model.")
    st.write("The internal hybrid score ranges from 0 to 1 and is displayed as a percentage. It is a normalized recommendation score, not a probability.")
    st.write("Gemini AI interprets natural-language viewing requests and maps supported genres and moods into MovieMind’s existing context inputs. The existing ML recommendation system ranks the results. TMDB supplies movie metadata and images.")
    st.markdown("### Data, storage, and technology")
    st.write(f"The development dataset contains {len(user_ids):,} users, {len(movies):,} movies, and {len(ratings):,} ratings. Ratings remain in a sparse matrix to avoid a large dense allocation. Saved profile names and movie ratings are stored locally in data/user_profiles.json; no passwords or sensitive information are requested.")
    st.write("Technology stack: Python, Streamlit, Pandas, NumPy, SciPy, scikit-learn, and Surprise.")
    st.markdown("### TMDB attribution")
    st.write("Movie metadata and images are provided by TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.")

st.divider()
st.markdown('<div class="footer-text">MovieMind | Personalized Hybrid Movie Recommendation System</div>', unsafe_allow_html=True)
