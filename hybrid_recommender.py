import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from surprise import SVD, Dataset, Reader


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

print("Loading cleaned data...")

ratings = pd.read_csv("data/ratings_clean.csv")
movies = pd.read_csv("data/movies_clean.csv")

print(f"Ratings: {len(ratings):,}")
print(f"Users: {ratings['userId'].nunique():,}")
print(f"Movies: {ratings['movieId'].nunique():,}")


# ============================================================
# STEP 2: CREATE USER-MOVIE MATRIX
# ============================================================

print("\nCreating user-movie matrix...")

user_ids = ratings["userId"].unique()
movie_ids = ratings["movieId"].unique()

user_to_index = {uid: i for i, uid in enumerate(user_ids)}
movie_to_index = {mid: i for i, mid in enumerate(movie_ids)}

row = ratings["userId"].map(user_to_index).values
col = ratings["movieId"].map(movie_to_index).values
data = ratings["rating"].values

user_movie_matrix = csr_matrix(
    (data, (row, col)),
    shape=(len(user_ids), len(movie_ids))
)

print(f"Matrix shape: {user_movie_matrix.shape}")


# ============================================================
# STEP 3: SELECT TARGET USER
# ============================================================

target_user = 10

if target_user not in user_to_index:
    raise ValueError("Target user not found.")

target_index = user_to_index[target_user]

user_ratings = ratings[ratings["userId"] == target_user]

print(f"\nTarget user: {target_user}")
print(f"Movies already rated: {len(user_ratings)}")
print(f"Average rating: {user_ratings['rating'].mean():.2f}")


# ============================================================
# STEP 4: USER-BASED CF
# ============================================================

print("\nRunning User-Based Collaborative Filtering...")

n_neighbors = 21

user_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=n_neighbors
)

user_model.fit(user_movie_matrix)

distances, indices = user_model.kneighbors(
    user_movie_matrix[target_index],
    n_neighbors=n_neighbors
)

similarities = 1 - distances[0]

neighbor_indices = indices[0][1:]
neighbor_similarities = similarities[1:]

user_avg = ratings.groupby("userId")["rating"].mean()

user_scores = {}

for neighbor_idx, similarity in zip(
    neighbor_indices,
    neighbor_similarities
):

    if similarity <= 0:
        continue

    neighbor_id = user_ids[neighbor_idx]

    neighbor_ratings = ratings[
        ratings["userId"] == neighbor_id
    ]

    neighbor_mean = user_avg[neighbor_id]

    for _, row_data in neighbor_ratings.iterrows():

        movie_id = row_data["movieId"]

        if movie_id in set(user_ratings["movieId"]):
            continue

        deviation = row_data["rating"] - neighbor_mean

        if movie_id not in user_scores:
            user_scores[movie_id] = {
                "weighted_sum": 0,
                "similarity_sum": 0
            }

        user_scores[movie_id]["weighted_sum"] += (
            similarity * deviation
        )

        user_scores[movie_id]["similarity_sum"] += similarity


target_mean = user_ratings["rating"].mean()

user_cf_scores = {}

for movie_id, values in user_scores.items():

    if values["similarity_sum"] == 0:
        continue

    prediction = (
        target_mean
        + values["weighted_sum"]
        / values["similarity_sum"]
    )

    prediction = np.clip(prediction, 0.5, 5.0)

    user_cf_scores[movie_id] = prediction

print(f"User-Based CF candidates: {len(user_cf_scores):,}")


# ============================================================
# STEP 5: ITEM-BASED CF
# ============================================================

print("\nRunning Item-Based Collaborative Filtering...")

movie_user_matrix = user_movie_matrix.T

normalized_movie_matrix = normalize(
    movie_user_matrix,
    axis=1
)

item_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=21
)

item_model.fit(normalized_movie_matrix)


# Use user's highest-rated movies as preference signals
top_rated = (
    user_ratings
    .sort_values("rating", ascending=False)
    .head(50)
)

item_scores = {}

for _, rated_movie in top_rated.iterrows():

    movie_id = rated_movie["movieId"]

    if movie_id not in movie_to_index:
        continue

    movie_index = movie_to_index[movie_id]

    distances, indices = item_model.kneighbors(
        normalized_movie_matrix[movie_index],
        n_neighbors=21
    )

    similarities = 1 - distances[0]

    for similar_idx, similarity in zip(
        indices[0][1:],
        similarities[1:]
    ):

        similar_movie_id = movie_ids[similar_idx]

        if similar_movie_id in set(user_ratings["movieId"]):
            continue

        if similarity <= 0:
            continue

        rating = rated_movie["rating"]

        if similar_movie_id not in item_scores:
            item_scores[similar_movie_id] = {
                "weighted_sum": 0,
                "similarity_sum": 0
            }

        item_scores[similar_movie_id]["weighted_sum"] += (
            similarity * rating
        )

        item_scores[similar_movie_id]["similarity_sum"] += similarity


item_cf_scores = {}

for movie_id, values in item_scores.items():

    if values["similarity_sum"] == 0:
        continue

    score = (
        values["weighted_sum"]
        / values["similarity_sum"]
    )

    item_cf_scores[movie_id] = score

print(f"Item-Based CF candidates: {len(item_cf_scores):,}")


# ============================================================
# STEP 6: TRAIN SVD
# ============================================================

print("\nTraining SVD model...")

sample_size = min(1_000_000, len(ratings))

svd_data = ratings.sample(
    n=sample_size,
    random_state=42
)

reader = Reader(
    rating_scale=(0.5, 5.0)
)

dataset = Dataset.load_from_df(
    svd_data[["userId", "movieId", "rating"]],
    reader
)

trainset = dataset.build_full_trainset()

svd_model = SVD(
    n_factors=100,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)

svd_model.fit(trainset)

print("SVD training completed.")


# ============================================================
# STEP 7: GENERATE SVD PREDICTIONS
# ============================================================

print("\nGenerating SVD predictions...")

rated_movie_ids = set(user_ratings["movieId"])

candidate_movies = [
    movie_id
    for movie_id in movie_ids
    if movie_id not in rated_movie_ids
]

svd_scores = {}

for movie_id in candidate_movies:

    prediction = svd_model.predict(
        target_user,
        movie_id
    ).est

    svd_scores[movie_id] = np.clip(
        prediction,
        0.5,
        5.0
    )

print(f"SVD candidates: {len(svd_scores):,}")


# ============================================================
# STEP 8: NORMALIZATION FUNCTION
# ============================================================

def normalize_scores(score_dict):

    if not score_dict:
        return {}

    values = np.array(
        list(score_dict.values()),
        dtype=float
    )

    min_value = values.min()
    max_value = values.max()

    if max_value == min_value:
        return {
            movie_id: 0.5
            for movie_id in score_dict
        }

    return {
        movie_id:
        (score - min_value) /
        (max_value - min_value)

        for movie_id, score
        in score_dict.items()
    }


# Normalize all three models

normalized_user_scores = normalize_scores(
    user_cf_scores
)

normalized_item_scores = normalize_scores(
    item_cf_scores
)

normalized_svd_scores = normalize_scores(
    svd_scores
)


# ============================================================
# STEP 9: HYBRID SCORING
# ============================================================

print("\nCombining recommendation models...")

hybrid_scores = {}

# Model weights
USER_WEIGHT = 0.30
ITEM_WEIGHT = 0.30
SVD_WEIGHT = 0.40


for movie_id in candidate_movies:

    user_score = normalized_user_scores.get(
        movie_id,
        0
    )

    item_score = normalized_item_scores.get(
        movie_id,
        0
    )

    svd_score = normalized_svd_scores.get(
        movie_id,
        0
    )

    hybrid_score = (
        USER_WEIGHT * user_score
        + ITEM_WEIGHT * item_score
        + SVD_WEIGHT * svd_score
    )

    hybrid_scores[movie_id] = hybrid_score


# ============================================================
# STEP 10: TOP 10 RECOMMENDATIONS
# ============================================================

top_movies = sorted(
    hybrid_scores.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]


movie_lookup = movies.set_index("movieId")


print("\n")
print("=" * 65)
print("             FINAL HYBRID RECOMMENDATIONS")
print("=" * 65)

for rank, (movie_id, score) in enumerate(
    top_movies,
    start=1
):

    title = movie_lookup.loc[
        movie_id,
        "title"
    ]

    genres = movie_lookup.loc[
        movie_id,
        "genres"
    ]

    print(
        f"{rank:2}. {title}"
    )

    print(
        f"    Genres: {genres}"
    )

    print(
        f"    Hybrid Score: {score:.4f}"
    )

    print()


# ============================================================
# STEP 11: SAVE RESULTS
# ============================================================

recommendations = []

for rank, (movie_id, score) in enumerate(
    top_movies,
    start=1
):

    movie_info = movie_lookup.loc[movie_id]

    recommendations.append({
        "rank": rank,
        "movieId": movie_id,
        "title": movie_info["title"],
        "genres": movie_info["genres"],
        "hybrid_score": round(score, 4)
    })


recommendations_df = pd.DataFrame(
    recommendations
)

recommendations_df.to_csv(
    "data/hybrid_recommendations_user10.csv",
    index=False
)

print("=" * 65)
print("Recommendations saved successfully!")
print(
    "File: data/hybrid_recommendations_user10.csv"
)
print("=" * 65)