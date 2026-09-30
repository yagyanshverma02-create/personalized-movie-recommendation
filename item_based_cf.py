import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

# ============================================================
# STEP 11: ITEM-BASED COLLABORATIVE FILTERING
# ============================================================

print("Loading cleaned ratings...")

ratings = pd.read_csv("data/ratings_clean.csv")
movies = pd.read_csv("data/movies_clean.csv")

print(f"Ratings: {len(ratings):,}")
print(f"Users: {ratings['userId'].nunique():,}")
print(f"Movies: {ratings['movieId'].nunique():,}")


# ------------------------------------------------------------
# 1. CREATE USER-MOVIE MATRIX
# ------------------------------------------------------------

print("\nCreating sparse user-movie matrix...")

user_ids = ratings["userId"].unique()
movie_ids = ratings["movieId"].unique()

user_to_index = {
    user_id: i
    for i, user_id in enumerate(user_ids)
}

movie_to_index = {
    movie_id: i
    for i, movie_id in enumerate(movie_ids)
}

rows = ratings["userId"].map(user_to_index).to_numpy()
cols = ratings["movieId"].map(movie_to_index).to_numpy()
values = ratings["rating"].astype(np.float32).to_numpy()

rating_matrix = csr_matrix(
    (values, (rows, cols)),
    shape=(len(user_ids), len(movie_ids))
)

print(f"Rating matrix: {rating_matrix.shape}")
print(f"Stored ratings: {rating_matrix.nnz:,}")


# ------------------------------------------------------------
# 2. CREATE ITEM-MOVIE MATRIX
# ------------------------------------------------------------

# Transpose:
# User × Movie  →  Movie × User

movie_user_matrix = rating_matrix.T.tocsr()

print(
    f"Movie-user matrix: "
    f"{movie_user_matrix.shape}"
)


# ------------------------------------------------------------
# 3. NORMALIZE MOVIE VECTORS
# ------------------------------------------------------------

print("\nNormalizing movie vectors...")

movie_user_normalized = normalize(
    movie_user_matrix,
    norm="l2",
    axis=1
)

print("Movie vectors normalized successfully.")


# ------------------------------------------------------------
# 4. BUILD ITEM-BASED MODEL
# ------------------------------------------------------------

print("\nTraining Item-Based Collaborative Filtering...")

model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=21,
    n_jobs=-1
)

model.fit(movie_user_normalized)

print("Model trained successfully!")


# ------------------------------------------------------------
# 5. SELECT TARGET USER
# ------------------------------------------------------------

target_user = 10

if target_user not in user_to_index:
    raise ValueError(
        f"User {target_user} does not exist."
    )

target_index = user_to_index[target_user]

print(f"\nTarget user: {target_user}")


# ------------------------------------------------------------
# 6. GET TARGET USER'S RATINGS
# ------------------------------------------------------------

target_ratings = rating_matrix[target_index]

rated_movie_indices = target_ratings.indices
rated_movie_ratings = target_ratings.data

print(
    f"Movies already rated by user: "
    f"{len(rated_movie_indices)}"
)


# ------------------------------------------------------------
# 7. FIND SIMILAR MOVIES
# ------------------------------------------------------------

print("\nFinding similar movies...")


# Dictionary:
# movie_index -> accumulated recommendation score

recommendation_scores = {}
similarity_totals = {}

# Limit the number of movies from the user's history
# to the highest-rated movies for efficiency.

movie_rating_pairs = list(
    zip(
        rated_movie_indices,
        rated_movie_ratings
    )
)

movie_rating_pairs.sort(
    key=lambda x: x[1],
    reverse=True
)

# Use the user's top-rated movies as recommendation signals.
top_rated_history = movie_rating_pairs[:50]

print(
    f"Using top "
    f"{len(top_rated_history)} "
    f"rated movies as recommendation signals."
)


for movie_index, user_rating in top_rated_history:

    distances, indices = model.kneighbors(
        movie_user_normalized[movie_index],
        n_neighbors=21
    )

    for distance, similar_movie_index in zip(
        distances[0],
        indices[0]
    ):

        # Skip the movie itself
        if similar_movie_index == movie_index:
            continue

        similarity = 1 - distance

        if similarity <= 0:
            continue

        # Skip movies already rated by the target user
        if rating_matrix[
            target_index,
            similar_movie_index
        ] > 0:
            continue

        # Weight:
        # similarity × user's rating
        score = similarity * user_rating

        recommendation_scores[
            similar_movie_index
        ] = (
            recommendation_scores.get(
                similar_movie_index,
                0.0
            ) + score
        )

        similarity_totals[
            similar_movie_index
        ] = (
            similarity_totals.get(
                similar_movie_index,
                0.0
            ) + similarity
        )


# ------------------------------------------------------------
# 8. CALCULATE FINAL SCORES
# ------------------------------------------------------------

recommendations = []

for movie_index in recommendation_scores:

    total_similarity = similarity_totals[
        movie_index
    ]

    if total_similarity <= 0:
        continue

    predicted_score = (
        recommendation_scores[movie_index]
        / total_similarity
    )

    recommendations.append(
        (
            movie_index,
            predicted_score,
            total_similarity
        )
    )


# ------------------------------------------------------------
# 9. SORT RECOMMENDATIONS
# ------------------------------------------------------------

recommendations.sort(
    key=lambda x: x[1],
    reverse=True
)

top_n = 10

top_recommendations = recommendations[:top_n]


# ------------------------------------------------------------
# 10. MOVIE LOOKUP
# ------------------------------------------------------------

index_to_movie = {
    index: movie_id
    for movie_id, index in movie_to_index.items()
}

movie_lookup = movies.set_index(
    "movieId"
)


# ------------------------------------------------------------
# 11. DISPLAY RESULTS
# ------------------------------------------------------------

print("\n==============================================")
print("       TOP 10 ITEM-BASED RECOMMENDATIONS")
print("==============================================")

for i, (
    movie_index,
    score,
    support
) in enumerate(
    top_recommendations,
    start=1
):

    movie_id = index_to_movie[
        movie_index
    ]

    if movie_id not in movie_lookup.index:
        continue

    movie_info = movie_lookup.loc[
        movie_id
    ]

    print(
        f"{i}. {movie_info['title']} | "
        f"Score: {score:.3f} | "
        f"Genres: {movie_info['genres']} | "
        f"Similarity Support: {support:.3f}"
    )


print("\n==============================================")
print("Item-Based Collaborative Filtering completed!")
print("==============================================")