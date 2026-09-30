import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# ============================================================
# STEP 10: IMPROVED USER-BASED COLLABORATIVE FILTERING
# ============================================================

print("Loading cleaned ratings...")

ratings = pd.read_csv("data/ratings_clean.csv")
movies = pd.read_csv("data/movies_clean.csv")

print(f"Ratings: {len(ratings):,}")
print(f"Users: {ratings['userId'].nunique():,}")
print(f"Movies: {ratings['movieId'].nunique():,}")


# ------------------------------------------------------------
# 1. CREATE USER-MOVIE SPARSE MATRIX
# ------------------------------------------------------------

print("\nCreating sparse user-movie matrix...")

user_ids = ratings["userId"].unique()
movie_ids = ratings["movieId"].unique()

user_to_index = {
    user_id: i for i, user_id in enumerate(user_ids)
}

movie_to_index = {
    movie_id: i for i, movie_id in enumerate(movie_ids)
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
# 2. CALCULATE USER AVERAGE RATINGS
# ------------------------------------------------------------

print("\nCalculating user rating averages...")

user_rating_sum = np.asarray(
    rating_matrix.sum(axis=1)
).flatten()

user_rating_count = np.diff(
    rating_matrix.indptr
)

user_means = np.divide(
    user_rating_sum,
    user_rating_count,
    out=np.zeros_like(user_rating_sum, dtype=np.float32),
    where=user_rating_count != 0
)


# ------------------------------------------------------------
# 3. SELECT TARGET USER
# ------------------------------------------------------------

target_user = 10

if target_user not in user_to_index:
    raise ValueError(
        f"User {target_user} does not exist."
    )

target_index = user_to_index[target_user]
target_mean = user_means[target_index]

print(f"\nTarget user: {target_user}")
print(f"Target user's average rating: {target_mean:.2f}")


# ------------------------------------------------------------
# 4. FIND SIMILAR USERS
# ------------------------------------------------------------

print("\nFinding similar users...")

model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=21,
    n_jobs=-1
)

model.fit(rating_matrix)

distances, indices = model.kneighbors(
    rating_matrix[target_index],
    n_neighbors=21
)


print("\n===== TOP SIMILAR USERS =====")

similar_users = []

for distance, index in zip(
    distances[0],
    indices[0]
):

    # Skip target user
    if index == target_index:
        continue

    similarity = 1 - distance

    if similarity <= 0:
        continue

    user_id = user_ids[index]

    similar_users.append(
        (user_id, index, similarity)
    )

    print(
        f"User {user_id} | "
        f"Similarity: {similarity:.4f}"
    )


# ------------------------------------------------------------
# 5. CALCULATE BASELINE-CENTERED PREDICTIONS
# ------------------------------------------------------------

print("\nCalculating personalized predictions...")

prediction_sum = {}
similarity_sum = {}

for user_id, user_index, similarity in similar_users:

    user_mean = user_means[user_index]

    user_ratings = rating_matrix[user_index]

    movie_indices = user_ratings.indices
    ratings_values = user_ratings.data

    for movie_index, rating in zip(
        movie_indices,
        ratings_values
    ):

        # Skip movies already rated by target user
        if rating_matrix[target_index, movie_index] > 0:
            continue

        # Rating deviation from this user's normal rating
        deviation = rating - user_mean

        weighted_deviation = (
            similarity * deviation
        )

        prediction_sum[movie_index] = (
            prediction_sum.get(movie_index, 0.0)
            + weighted_deviation
        )

        similarity_sum[movie_index] = (
            similarity_sum.get(movie_index, 0.0)
            + similarity
        )


# ------------------------------------------------------------
# 6. GENERATE FINAL PREDICTIONS
# ------------------------------------------------------------

predictions = []

for movie_index in prediction_sum:

    sim_total = similarity_sum[movie_index]

    if sim_total <= 0:
        continue

    predicted_rating = (
        target_mean
        + prediction_sum[movie_index] / sim_total
    )

    # Keep within MovieLens rating range.
    predicted_rating = np.clip(
        predicted_rating,
        0.5,
        5.0
    )

    predictions.append(
        (movie_index, float(predicted_rating), sim_total)
    )


# ------------------------------------------------------------
# 7. REQUIRE ENOUGH SUPPORT
# ------------------------------------------------------------

# Movies supported by more similar-user ratings
# are generally more reliable recommendations.

MIN_SUPPORT = 3

predictions = [
    prediction
    for prediction in predictions
    if prediction[2] >= MIN_SUPPORT
]


# ------------------------------------------------------------
# 8. SORT PREDICTIONS
# ------------------------------------------------------------

predictions.sort(
    key=lambda x: x[1],
    reverse=True
)

top_n = 10
top_predictions = predictions[:top_n]


# ------------------------------------------------------------
# 9. CONVERT MOVIE INDEX TO MOVIE INFORMATION
# ------------------------------------------------------------

index_to_movie = {
    index: movie_id
    for movie_id, index in movie_to_index.items()
}

movie_lookup = movies.set_index("movieId")


recommendations = []

for movie_index, predicted_rating, support in top_predictions:

    movie_id = index_to_movie[movie_index]

    if movie_id not in movie_lookup.index:
        continue

    movie_info = movie_lookup.loc[movie_id]

    recommendations.append({
        "Movie": movie_info["title"],
        "Predicted Rating": round(
            predicted_rating,
            2
        ),
        "Genres": movie_info["genres"],
        "Similar User Support": round(
            support,
            2
        )
    })


# ------------------------------------------------------------
# 10. DISPLAY RESULTS
# ------------------------------------------------------------

print("\n==============================================")
print("       TOP 10 PERSONALIZED RECOMMENDATIONS")
print("==============================================")

for i, recommendation in enumerate(
    recommendations,
    start=1
):

    print(
        f"{i}. {recommendation['Movie']} | "
        f"Predicted Rating: "
        f"{recommendation['Predicted Rating']:.2f} | "
        f"Genres: {recommendation['Genres']} | "
        f"Support: "
        f"{recommendation['Similar User Support']:.2f}"
    )


print("\n==============================================")
print("User-Based Collaborative Filtering completed!")
print("==============================================")