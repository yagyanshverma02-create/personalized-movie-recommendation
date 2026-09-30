import pandas as pd
import numpy as np

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy


# ============================================================
# STEP 12: MATRIX FACTORIZATION USING SVD
# ============================================================

print("Loading cleaned ratings...")

ratings = pd.read_csv("data/ratings_clean.csv")

print(f"Total ratings: {len(ratings):,}")
print(f"Users: {ratings['userId'].nunique():,}")
print(f"Movies: {ratings['movieId'].nunique():,}")


# ------------------------------------------------------------
# 1. CREATE SVD DEVELOPMENT DATASET
# ------------------------------------------------------------

SVD_SAMPLE_SIZE = 1_000_000

if len(ratings) > SVD_SAMPLE_SIZE:

    print(
        f"\nSampling {SVD_SAMPLE_SIZE:,} ratings "
        "for SVD training..."
    )

    svd_ratings = ratings.sample(
        n=SVD_SAMPLE_SIZE,
        random_state=42
    ).copy()

else:

    svd_ratings = ratings.copy()


print(
    f"SVD dataset size: "
    f"{len(svd_ratings):,}"
)


# ------------------------------------------------------------
# 2. DEFINE RATING SCALE
# ------------------------------------------------------------

reader = Reader(
    rating_scale=(0.5, 5.0)
)


# ------------------------------------------------------------
# 3. CONVERT DATA FOR SURPRISE
# ------------------------------------------------------------

print("\nPreparing Surprise dataset...")

data = Dataset.load_from_df(
    svd_ratings[
        ["userId", "movieId", "rating"]
    ],
    reader
)


# ------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ------------------------------------------------------------

print("Creating train/test split...")

trainset, testset = train_test_split(
    data,
    test_size=0.20,
    random_state=42
)

print(
    f"Training ratings: "
    f"{trainset.n_ratings:,}"
)

print(
    f"Testing ratings: "
    f"{len(testset):,}"
)


# ------------------------------------------------------------
# 5. CREATE SVD MODEL
# ------------------------------------------------------------

print("\nCreating SVD model...")

model = SVD(
    n_factors=100,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)


# ------------------------------------------------------------
# 6. TRAIN MODEL
# ------------------------------------------------------------

print("\nTraining SVD model...")

model.fit(trainset)

print("SVD model trained successfully!")


# ------------------------------------------------------------
# 7. EVALUATE MODEL
# ------------------------------------------------------------

print("\nEvaluating model...")

predictions = model.test(testset)

rmse = accuracy.rmse(
    predictions,
    verbose=True
)

mae = accuracy.mae(
    predictions,
    verbose=True
)


print("\n==============================================")
print("             SVD MODEL RESULTS")
print("==============================================")

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"MAE:  {mae:.4f}"
)


# ------------------------------------------------------------
# 8. GENERATE RECOMMENDATIONS
# ------------------------------------------------------------

target_user = 10

print(
    f"\nGenerating recommendations "
    f"for User {target_user}..."
)

movies = pd.read_csv(
    "data/movies_clean.csv"
)

# Movies already rated by target user
rated_movies = set(
    ratings[
        ratings["userId"] == target_user
    ]["movieId"]
)

print(
    f"Movies already rated: "
    f"{len(rated_movies)}"
)


# Candidate movies
candidate_movies = movies[
    ~movies["movieId"].isin(rated_movies)
].copy()


print(
    f"Candidate movies: "
    f"{len(candidate_movies)}"
)


# ------------------------------------------------------------
# 9. PREDICT RATINGS
# ------------------------------------------------------------

predicted_ratings = []

for movie_id in candidate_movies["movieId"]:

    prediction = model.predict(
        target_user,
        movie_id
    )

    predicted_ratings.append(
        prediction.est
    )


candidate_movies[
    "Predicted Rating"
] = predicted_ratings


# ------------------------------------------------------------
# 10. SORT RECOMMENDATIONS
# ------------------------------------------------------------

top_recommendations = candidate_movies.sort_values(
    by="Predicted Rating",
    ascending=False
).head(10)


# ------------------------------------------------------------
# 11. DISPLAY RESULTS
# ------------------------------------------------------------

print("\n==============================================")
print("          TOP 10 SVD RECOMMENDATIONS")
print("==============================================")

for i, (_, movie) in enumerate(
    top_recommendations.iterrows(),
    start=1
):

    print(
        f"{i}. {movie['title']} | "
        f"Predicted Rating: "
        f"{movie['Predicted Rating']:.2f} | "
        f"Genres: {movie['genres']}"
    )


print("\n==============================================")
print("SVD Matrix Factorization completed!")
print("==============================================")