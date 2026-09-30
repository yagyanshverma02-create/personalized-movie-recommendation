import pandas as pd

print("Loading ratings...")
ratings = pd.read_csv("data/ratings.csv")

print(f"Original ratings: {len(ratings):,}")
print(f"Original users: {ratings['userId'].nunique():,}")
print(f"Original movies: {ratings['movieId'].nunique():,}")


# Minimum number of ratings required
MIN_USER_RATINGS = 20
MIN_MOVIE_RATINGS = 20


# --------------------------------------------------
# Step 1: Filter users
# --------------------------------------------------

user_counts = ratings["userId"].value_counts()

active_users = user_counts[
    user_counts >= MIN_USER_RATINGS
].index

ratings_filtered = ratings[
    ratings["userId"].isin(active_users)
].copy()

print(f"\nAfter user filtering:")
print(f"Ratings: {len(ratings_filtered):,}")
print(f"Users: {ratings_filtered['userId'].nunique():,}")
print(f"Movies: {ratings_filtered['movieId'].nunique():,}")


# --------------------------------------------------
# Step 2: Filter movies
# --------------------------------------------------

movie_counts = ratings_filtered["movieId"].value_counts()

popular_movies = movie_counts[
    movie_counts >= MIN_MOVIE_RATINGS
].index

ratings_filtered = ratings_filtered[
    ratings_filtered["movieId"].isin(popular_movies)
].copy()


print(f"\nAfter movie filtering:")
print(f"Ratings: {len(ratings_filtered):,}")
print(f"Users: {ratings_filtered['userId'].nunique():,}")
print(f"Movies: {ratings_filtered['movieId'].nunique():,}")


# --------------------------------------------------
# Step 3: Load movie information
# --------------------------------------------------

movies = pd.read_csv("data/movies.csv")

movies_filtered = movies[
    movies["movieId"].isin(
        ratings_filtered["movieId"].unique()
    )
].copy()


# --------------------------------------------------
# Step 4: Save filtered datasets
# --------------------------------------------------

ratings_filtered.to_csv(
    "data/ratings_filtered.csv",
    index=False
)

movies_filtered.to_csv(
    "data/movies_filtered.csv",
    index=False
)


print("\n===== FILTERING COMPLETE =====")

print("Saved:")
print("data/ratings_filtered.csv")
print("data/movies_filtered.csv")