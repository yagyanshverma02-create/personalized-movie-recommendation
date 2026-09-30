import pandas as pd

INPUT_FILE = "data/ratings_filtered.csv"

MAX_USERS = 10_000
MAX_MOVIES = 5_000

print("Loading ratings...")
ratings = pd.read_csv(INPUT_FILE)

print(f"Original ratings: {len(ratings):,}")
print(f"Original users: {ratings['userId'].nunique():,}")
print(f"Original movies: {ratings['movieId'].nunique():,}")


# --------------------------------------------------
# 1. Select the most active users
# --------------------------------------------------

user_counts = ratings["userId"].value_counts()

selected_users = user_counts.head(MAX_USERS).index

subset = ratings[
    ratings["userId"].isin(selected_users)
].copy()

print("\nAfter selecting users:")
print(f"Users: {subset['userId'].nunique():,}")
print(f"Ratings: {len(subset):,}")


# --------------------------------------------------
# 2. Select the most rated movies among those users
# --------------------------------------------------

movie_counts = subset["movieId"].value_counts()

selected_movies = movie_counts.head(MAX_MOVIES).index

subset = subset[
    subset["movieId"].isin(selected_movies)
].copy()


print("\n===== FINAL SUBSET =====")
print(f"Users: {subset['userId'].nunique():,}")
print(f"Movies: {subset['movieId'].nunique():,}")
print(f"Ratings: {len(subset):,}")


# --------------------------------------------------
# 3. Save subset
# --------------------------------------------------

subset.to_csv(
    "data/ratings_subset.csv",
    index=False
)

print("\nSaved:")
print("data/ratings_subset.csv")


# --------------------------------------------------
# 4. Save corresponding movie information
# --------------------------------------------------

movies = pd.read_csv("data/movies.csv")

movies_subset = movies[
    movies["movieId"].isin(
        subset["movieId"].unique()
    )
].copy()

movies_subset.to_csv(
    "data/movies_subset.csv",
    index=False
)

print("data/movies_subset.csv")