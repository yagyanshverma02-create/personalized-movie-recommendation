import pandas as pd


# ==================================================
# 1. LOAD DATA
# ==================================================

print("Loading datasets...")

ratings = pd.read_csv("data/ratings_subset.csv")
movies = pd.read_csv("data/movies_subset.csv")

print(f"Ratings loaded: {len(ratings):,}")
print(f"Movies loaded: {len(movies):,}")


# ==================================================
# 2. CHECK DATA TYPES
# ==================================================

print("\n===== DATA TYPES =====")

print(ratings.dtypes)

print("\nMovie data types:")
print(movies.dtypes)


# ==================================================
# 3. CHECK MISSING VALUES
# ==================================================

print("\n===== MISSING VALUES =====")

print("Ratings:")
print(ratings.isnull().sum())

print("\nMovies:")
print(movies.isnull().sum())


# ==================================================
# 4. CHECK DUPLICATES
# ==================================================

print("\n===== DUPLICATES =====")

duplicate_ratings = ratings.duplicated().sum()
duplicate_movies = movies.duplicated().sum()

print(f"Duplicate rating rows: {duplicate_ratings:,}")
print(f"Duplicate movie rows: {duplicate_movies:,}")


# ==================================================
# 5. CHECK DUPLICATE USER-MOVIE PAIRS
# ==================================================

duplicate_pairs = ratings.duplicated(
    subset=["userId", "movieId"]
).sum()

print(
    f"Duplicate user-movie pairs: "
    f"{duplicate_pairs:,}"
)


# ==================================================
# 6. CHECK RATING VALUES
# ==================================================

print("\n===== RATING VALIDATION =====")

print(
    "Minimum rating:",
    ratings["rating"].min()
)

print(
    "Maximum rating:",
    ratings["rating"].max()
)

print(
    "Unique ratings:",
    sorted(ratings["rating"].unique())
)

invalid_ratings = ratings[
    (ratings["rating"] < 0.5) |
    (ratings["rating"] > 5.0)
]

print(
    f"Invalid ratings: "
    f"{len(invalid_ratings):,}"
)


# ==================================================
# 7. CHECK USER IDs
# ==================================================

print("\n===== USER ID VALIDATION =====")

invalid_users = ratings[
    ratings["userId"] <= 0
]

print(
    f"Invalid user IDs: "
    f"{len(invalid_users):,}"
)


# ==================================================
# 8. CHECK MOVIE IDs
# ==================================================

print("\n===== MOVIE ID VALIDATION =====")

invalid_movies = ratings[
    ratings["movieId"] <= 0
]

print(
    f"Invalid movie IDs: "
    f"{len(invalid_movies):,}"
)


# ==================================================
# 9. CHECK MOVIE REFERENCES
# ==================================================

print("\n===== MOVIE REFERENCE CHECK =====")

rating_movie_ids = set(
    ratings["movieId"].unique()
)

movie_ids = set(
    movies["movieId"].unique()
)

missing_movie_metadata = (
    rating_movie_ids - movie_ids
)

print(
    "Movies with missing metadata:",
    len(missing_movie_metadata)
)


# ==================================================
# 10. CHECK TIMESTAMP
# ==================================================

print("\n===== TIMESTAMP CHECK =====")

print(
    "Minimum timestamp:",
    ratings["timestamp"].min()
)

print(
    "Maximum timestamp:",
    ratings["timestamp"].max()
)

invalid_timestamps = ratings[
    ratings["timestamp"] <= 0
]

print(
    f"Invalid timestamps: "
    f"{len(invalid_timestamps):,}"
)


# ==================================================
# 11. CLEAN DATA
# ==================================================

print("\n===== CLEANING DATA =====")

# Remove exact duplicate rows
ratings = ratings.drop_duplicates()

# Remove invalid ratings
ratings = ratings[
    ratings["rating"].between(0.5, 5.0)
]

# Remove invalid IDs
ratings = ratings[
    (ratings["userId"] > 0) &
    (ratings["movieId"] > 0)
]

# Remove invalid timestamps
ratings = ratings[
    ratings["timestamp"] > 0
]

# Keep only movies that exist in movie metadata
ratings = ratings[
    ratings["movieId"].isin(movies["movieId"])
]


# ==================================================
# 12. HANDLE DUPLICATE USER-MOVIE RATINGS
# ==================================================

# If the same user has rated the same movie
# multiple times, keep the most recent rating.

ratings = ratings.sort_values(
    "timestamp"
)

ratings = ratings.drop_duplicates(
    subset=["userId", "movieId"],
    keep="last"
)


# ==================================================
# 13. CLEAN MOVIE DATA
# ==================================================

movies = movies.drop_duplicates(
    subset=["movieId"]
)

movies = movies.dropna(
    subset=["movieId", "title"]
)


# ==================================================
# 14. OPTIMIZE DATA TYPES
# ==================================================

ratings["userId"] = ratings["userId"].astype("int32")
ratings["movieId"] = ratings["movieId"].astype("int32")
ratings["rating"] = ratings["rating"].astype("float32")
ratings["timestamp"] = ratings["timestamp"].astype("int64")

movies["movieId"] = movies["movieId"].astype("int32")


# ==================================================
# 15. FINAL VALIDATION
# ==================================================

print("\n===== FINAL CLEAN DATA =====")

print(
    f"Ratings: {len(ratings):,}"
)

print(
    f"Users: {ratings['userId'].nunique():,}"
)

print(
    f"Movies: {ratings['movieId'].nunique():,}"
)

print(
    f"Movie metadata rows: {len(movies):,}"
)

print(
    f"Missing rating values: "
    f"{ratings.isnull().sum().sum()}"
)

print(
    f"Duplicate user-movie pairs: "
    f"{ratings.duplicated(['userId', 'movieId']).sum():,}"
)


# ==================================================
# 16. SAVE CLEAN DATA
# ==================================================

ratings.to_csv(
    "data/ratings_clean.csv",
    index=False
)

movies.to_csv(
    "data/movies_clean.csv",
    index=False
)

print("\n===== SAVED =====")

print("data/ratings_clean.csv")
print("data/movies_clean.csv")