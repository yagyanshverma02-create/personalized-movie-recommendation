import pandas as pd

ratings = pd.read_csv("data/ratings.csv")

print("===== USER ACTIVITY =====")

user_counts = ratings["userId"].value_counts()

print("Average ratings per user:", user_counts.mean())
print("Median ratings per user:", user_counts.median())
print("Maximum ratings by one user:", user_counts.max())

print("\nUsers with at least 20 ratings:",
      (user_counts >= 20).sum())

print("\nUsers with at least 50 ratings:",
      (user_counts >= 50).sum())


print("\n===== MOVIE ACTIVITY =====")

movie_counts = ratings["movieId"].value_counts()

print("Average ratings per movie:", movie_counts.mean())
print("Median ratings per movie:", movie_counts.median())
print("Maximum ratings for one movie:", movie_counts.max())

print("\nMovies with at least 20 ratings:",
      (movie_counts >= 20).sum())

print("\nMovies with at least 50 ratings:",
      (movie_counts >= 50).sum())

print("\n===== TOP 10 MOST RATED MOVIES =====")

movies = pd.read_csv("data/movies.csv")

top_movies = (
    movie_counts
    .head(10)
    .reset_index()
)

top_movies.columns = ["movieId", "rating_count"]

top_movies = top_movies.merge(
    movies[["movieId", "title"]],
    on="movieId",
    how="left"
)

print(top_movies[["title", "rating_count"]].to_string(index=False))