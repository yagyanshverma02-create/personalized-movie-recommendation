import pandas as pd

# Load datasets
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# Basic information
print("===== MOVIE DATASET =====")
print(f"Number of movies: {len(movies)}")
print(f"Number of columns: {len(movies.columns)}")
print("\nColumns:")
print(movies.columns.tolist())

print("\nFirst 5 movies:")
print(movies.head())

print("\n===== RATINGS DATASET =====")
print(f"Number of ratings: {len(ratings)}")
print(f"Number of users: {ratings['userId'].nunique()}")
print(f"Number of movies rated: {ratings['movieId'].nunique()}")

print("\nColumns:")
print(ratings.columns.tolist())

print("\nFirst 5 ratings:")
print(ratings.head())

print("\n===== RATING STATISTICS =====")
print(ratings["rating"].describe())

print("\n===== MISSING VALUES =====")
print("Movies:")
print(movies.isnull().sum())

print("\nRatings:")
print(ratings.isnull().sum())