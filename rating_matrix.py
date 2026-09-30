import pandas as pd
from scipy.sparse import csr_matrix

print("Loading subset...")

ratings = pd.read_csv("data/ratings_subset.csv")

print(f"Ratings: {len(ratings):,}")
print(f"Users: {ratings['userId'].nunique():,}")
print(f"Movies: {ratings['movieId'].nunique():,}")


# Create ID mappings
user_ids = ratings["userId"].unique()
movie_ids = ratings["movieId"].unique()

user_to_index = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}

movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(movie_ids)
}


# Convert IDs to matrix indices
user_indices = ratings["userId"].map(user_to_index)
movie_indices = ratings["movieId"].map(movie_to_index)


# Create sparse matrix
rating_matrix = csr_matrix(
    (
        ratings["rating"].astype("float32"),
        (user_indices, movie_indices)
    ),
    shape=(len(user_ids), len(movie_ids))
)


print("\n===== RATING MATRIX =====")

print("Matrix shape:", rating_matrix.shape)
print("Stored ratings:", rating_matrix.nnz)

total_cells = (
    rating_matrix.shape[0] *
    rating_matrix.shape[1]
)

sparsity = 1 - (
    rating_matrix.nnz / total_cells
)

print(f"Possible user-movie combinations: {total_cells:,}")
print(f"Actual ratings: {rating_matrix.nnz:,}")
print(f"Sparsity: {sparsity:.2%}")


# Approximate memory used by sparse matrix
memory_mb = (
    rating_matrix.data.nbytes +
    rating_matrix.indices.nbytes +
    rating_matrix.indptr.nbytes
) / (1024 ** 2)

print(f"Sparse matrix memory: {memory_mb:.2f} MB")

print("\nSparse rating matrix created successfully!")