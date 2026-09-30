import pandas as pd
import numpy as np

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy


# ============================================================
# STEP 13: MODEL EVALUATION
# ============================================================

print("Loading cleaned ratings...")

ratings = pd.read_csv("data/ratings_clean.csv")
movies = pd.read_csv("data/movies_clean.csv")

print(f"Total ratings: {len(ratings):,}")
print(f"Users: {ratings['userId'].nunique():,}")
print(f"Movies: {ratings['movieId'].nunique():,}")


# ------------------------------------------------------------
# 1. CREATE REPRODUCIBLE EVALUATION DATASET
# ------------------------------------------------------------

SAMPLE_SIZE = 1_000_000

print(
    f"\nCreating evaluation sample "
    f"of {SAMPLE_SIZE:,} ratings..."
)

evaluation_data = ratings.sample(
    n=SAMPLE_SIZE,
    random_state=42
).copy()

print(
    f"Evaluation ratings: "
    f"{len(evaluation_data):,}"
)


# ------------------------------------------------------------
# 2. CREATE SURPRISE DATASET
# ------------------------------------------------------------

reader = Reader(
    rating_scale=(0.5, 5.0)
)

data = Dataset.load_from_df(
    evaluation_data[
        ["userId", "movieId", "rating"]
    ],
    reader
)


# ------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ------------------------------------------------------------

print("\nCreating 80/20 train-test split...")

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
# 4. TRAIN SVD
# ------------------------------------------------------------

print("\nTraining SVD model...")

svd = SVD(
    n_factors=100,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)

svd.fit(trainset)

print("SVD training completed.")


# ------------------------------------------------------------
# 5. RATING PREDICTION EVALUATION
# ------------------------------------------------------------

print("\nEvaluating rating predictions...")

predictions = svd.test(testset)

rmse = accuracy.rmse(
    predictions,
    verbose=False
)

mae = accuracy.mae(
    predictions,
    verbose=False
)

print("\n==============================================")
print("          RATING PREDICTION RESULTS")
print("==============================================")

print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")


# ------------------------------------------------------------
# 6. CREATE TEST DATAFRAME
# ------------------------------------------------------------

test_rows = []

for prediction in predictions:

    test_rows.append({
        "userId": prediction.uid,
        "movieId": prediction.iid,
        "actual": prediction.r_ui,
        "predicted": prediction.est
    })

test_df = pd.DataFrame(test_rows)


# ------------------------------------------------------------
# 7. DEFINE RELEVANT MOVIES
# ------------------------------------------------------------

# A movie is considered relevant if the user actually
# rated it 4.0 or higher in the held-out test set.

RELEVANCE_THRESHOLD = 4.0

relevant_test = test_df[
    test_df["actual"] >= RELEVANCE_THRESHOLD
].copy()

print(
    f"\nRelevant test ratings "
    f"(rating >= {RELEVANCE_THRESHOLD}): "
    f"{len(relevant_test):,}"
)


# ------------------------------------------------------------
# 8. TOP-N RECOMMENDATION EVALUATION
# ------------------------------------------------------------

TOP_N = 10

print(
    f"\nCalculating Precision@{TOP_N} "
    f"and Recall@{TOP_N}..."
)

precision_values = []
recall_values = []

users_evaluated = 0


for user_id, user_data in test_df.groupby("userId"):

    # Actual relevant movies for this user
    relevant_movies = set(
        user_data[
            user_data["actual"] >= RELEVANCE_THRESHOLD
        ]["movieId"]
    )

    # Skip users without relevant test movies
    if len(relevant_movies) == 0:
        continue

    # Top-N predicted movies
    top_predictions = (
        user_data
        .sort_values(
            by="predicted",
            ascending=False
        )
        .head(TOP_N)
    )

    recommended_movies = set(
        top_predictions["movieId"]
    )

    hits = len(
        relevant_movies.intersection(
            recommended_movies
        )
    )

    precision = hits / TOP_N

    recall = hits / len(relevant_movies)

    precision_values.append(precision)
    recall_values.append(recall)

    users_evaluated += 1


# ------------------------------------------------------------
# 9. FINAL TOP-N METRICS
# ------------------------------------------------------------

if users_evaluated > 0:

    precision_at_10 = np.mean(
        precision_values
    )

    recall_at_10 = np.mean(
        recall_values
    )

else:

    precision_at_10 = 0
    recall_at_10 = 0


# ------------------------------------------------------------
# 10. DISPLAY FINAL RESULTS
# ------------------------------------------------------------

print("\n==============================================")
print("          TOP-N RECOMMENDATION RESULTS")
print("==============================================")

print(
    f"Users evaluated: "
    f"{users_evaluated:,}"
)

print(
    f"Precision@10: "
    f"{precision_at_10:.4f}"
)

print(
    f"Recall@10: "
    f"{recall_at_10:.4f}"
)


# ------------------------------------------------------------
# 11. SUMMARY
# ------------------------------------------------------------

print("\n==============================================")
print("             STEP 13 SUMMARY")
print("==============================================")

print(f"RMSE:         {rmse:.4f}")
print(f"MAE:          {mae:.4f}")
print(f"Precision@10: {precision_at_10:.4f}")
print(f"Recall@10:    {recall_at_10:.4f}")

print("\nModel evaluation completed successfully!")