import unittest

from movie_details import normalize_movie_for_details, select_movie_details


class SharedMovieDetailsTests(unittest.TestCase):
    def test_recommendation_movie_keeps_catalog_and_recommendation_data(self):
        movie = {
            "movieId": 42,
            "tmdbId": 550,
            "imdbId": 1234567,
            "title": "Example (1999)",
            "genres": "Comedy|Romance",
            "hybrid_score": 0.8,
            "user_score": 0.7,
            "item_score": 0.8,
            "svd_score": 0.9,
        }
        normalized = normalize_movie_for_details(movie)
        self.assertEqual(normalized["movieId"], 42)
        self.assertEqual(normalized["tmdbId"], 550)
        self.assertEqual(normalized["title"], "Example (1999)")
        self.assertEqual(normalized["_recommendation"], movie)

    def test_local_search_movie_selects_the_same_details_state(self):
        state = {}
        source = {"movieId": "42", "tmdbId": "550", "title": "Example (1999)", "genres": "Comedy"}
        selected = select_movie_details(state, source)
        self.assertIs(state["selected_movie_details"], selected)
        self.assertEqual(selected["movieId"], 42)
        self.assertEqual(selected["tmdbId"], 550)
        self.assertEqual(selected["title"], "Example (1999)")

    def test_watch_later_local_movie_resolves_catalog_identifier(self):
        saved = {"movieId": 42, "tmdbId": 550, "title": "Example (1999)", "genres": "Comedy|Romance"}
        normalized = normalize_movie_for_details(saved)
        self.assertEqual(normalized["movieId"], 42)
        self.assertEqual(normalized["tmdbId"], 550)
        self.assertFalse(normalized["_external_tmdb"])

    def test_watch_later_discover_movie_keeps_tmdb_identity_and_images(self):
        saved = {
            "movieId": -550,
            "tmdbId": 550,
            "title": "Example",
            "genres": "Comedy|Romance",
            "release_date": "2024-01-02",
            "poster_path": "/poster.jpg",
            "backdrop_path": "/backdrop.jpg",
            "_external_tmdb": True,
        }
        normalized = normalize_movie_for_details(saved)
        self.assertEqual(normalized["movieId"], -550)
        self.assertEqual(normalized["tmdbId"], 550)
        self.assertTrue(normalized["_external_tmdb"])
        self.assertEqual(normalized["poster_path"], "/poster.jpg")
        self.assertEqual(normalized["backdrop_path"], "/backdrop.jpg")

    def test_discover_movie_without_local_id_still_resolves_by_tmdb_id(self):
        normalized = normalize_movie_for_details({"tmdbId": 550, "title": "Example", "_external_tmdb": True})
        self.assertIsNone(normalized["movieId"])
        self.assertEqual(normalized["tmdbId"], 550)
        self.assertTrue(normalized["_external_tmdb"])


if __name__ == "__main__":
    unittest.main()
