import json
import tempfile
import unittest
from pathlib import Path

from watch_later import toggle_watch_later_entry, watch_later_key


class WatchLaterTests(unittest.TestCase):
    def test_add_deduplicates_and_remove_toggles(self):
        entries, saved = toggle_watch_later_entry([], {"movieId": 42})
        self.assertTrue(saved)
        self.assertEqual(entries, [{"movieId": 42}])

        entries, saved = toggle_watch_later_entry(entries, {"movieId": 42})
        self.assertFalse(saved)
        self.assertEqual(entries, [])

    def test_legacy_duplicates_are_removed_together(self):
        entries, saved = toggle_watch_later_entry(
            [{"movieId": 42}, {"movieId": 42}, {"movieId": 7}],
            {"movieId": 42},
        )
        self.assertFalse(saved)
        self.assertEqual(entries, [{"movieId": 7}])

    def test_external_tmdb_movies_have_stable_keys(self):
        movie = {
            "_external_tmdb": True,
            "tmdbId": 9001,
            "title": "Example",
            "genres": "Drama",
            "poster_path": "/poster.jpg",
        }
        entries, saved = toggle_watch_later_entry([], movie)
        self.assertTrue(saved)
        self.assertEqual(watch_later_key(entries[0]), "tmdb:9001")
        self.assertTrue(entries[0]["external_tmdb"])

    def test_profile_lists_persist_independently_in_existing_json_shape(self):
        profiles = [
            {"profile_id": "one", "name": "One", "movies": [], "watch_later": []},
            {"profile_id": "two", "name": "Two", "movies": [], "watch_later": []},
        ]
        profiles[0]["watch_later"], _ = toggle_watch_later_entry(
            profiles[0]["watch_later"], {"movieId": 42}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "user_profiles.json"
            path.write_text(json.dumps(profiles), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded[0]["watch_later"], [{"movieId": 42}])
        self.assertEqual(loaded[1]["watch_later"], [])


if __name__ == "__main__":
    unittest.main()
