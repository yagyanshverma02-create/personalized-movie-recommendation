import unittest

from intent_match import get_intent_match_details


MOODS = {"Funny": {"Comedy"}, "Suspenseful": {"Thriller", "Mystery", "Crime"}}


class IntentMatchTests(unittest.TestCase):
    def test_single_genre_uses_movie_catalog_overlap(self):
        exact = get_intent_match_details("Thriller", ["Thriller"])
        mixed = get_intent_match_details("Thriller|Drama", ["Thriller"])
        self.assertGreater(exact[0], mixed[0])
        self.assertIn("Thriller intent", exact[1])

    def test_multiple_genres_reward_full_overlap_and_penalize_partial(self):
        intent = ["Comedy", "Romance"]
        full = get_intent_match_details("Comedy|Romance", intent)
        comedy_only = get_intent_match_details("Comedy", intent)
        romance_only = get_intent_match_details("Romance", intent)
        neither = get_intent_match_details("Drama", intent)
        self.assertGreater(full[0], comedy_only[0])
        self.assertEqual(comedy_only[0], romance_only[0])
        self.assertGreater(comedy_only[0], neither[0])

    def test_no_matching_genre_scores_zero(self):
        score, explanation = get_intent_match_details("Drama|War", ["Comedy"])
        self.assertEqual(score, 0)
        self.assertIn("No requested genre", explanation)

    def test_genre_and_verified_mood_are_both_used(self):
        score, explanation = get_intent_match_details(
            "Comedy|Romance", ["Comedy", "Romance"], "Funny", MOODS
        )
        genre_only, _ = get_intent_match_details("Comedy|Romance", ["Comedy", "Romance"])
        comedy_only, _ = get_intent_match_details("Comedy", ["Comedy", "Romance"], "Funny", MOODS)
        self.assertIn("Funny", explanation)
        self.assertLess(score, genre_only)
        self.assertGreater(score, comedy_only)
        partial_mood, _ = get_intent_match_details(
            "Romance", ["Comedy", "Romance"], "Funny", MOODS
        )
        self.assertGreater(score, partial_mood)

    def test_reference_similarity_contributes_only_when_available(self):
        score, explanation = get_intent_match_details(
            "", reference_similarity=0.72, reference_title="Inception"
        )
        self.assertEqual(score, 72)
        self.assertIn("Inception", explanation)

    def test_no_active_intent_has_no_score(self):
        self.assertEqual(get_intent_match_details("Comedy|Romance"), (None, None))


if __name__ == "__main__":
    unittest.main()
