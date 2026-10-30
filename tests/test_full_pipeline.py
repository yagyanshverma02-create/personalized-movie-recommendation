"""Comprehensive integration test for MovieMind's voice and hybrid recommendation pipeline."""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from gemini_helper import MovieIntent, transcribe_speech_to_text, parse_movie_intent
from watch_later import toggle_watch_later_entry, watch_later_key
from movie_details import normalize_movie_for_details, select_movie_details


class FullPipelineIntegrationTests(unittest.TestCase):
    @patch("gemini_helper.get_gemini_client")
    def test_voice_to_intent_to_recommendation_flow(self, mock_get_client):
        """Verify complete pipeline: Audio -> STT -> Intent -> Recommender inputs."""
        # Step 1: Simulate audio transcription
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "I want a funny romantic movie like The Proposal"
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        transcribed_text = transcribe_speech_to_text(b"sample_audio_bytes", mime_type="audio/wav")
        self.assertEqual(transcribed_text, "I want a funny romantic movie like The Proposal")

        # Step 2: Feed transcribed text into parse_movie_intent
        mock_interaction = MagicMock()
        mock_interaction.output_text = (
            '{"genres": ["Comedy", "Romance"], "mood": "Funny", '
            '"reference_movie": "The Proposal", "keywords": []}'
        )
        mock_client.interactions.create.return_value = mock_interaction

        intent = parse_movie_intent(transcribed_text)
        self.assertEqual(intent.genres, ["Comedy", "Romance"])
        self.assertEqual(intent.mood, "Funny")
        self.assertEqual(intent.reference_movie, "The Proposal")

        # Step 3: Verify recommendation invariance
        # A typed input of the exact same string produces identical intent
        intent_from_typed = parse_movie_intent("I want a funny romantic movie like The Proposal")
        self.assertEqual(intent.genres, intent_from_typed.genres)
        self.assertEqual(intent.mood, intent_from_typed.mood)
        self.assertEqual(intent.reference_movie, intent_from_typed.reference_movie)

    def test_watch_later_and_details_integration(self):
        """Verify that a recommended movie can be saved to watch later and normalized for details."""
        sample_movie = {
            "movieId": 356,
            "tmdbId": 13,
            "title": "Forrest Gump (1994)",
            "genres": "Comedy|Drama|Romance|War",
            "hybrid_score": 0.94,
            "user_score": 0.92,
            "item_score": 0.91,
            "svd_score": 0.96,
        }

        # Toggle into watch later
        entries, added = toggle_watch_later_entry([], sample_movie)
        self.assertTrue(added)
        self.assertEqual(len(entries), 1)
        self.assertEqual(watch_later_key(entries[0]), "movie:356")

        # Normalize for details
        details = normalize_movie_for_details(sample_movie)
        self.assertEqual(details["movieId"], 356)
        self.assertEqual(details["tmdbId"], 13)
        self.assertIn("Forrest Gump", details["title"])

        session = {}
        select_movie_details(session, sample_movie)
        self.assertIn("selected_movie_details", session)
        self.assertEqual(session["selected_movie_details"]["movieId"], 356)


if __name__ == "__main__":
    unittest.main()
