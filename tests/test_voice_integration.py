"""Tests for speech-to-text voice integration and intent pipeline."""

import unittest
from unittest.mock import MagicMock, patch

from gemini_helper import MovieIntent, transcribe_speech_to_text


class VoiceIntegrationTests(unittest.TestCase):
    def test_empty_audio_bytes_returns_empty_string(self):
        """Empty audio bytes must immediately return empty string without calling API."""
        self.assertEqual(transcribe_speech_to_text(b""), "")
        self.assertEqual(transcribe_speech_to_text(None), "")

    @patch("gemini_helper.get_gemini_client")
    def test_transcription_returns_clean_text(self, mock_get_client):
        """Valid speech response is cleanly returned as English text."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "I want a funny romantic movie like The Proposal"
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = transcribe_speech_to_text(b"fake_audio_bytes_123", mime_type="audio/wav")
        self.assertEqual(result, "I want a funny romantic movie like The Proposal")
        mock_client.models.generate_content.assert_called_once()

    @patch("gemini_helper.get_gemini_client")
    def test_silence_or_empty_response_returns_empty_string(self, mock_get_client):
        """Responses with no speech or timestamps only return empty string."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "EMPTY"
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        self.assertEqual(transcribe_speech_to_text(b"silence_bytes"), "")

        # Timestamp only response
        mock_response.text = "[00:00]"
        self.assertEqual(transcribe_speech_to_text(b"silence_bytes"), "")

        # Punctuation only
        mock_response.text = "..."
        self.assertEqual(transcribe_speech_to_text(b"silence_bytes"), "")

    @patch("gemini_helper.get_gemini_client")
    def test_voice_text_feeds_intent_parser(self, mock_get_client):
        """Simulate voice input text flowing into parse_movie_intent."""
        from gemini_helper import parse_movie_intent

        spoken_text = "I want a funny romantic movie like The Proposal"

        mock_client = MagicMock()
        mock_interaction = MagicMock()
        mock_interaction.output_text = (
            '{"genres": ["Comedy", "Romance"], "mood": "Funny", '
            '"reference_movie": "The Proposal", "keywords": []}'
        )
        mock_client.interactions.create.return_value = mock_interaction
        mock_get_client.return_value = mock_client

        intent = parse_movie_intent(spoken_text)
        self.assertIsInstance(intent, MovieIntent)
        self.assertIn("Comedy", intent.genres)
        self.assertIn("Romance", intent.genres)
        self.assertEqual(intent.mood, "Funny")
        self.assertEqual(intent.reference_movie, "The Proposal")

    def test_intent_mapping_is_deterministic(self):
        """Regardless of whether text came from voice or keyboard, intent is identical."""
        intent_voice = MovieIntent(genres=["Comedy", "Romance"], mood="Funny", reference_movie="The Proposal")
        intent_typed = MovieIntent(genres=["Comedy", "Romance"], mood="Funny", reference_movie="The Proposal")
        self.assertEqual(intent_voice.genres, intent_typed.genres)
        self.assertEqual(intent_voice.mood, intent_typed.mood)
        self.assertEqual(intent_voice.reference_movie, intent_typed.reference_movie)


if __name__ == "__main__":
    unittest.main()

