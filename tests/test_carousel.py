"""Unit tests for the 3D circular carousel component."""

import unittest
import pandas as pd
from unittest.mock import patch

from carousel_component import _clean_title_year, _card_image_uri, render_circular_carousel


class CarouselTests(unittest.TestCase):
    def test_clean_title_year(self):
        title, year = _clean_title_year("Inception (2010)")
        self.assertEqual(title, "Inception")
        self.assertEqual(year, "2010")

        title2, year2 = _clean_title_year("Se7en (1995)")
        self.assertEqual(title2, "Se7en")
        self.assertEqual(year2, "1995")

        title3, year3 = _clean_title_year("Movie Without Year")
        self.assertEqual(title3, "Movie Without Year")
        self.assertEqual(year3, "")

    def test_card_image_uri_fallback(self):
        fallback_uri = _card_image_uri(None)
        self.assertTrue(fallback_uri.startswith("data:image/svg+xml;base64,"))

        custom_bytes = b"fake_jpeg_data"
        jpeg_uri = _card_image_uri(custom_bytes)
        self.assertTrue(jpeg_uri.startswith("data:image/jpeg;base64,"))

    @patch("streamlit.components.v1.html")
    def test_render_circular_carousel_with_movies(self, mock_html):
        sample_df = pd.DataFrame([
            {
                "movieId": 1,
                "tmdbId": 862,
                "title": "Toy Story (1995)",
                "genres": "Animation|Children|Comedy",
                "hybrid_score": 0.92,
            },
            {
                "movieId": 2,
                "tmdbId": 8844,
                "title": "Jumanji (1995)",
                "genres": "Adventure|Children|Fantasy",
                "hybrid_score": 0.88,
            },
        ])
        render_circular_carousel(sample_df, "personalized", {})
        mock_html.assert_called_once()
        html_arg = mock_html.call_args[0][0]
        self.assertIn("Toy Story", html_arg)
        self.assertIn("Jumanji", html_arg)
        self.assertIn('"score": 92', html_arg)

    @patch("streamlit.components.v1.html")
    def test_render_circular_carousel_empty_dataframe(self, mock_html):
        render_circular_carousel(pd.DataFrame(), "personalized", {})
        mock_html.assert_not_called()
        render_circular_carousel(None, "personalized", {})
        mock_html.assert_not_called()


if __name__ == "__main__":
    unittest.main()
