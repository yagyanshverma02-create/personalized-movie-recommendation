import ast
from datetime import date
from pathlib import Path
import unittest
from typing import Any


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSecrets:
    def get(self, name):
        return "test-key" if name == "TMDB_API_KEY" else None


class TmdbDiscoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "tmdb_helper.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_get_discover_movies_cached", "get_discover_movies"}
        ]
        cls.calls = []
        cls.fail_request = False

        class Requests:
            @classmethod
            def get(cls, url, params, timeout):
                TmdbDiscoverTests.calls.append((url, params, timeout))
                if TmdbDiscoverTests.fail_request:
                    raise TimeoutError("TMDB timeout")
                return FakeResponse({"results": [{
                    "id": 10,
                    "title": "Example",
                    "overview": "An overview",
                    "genre_ids": [35],
                    "vote_average": 7.5,
                }]})

        namespace = {
            "Any": Any,
            "date": date,
            "TMDB_API_BASE": "https://api.themoviedb.org/3",
            "st": type("StreamlitStub", (), {
                "secrets": FakeSecrets(),
                "cache_data": staticmethod(lambda **kwargs: lambda function: function),
            })(),
            "requests": Requests,
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        cls.get_movies = staticmethod(namespace["get_discover_movies"])

    def setUp(self):
        self.calls.clear()
        type(self).fail_request = False

    def test_all_supported_categories_use_live_tmdb_routes(self):
        expected = {
            "Trending": "/trending/movie/week",
            "Popular": "/movie/popular",
            "Latest": "/discover/movie",
            "Upcoming": "/movie/upcoming",
            "Highest Rated": "/movie/top_rated",
        }
        for category, suffix in expected.items():
            with self.subTest(category=category):
                result = self.get_movies(category)
                self.assertEqual(len(result), 1)
                self.assertIn(suffix, self.calls[-1][0])
                self.assertEqual(self.calls[-1][2], 8)
        latest_params = next(params for url, params, _ in self.calls if url.endswith("/discover/movie"))
        self.assertEqual(latest_params["sort_by"], "primary_release_date.desc")
        self.assertIn("release_date.lte", latest_params)

    def test_invalid_category_and_tmdb_failure_are_safe(self):
        self.assertEqual(self.get_movies("Kids & Family"), [])
        type(self).fail_request = True
        self.assertEqual(self.get_movies("Trending"), [])


if __name__ == "__main__":
    unittest.main()
