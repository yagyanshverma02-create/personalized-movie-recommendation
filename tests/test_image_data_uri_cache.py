import ast
import base64
from functools import wraps
from pathlib import Path
import unittest


class ImageDataUriCacheTests(unittest.TestCase):
    def test_identical_image_conversion_is_cached_without_changing_uri(self):
        source = Path(__file__).resolve().parents[1] / "app.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "image_data_uri"
        )
        calls = []

        def cache_data(**_options):
            def decorate(target):
                values = {}

                @wraps(target)
                def cached(*args):
                    if args not in values:
                        calls.append(1)
                        values[args] = target(*args)
                    return values[args]
                return cached
            return decorate

        streamlit_stub = type(
            "StreamlitStub", (), {"cache_data": staticmethod(cache_data)}
        )()
        namespace = {
            "st": streamlit_stub,
            "base64": base64,
            "POSTER_PLACEHOLDER": "<svg/>",
        }
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        image_data_uri = namespace["image_data_uri"]
        image = b"poster-image-bytes"
        expected = "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
        self.assertEqual(image_data_uri(image), expected)
        self.assertEqual(image_data_uri(image), expected)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
