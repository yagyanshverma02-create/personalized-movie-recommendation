import ast
import json
from pathlib import Path
import tempfile
import unittest

from watch_later import DEFAULT_WATCH_LATER_PROFILE_ID, toggle_watch_later_entry


class WidgetLockedSessionState(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.locked_widget_keys = set()

    def __setitem__(self, key, value):
        if key in self.locked_widget_keys:
            raise AssertionError(f"widget key {key!r} was modified after instantiation")
        super().__setitem__(key, value)


class ProfileWatchLaterIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "user_profiles.json"
        self.path.write_text(json.dumps([
            {"profile_id": "one", "name": "One", "movies": [], "watch_later": []},
            {"profile_id": "two", "name": "Two", "movies": [], "watch_later": []},
        ]), encoding="utf-8")
        self.state = WidgetLockedSessionState(
            {"profile_id": "one", "profile": [], "profile_name": "One"}
        )

        def load_profiles():
            return json.loads(self.path.read_text(encoding="utf-8"))

        def save_profiles(profiles):
            self.path.write_text(json.dumps(profiles), encoding="utf-8")

        source = Path(__file__).resolve().parents[1] / "app.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {"get_active_watch_later_entries", "toggle_active_watch_later"}
        functions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        streamlit_stub = type("StreamlitStub", (), {"session_state": self.state})()
        self.namespace = {
            "st": streamlit_stub,
            "load_profiles": load_profiles,
            "save_profiles": save_profiles,
            "toggle_watch_later_entry": toggle_watch_later_entry,
            "DEFAULT_WATCH_LATER_PROFILE_ID": DEFAULT_WATCH_LATER_PROFILE_ID,
            "profile_preview": [],
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), self.namespace)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_remove_persists_on_active_profile_without_touching_other_profiles(self):
        self.state.locked_widget_keys.add("profile_name")
        toggle = self.namespace["toggle_active_watch_later"]
        toggle({"movieId": 42})
        persisted = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(persisted[0]["watch_later"], [{"movieId": 42}])
        self.assertEqual(persisted[1]["watch_later"], [])

        toggle({"movieId": 42})
        persisted = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(persisted[0]["watch_later"], [])

    def test_watch_later_without_active_profile_does_not_mutate_profile_name_widget(self):
        self.state.pop("profile_id")
        self.state.locked_widget_keys.add("profile_name")
        saved = self.namespace["toggle_active_watch_later"]({"movieId": 77})
        persisted = json.loads(self.path.read_text(encoding="utf-8"))
        default_profile = next(
            item for item in persisted
            if item["profile_id"] == DEFAULT_WATCH_LATER_PROFILE_ID
        )
        self.assertTrue(saved)
        self.assertEqual(default_profile["name"], "One")
        self.assertEqual(default_profile["watch_later"], [{"movieId": 77}])

    def test_switching_profile_loads_its_own_watch_later_list(self):
        toggle = self.namespace["toggle_active_watch_later"]
        get_entries = self.namespace["get_active_watch_later_entries"]
        toggle({"movieId": 42})
        self.state["profile_id"] = "two"
        self.assertEqual(get_entries(), [])
        self.state["profile_id"] = "one"
        self.assertEqual(get_entries(), [{"movieId": 42}])


if __name__ == "__main__":
    unittest.main()
