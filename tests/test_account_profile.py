import json
import re
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from account_profile import (
    clear_auth_session,
    get_or_create_account_profile,
    is_valid_gmail,
    load_profiles,
    normalize_username,
    set_account_password,
    verify_account_password,
)


class AccountProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "user_profiles.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_profiles(self, profiles):
        self.path.write_text(json.dumps(profiles), encoding="utf-8")

    def test_username_is_required_trimmed_and_simple(self):
        self.assertIsNone(normalize_username("   "))
        self.assertIsNone(normalize_username("a" * 41))
        self.assertIsNone(normalize_username("<script>"))
        self.assertEqual(normalize_username("  Movie Fan  "), "Movie Fan")

    def test_gmail_validation_preserves_gmail_only_behavior(self):
        self.assertTrue(is_valid_gmail(" Viewer.Name+demo@GMAIL.COM "))
        self.assertFalse(is_valid_gmail("viewer@example.com"))
        self.assertFalse(is_valid_gmail("not-an-email"))

    def test_existing_gmail_returns_existing_profile(self):
        existing = {
            "profile_id": "MM-123ABC",
            "email": "viewer@gmail.com",
            "name": "Viewer",
            "movies": [{"movieId": 10, "rating": 4.5}],
            "watch_later": [{"movieId": 11}],
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        self.write_profiles([existing])

        profile, profiles = get_or_create_account_profile(
            " VIEWER@GMAIL.COM ", self.path, username="  New Display Name  "
        )

        self.assertEqual(profile["profile_id"], "MM-123ABC")
        self.assertEqual(profile["name"], "New Display Name")
        self.assertEqual(profile["movies"], existing["movies"])
        self.assertEqual(profile["watch_later"], existing["watch_later"])
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0]["email"], existing["email"])

    def test_new_gmail_creates_one_profile_with_existing_id_format(self):
        self.write_profiles([])

        profile, profiles = get_or_create_account_profile(
            "new.user@gmail.com", self.path, username="The Movie Fan"
        )

        self.assertEqual(len(profiles), 1)
        self.assertRegex(profile["profile_id"], re.compile(r"^MM-[0-9A-F]{6}$"))
        self.assertEqual(profile["email"], "new.user@gmail.com")
        self.assertEqual(profile["name"], "The Movie Fan")
        self.assertEqual(profile["movies"], [])
        self.assertEqual(profile["watch_later"], [])
        self.assertIn("created_at", profile)
        self.assertIn("updated_at", profile)

    def test_repeated_login_does_not_create_duplicate_profiles(self):
        first, _ = get_or_create_account_profile("same@gmail.com", self.path)
        second, profiles = get_or_create_account_profile("same@gmail.com", self.path)

        self.assertEqual(first["profile_id"], second["profile_id"])
        self.assertEqual(len(profiles), 1)

    def test_profiles_without_email_still_load_unchanged(self):
        legacy = {
            "profile_id": "MM-LEGACY",
            "name": "Legacy",
            "movies": [{"movieId": 1, "rating": 4.0}],
        }
        self.write_profiles([legacy])

        self.assertEqual(load_profiles(self.path), [legacy])
        self.assertNotIn("email", load_profiles(self.path)[0])

    def test_sign_out_clears_auth_state_without_deleting_profile_data(self):
        profile = {"profile_id": "MM-ABC123", "email": "saved@gmail.com", "movies": [{"movieId": 2, "rating": 5.0}]}
        self.write_profiles([profile])
        state = {
            "_auth_state": "done",
            "_auth_user": "saved@gmail.com",
            "_auth_profile_id": profile["profile_id"],
            "_otp_demo_value": "123456",
            "profile_id": profile["profile_id"],
        }

        clear_auth_session(state)

        self.assertEqual(state, {"profile_id": profile["profile_id"]})
        self.assertEqual(load_profiles(self.path), [profile])

    def test_password_is_persisted_as_salted_hash_in_sqlite(self):
        self.write_profiles([{
            "profile_id": "MM-ABC123",
            "email": "viewer@gmail.com",
            "name": "Viewer",
            "movies": [],
            "watch_later": [],
        }])

        test_password = "correct horse battery staple"
        self.assertTrue(set_account_password("VIEWER@gmail.com", test_password, self.path))
        self.assertEqual(verify_account_password("viewer@gmail.com", test_password, self.path)["profile_id"], "MM-ABC123")
        self.assertIsNone(verify_account_password("viewer@gmail.com", "incorrect password", self.path))

        database_path = self.path.parent / "accounts.sqlite3"
        with closing(sqlite3.connect(database_path)) as connection:
            email, salt, password_hash = connection.execute(
                "SELECT email, password_salt, password_hash FROM account_credentials"
            ).fetchone()
        self.assertEqual(email, "viewer@gmail.com")
        self.assertNotEqual(password_hash, test_password)
        self.assertNotEqual(salt, test_password)
        stored_profile = load_profiles(self.path)[0]
        self.assertNotIn("password_hash", stored_profile)
        self.assertNotIn("password_salt", stored_profile)


if __name__ == "__main__":
    unittest.main()
