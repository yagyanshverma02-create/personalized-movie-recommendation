import sys
import types
import unittest
from unittest.mock import MagicMock, patch


try:
    import streamlit  # noqa: F401
except ImportError:
    streamlit_stub = types.ModuleType("streamlit")
    streamlit_stub.session_state = {}
    streamlit_stub.secrets = {}
    sys.modules["streamlit"] = streamlit_stub

import login


class LoginEmailTests(unittest.TestCase):
    def setUp(self):
        login.st.session_state.clear()
        self.smtp = MagicMock()
        smtp_context = MagicMock()
        smtp_context.__enter__.return_value = self.smtp
        smtp_context.__exit__.return_value = False
        self.smtp_factory = patch("login.smtplib.SMTP", return_value=smtp_context)
        self.mock_smtp = self.smtp_factory.start()
        self.secrets_patch = patch.object(
            login.st,
            "secrets",
            {
                "SMTP_HOST": "smtp.gmail.com",
                "SMTP_PORT": 587,
                "SMTP_EMAIL": "moviemind.sender@gmail.com",
                "SMTP_APP_PASSWORD": "app-password-from-secrets",
            },
        )
        self.secrets_patch.start()

    def tearDown(self):
        self.secrets_patch.stop()
        self.smtp_factory.stop()
        login.st.session_state.clear()

    def test_email_uses_configured_smtp_and_correct_recipient_subject_and_otp(self):
        login._send_otp("viewer@gmail.com", "123456", "Movie Fan")

        self.mock_smtp.assert_called_once_with("smtp.gmail.com", 587, timeout=15)
        self.smtp.login.assert_called_once_with("moviemind.sender@gmail.com", "app-password-from-secrets")
        message = self.smtp.send_message.call_args.args[0]
        self.assertEqual(message["To"], "viewer@gmail.com")
        self.assertEqual(message["Subject"], "MovieMind Login OTP")
        body = message.get_content()
        self.assertIn("Hello Movie Fan,", body)
        self.assertIn("123456", body)
        self.assertIn("This code expires in 5 minutes.", body)

    def test_missing_smtp_secrets_raise_safe_configuration_error(self):
        with patch.object(login.st, "secrets", {"SMTP_HOST": "smtp.gmail.com"}):
            with self.assertRaises(login.EmailConfigurationError) as raised:
                login._send_otp("viewer@gmail.com", "123456", "Viewer")

        self.assertIn("SMTP_HOST", str(raised.exception))
        self.assertNotIn("app-password-from-secrets", str(raised.exception))

    def test_send_failure_does_not_advance_authentication_or_store_otp(self):
        with patch.object(login, "_send_otp", side_effect=OSError("private smtp detail")):
            with self.assertRaises(OSError):
                login._start_otp_challenge("viewer@gmail.com", "Viewer")

        self.assertNotEqual(login.st.session_state.get("_auth_state"), "done")
        self.assertNotIn("_auth_otp", login.st.session_state)
        self.assertNotIn("_auth_user", login.st.session_state)

    def test_otp_generation_is_exactly_six_digits(self):
        for _ in range(100):
            self.assertRegex(login._generate_otp(), r"^\d{6}$")

    def test_successful_authentication_requires_password_setup_and_clears_otp(self):
        login.st.session_state.update({
            "_auth_otp": "123456",
            "_auth_otp_ts": 100.0,
            "_otp_attempts": 1,
            "_otp_demo_value": "123456",
            "_auth_pending_username": "Viewer",
        })
        with patch.object(login, "get_or_create_account_profile", return_value=({"profile_id": "MM-ABC123"}, [])):
            login._complete_otp_authentication("viewer@gmail.com", "Viewer")

        self.assertEqual(login.st.session_state["_auth_state"], "password_setup")
        self.assertEqual(login.st.session_state["_auth_setup_email"], "viewer@gmail.com")
        self.assertEqual(login.st.session_state["_auth_setup_username"], "Viewer")
        self.assertNotIn("_auth_user", login.st.session_state)
        self.assertNotIn("_auth_username", login.st.session_state)
        self.assertNotIn("_auth_otp", login.st.session_state)
        self.assertNotIn("_otp_demo_value", login.st.session_state)

    def test_expiry_and_excessive_attempts_clear_otp_and_stop_challenge(self):
        login.st.session_state.update({
            "_auth_state": "otp",
            "_auth_email": "viewer@gmail.com",
            "_auth_otp": "123456",
            "_auth_otp_ts": 100.0,
            "_auth_pending_username": "Viewer",
        })
        login._expire_otp_challenge()
        self.assertEqual(login.st.session_state["_auth_state"], "email")
        self.assertNotIn("_auth_otp", login.st.session_state)

        login.st.session_state.update({
            "_auth_state": "otp",
            "_auth_email": "viewer@gmail.com",
            "_auth_otp": "654321",
            "_otp_attempts": 0,
        })
        remaining = [login._record_failed_otp_attempt() for _ in range(login._OTP_MAX_ATTEMPTS)]
        self.assertEqual(remaining, [4, 3, 2, 1, 0])
        self.assertTrue(login._otp_attempts_exhausted())
        self.assertEqual(login.st.session_state["_auth_state"], "email")
        self.assertNotIn("_auth_otp", login.st.session_state)

    def test_sign_out_clears_otp_session_values(self):
        login.st.session_state.update({"_auth_otp": "123456", "_otp_attempts": 2, "_auth_user": "viewer@gmail.com"})
        login.clear_auth_session(login.st.session_state)
        self.assertFalse(login.st.session_state)


if __name__ == "__main__":
    unittest.main()
