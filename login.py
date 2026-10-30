"""
MovieMind Authentication Module
--------------------------------
Gmail-format email validation + 6-digit OTP verification.
OTP email is sent through the SMTP settings configured in Streamlit Secrets.
"""

from __future__ import annotations

import time
import html
import secrets
import smtplib
import ssl
from email.message import EmailMessage

import streamlit as st
from account_profile import (
    clear_auth_session,
    get_or_create_account_profile,
    is_valid_gmail,
    normalize_username,
    set_account_password,
    verify_account_password,
)

_OTP_TTL_SECONDS = 300  # OTP expires after 5 minutes
_OTP_MAX_ATTEMPTS = 5


@st.fragment(run_every="1s")
def _render_otp_countdown() -> None:
    """Refresh the visible expiry clock without rerunning the entire login page."""
    started_at = st.session_state.get("_auth_otp_ts", time.time())
    remaining = max(0, int(_OTP_TTL_SECONDS - (time.time() - started_at)))
    if remaining:
        mins, secs = divmod(remaining, 60)
        label = f"Expires in {mins:01d}:{secs:02d}"
    else:
        label = "Code expired · resend a new one"
    st.markdown(f'<div class="otp-timer">{html.escape(label)}</div>', unsafe_allow_html=True)


class EmailConfigurationError(Exception):
    """Raised when required SMTP settings are not present or usable."""


def _smtp_configuration() -> tuple[str, int, str, str, str]:
    """Read SMTP configuration from Streamlit Secrets without exposing it."""
    try:
        host = str(st.secrets["SMTP_HOST"]).strip()
        port = int(st.secrets["SMTP_PORT"])
        sender = str(st.secrets["SMTP_EMAIL"]).strip()
        password = str(st.secrets["SMTP_APP_PASSWORD"]).strip()
        # Brevo's SMTP login can differ from the verified sender address.
        # Keep SMTP_USER optional for compatibility with providers where they match.
        smtp_user = str(st.secrets.get("SMTP_USER", sender)).strip()
    except Exception:
        raise EmailConfigurationError(
            "Brevo email delivery is not configured. Set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_EMAIL, and SMTP_APP_PASSWORD in Streamlit Secrets."
        ) from None

    if not host or not smtp_user or not sender or not password or not 1 <= port <= 65535:
        raise EmailConfigurationError(
            "Brevo email delivery is not configured. Set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_EMAIL, and SMTP_APP_PASSWORD in Streamlit Secrets."
        )
    return host, port, smtp_user, sender, password


def _is_gmail(email: str) -> bool:
    """Return True only for valid @gmail.com addresses."""
    return is_valid_gmail(email)


def _generate_otp() -> str:
    return f"{secrets.randbelow(900000) + 100000}"


def _send_otp(email: str, otp: str, username: str) -> None:
    """Send a login OTP using the server-side SMTP settings in Streamlit Secrets."""
    host, port, smtp_user, sender, password = _smtp_configuration()
    message = EmailMessage()
    message["Subject"] = "MovieMind Login OTP"
    message["From"] = sender
    message["To"] = email
    message.set_content(
        f"Hello {username},\n\n"
        "Your MovieMind verification code is:\n\n"
        f"{otp}\n\n"
        "This code expires in 5 minutes.\n\n"
        "If you did not request this code, you can ignore this email.\n\n"
        "MovieMind\n"
        "B.Tech CSE Project"
    )

    with smtplib.SMTP(host, port, timeout=15) as smtp:
        smtp.ehlo()
        smtp.starttls(context=ssl.create_default_context())
        smtp.ehlo()
        smtp.login(smtp_user, password)
        smtp.send_message(message)


def _email_send_error(exc: Exception) -> str:
    """Return a useful SMTP failure hint without exposing credentials."""
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        if exc.smtp_code == 525:
            return (
                "Brevo blocked this computer's IP address (SMTP error 525). "
                "Authorize the blocked IP in Brevo under Settings → Security → Authorized IPs, then try again."
            )
        return (
            "Brevo rejected the SMTP login. Check SMTP_USER and make sure "
            "SMTP_APP_PASSWORD is the SMTP key from Brevo, not an API key."
        )
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        return (
            "Brevo rejected the recipient address. Check the Gmail address "
            "and your Brevo account's sending status."
        )
    if isinstance(exc, smtplib.SMTPResponseException):
        return f"Brevo SMTP returned error {exc.smtp_code}. Check the SMTP settings and verified sender."
    if isinstance(exc, (TimeoutError, OSError, smtplib.SMTPServerDisconnected)):
        return "Could not complete the connection to Brevo. Check SMTP_HOST, SMTP_PORT, and network access."
    return f"We couldn't send the verification email ({type(exc).__name__}). Please check the Brevo SMTP settings."


def _clear_otp_challenge(clear_email: bool = False) -> None:
    for key in ("_auth_otp", "_auth_otp_ts", "_otp_attempts", "_otp_demo_value", "_auth_pending_username"):
        st.session_state.pop(key, None)
    st.session_state.pop("_auth_email_sent_notice", None)
    if clear_email:
        st.session_state.pop("_auth_email", None)


def _start_otp_challenge(email: str, username: str) -> None:
    """Send first; only then put the OTP challenge into the auth session."""
    otp = _generate_otp()
    _send_otp(email, otp, username)
    st.session_state["_auth_pending_username"] = username
    st.session_state["_auth_email"] = email
    st.session_state["_auth_otp"] = otp
    st.session_state["_auth_otp_ts"] = time.time()
    st.session_state["_auth_method"] = "otp"
    st.session_state["_auth_state"] = "otp"
    st.session_state["_otp_attempts"] = 0
    st.session_state.pop("_otp_demo_value", None)
    st.session_state["_auth_email_sent_notice"] = True


def _complete_otp_authentication(email: str, username: str) -> None:
    get_or_create_account_profile(email, username=username)
    # Completing OTP always leads to password setup. For an account that
    # already has a password, this lets the user reset it after re-verifying.
    st.session_state["_auth_setup_email"] = email.strip().lower()
    st.session_state["_auth_setup_username"] = username
    st.session_state["_auth_state"] = "password_setup"
    _clear_otp_challenge()


def _complete_password_setup(email: str, username: str, password: str) -> bool:
    if not set_account_password(email, password):
        return False
    profile, _ = get_or_create_account_profile(email, username=username)
    _complete_password_authentication(profile)
    st.session_state.pop("_auth_setup_email", None)
    st.session_state.pop("_auth_setup_username", None)
    return True


def _complete_password_authentication(profile: dict) -> None:
    st.session_state["_auth_state"] = "done"
    st.session_state["_auth_method"] = "password"
    st.session_state["_auth_user"] = str(profile.get("email", "")).strip().lower()
    st.session_state["_auth_username"] = profile.get("name", "MovieMind user")
    st.session_state["_auth_profile_id"] = profile.get("profile_id", "")


def _record_failed_otp_attempt() -> int:
    attempts = st.session_state.get("_otp_attempts", 0)
    st.session_state["_otp_attempts"] = attempts + 1
    return max(0, _OTP_MAX_ATTEMPTS - attempts - 1)


def _otp_attempts_exhausted() -> bool:
    if st.session_state.get("_otp_attempts", 0) < _OTP_MAX_ATTEMPTS:
        return False
    _clear_otp_challenge(clear_email=True)
    st.session_state["_auth_state"] = "email"
    return True


def _expire_otp_challenge() -> None:
    _clear_otp_challenge(clear_email=True)
    st.session_state["_auth_state"] = "email"


def render_login_page(background_image_uri: str = "") -> bool:
    """
    Render the full-screen login / OTP page.
    Returns True when the user is authenticated, False otherwise.
    """
    background_image_uri = background_image_uri or str(st.session_state.get("_login_background_uri", ""))

    # ── Auth state machine ──────────────────────────────────────────────
    auth_state = st.session_state.get("_auth_state", "password")  # password | email | otp | password_setup | done

    if auth_state == "done":
        if st.session_state.get("_auth_method") == "password":
            return True
        # Upgrade an already-open session authenticated by the older OTP-only
        # flow. It must set a password before continuing to the app.
        st.session_state["_auth_setup_email"] = str(st.session_state.get("_auth_user", "")).strip().lower()
        st.session_state["_auth_setup_username"] = st.session_state.get("_auth_username", "")
        st.session_state["_auth_state"] = "password_setup"
        auth_state = "password_setup"

    # ── Full-screen login CSS ───────────────────────────────────────────
    st.markdown("""
<style>
/* Hide default Streamlit chrome on login page */
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
header { display: none !important; }
.block-container { padding-top: 0 !important; padding-bottom: 0 !important; max-width: 1100px !important; }
[data-testid="stMainBlockContainer"] {
  min-height: 100vh; display: flex; flex-direction: column; justify-content: center;
  padding-top: 28px !important; padding-bottom: 28px !important;
}
[data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"] { width: 100%; }

/* Page backdrop */
[data-testid="stAppViewContainer"] {
  background: radial-gradient(ellipse at 30% 20%, #1a2235 0%, #0a0d12 60%, #060709 100%) !important;
}

/* Real Streamlit container keeps the entire form inside the centered card. */
.st-key-login_card {
  width: 100%;
  max-width: 520px;
  margin: 0 auto;
  padding: 34px 36px 26px;
  background: linear-gradient(160deg, rgba(22,28,40,.98), rgba(12,15,20,.98));
  border: 1px solid rgba(255,255,255,.09);
  border-radius: 22px;
  box-shadow: 0 32px 80px rgba(0,0,0,.55), 0 0 0 1px rgba(255,255,255,.04);
}
.login-logo-wrap {
  display: flex;
  align-items: center;
  gap: 12px;
  justify-content: center;
  margin-bottom: 22px;
}
.login-logo-mark {
  width: 44px; height: 44px;
  border-radius: 14px;
  background: linear-gradient(145deg, #e0a66d, #b76575);
  display: grid; place-items: center;
  font-size: 22px; font-weight: 900; color: #13111a;
}
.login-logo-name {
  font-size: 26px; font-weight: 780; letter-spacing: -.7px; color: #f6f7fb;
}
.login-logo-sub {
  font-size: 11px; letter-spacing: .3px; color: #9aa5b4; margin-top: 4px;
}
.login-heading {
  text-align: center; font-size: 22px; font-weight: 720;
  color: #f0f2f6; margin-bottom: 6px;
}
.login-sub {
  text-align: center; font-size: 13px; color: #8a95a6; margin-bottom: 18px;
}
.login-divider {
  border: none; border-top: 1px solid rgba(255,255,255,.08); margin: 22px 0;
}
.login-footer {
  text-align: center; font-size: 10px; color: #596070; margin-top: 18px; line-height: 1.5;
}
.st-key-login_card [data-testid="stForm"] { padding: 0; border: 0; background: transparent; }
.st-key-login_card [data-testid="stWidgetLabel"] p { color: #e1e5eb; font-size: 12px; font-weight: 650; }
.st-key-login_card input { min-height: 46px !important; }
.st-key-login_card [data-testid="stFormSubmitButton"] button { min-height: 46px; margin-top: 8px; }
.otp-timer { font-size: 11px; color: #7a8698; text-align: center; margin-top: 4px; }
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding: 20px 12px !important; }
  .st-key-login_card { padding: 26px 20px 20px; border-radius: 18px; }
  .login-logo-name { font-size: 23px; }
}

/* Match the main app's cinematic, red-accented streaming interface. */
[data-testid="stAppViewContainer"] {
  background:radial-gradient(ellipse at 50% 0%,#4a0d14 0%,#21090d 35%,#0b090a 78%)!important;
}
[data-testid="stMainBlockContainer"] { max-width:1180px!important; }
.st-key-login_card {
  max-width:550px;
  padding:34px 40px 28px;
  background:linear-gradient(150deg,rgba(35,13,17,.97),rgba(15,12,14,.97));
  border:1px solid rgba(255,255,255,.13);
  border-radius:10px;
  box-shadow:0 28px 85px rgba(0,0,0,.65),0 0 90px rgba(229,9,20,.08);
}
.login-logo-wrap { justify-content:flex-start;margin-bottom:30px; }
.login-logo-mark { background:#e50914;color:#fff;border-radius:7px;box-shadow:0 5px 22px #e5091433; }
.login-logo-name { color:#f6121d;font-size:27px;letter-spacing:.6px;text-transform:uppercase; }
.login-logo-sub { color:#bcb0b3;letter-spacing:1.1px; }
.login-heading { text-align:left;font-size:30px;font-weight:800;color:#fff;margin-bottom:8px; }
.login-sub { text-align:left;font-size:15px;color:#c8bfc1;margin-bottom:22px; }
.login-divider { border-color:rgba(255,255,255,.1);margin:24px 0 16px; }
.login-footer { text-align:left;font-size:11px;color:#8f8387; }
.st-key-login_card [data-testid="stWidgetLabel"] p { color:#f2e9eb;font-size:14px;font-weight:600; }
.st-key-login_card input {
  min-height:52px!important;
  background:#21171a!important;
  border:1px solid #6f3e45!important;
  border-radius:5px!important;
  color:#fff!important;
  font-size:16px!important;
}
.st-key-login_card input:focus { border-color:#f6121d!important;box-shadow:0 0 0 2px #e509142b!important; }
.st-key-login_card [data-testid="stFormSubmitButton"] button {
  min-height:52px;
  margin-top:8px;
  border-radius:5px;
  background:#e50914!important;
  border:0;
  color:#fff!important;
  font-size:17px;
  font-weight:750;
}
.st-key-login_card [data-testid="stFormSubmitButton"] button:hover { background:#c11119!important; }
.st-key-login_card .stButton > button {
  border-radius:5px;
  background:#251619;
  border:1px solid #6f3e45;
  color:#fff;
  min-height:44px;
}
.st-key-login_card .stButton > button:hover { border-color:#f6121d;background:#34191e; }
.st-key-login_card [data-testid="stAlert"] { border-radius:5px; }
.otp-timer { color:#c7b7bb; }
@media (max-width:640px) {
  .st-key-login_card { padding:26px 22px 22px;border-radius:8px; }
  .login-heading { font-size:26px; }
  .login-logo-name { font-size:23px; }
}
/* Retro 35mm palette shared with the signed-in MovieMind interface. */
:root { --retro-espresso:#1A120B; --retro-leather:#3C2A21; --retro-parchment:#F5EBE6; --retro-terracotta:#E07A5F; --retro-gold:#F2CC8F; }
[data-testid="stAppViewContainer"] { background:radial-gradient(ellipse at 50% 0%,#3C2A21 0%,#241910 38%,#1A120B 82%)!important;color:var(--retro-parchment)!important; }
.st-key-login_card { background:linear-gradient(150deg,rgba(60,42,33,.97),rgba(26,18,11,.98))!important;border-color:#F2CC8F42!important;box-shadow:0 28px 85px rgba(8,5,3,.7),0 0 90px rgba(224,122,95,.12)!important; }
.login-logo-mark { background:var(--retro-terracotta)!important;color:var(--retro-espresso)!important;box-shadow:0 5px 22px #E07A5F55!important; }
.login-logo-name { color:var(--retro-gold)!important; }
.login-logo-sub,.login-sub,.login-footer { color:#d7c4b9!important; }
.login-heading { color:var(--retro-parchment)!important; }
.st-key-login_card [data-testid="stWidgetLabel"] p { color:var(--retro-parchment)!important; }
.st-key-login_card input { background:#3C2A21!important;border-color:#F2CC8F55!important;color:var(--retro-parchment)!important; }
.st-key-login_card input:focus { border-color:var(--retro-gold)!important;box-shadow:0 0 0 2px #F2CC8F35!important; }
.st-key-login_card [data-testid="stFormSubmitButton"] button { background:var(--retro-terracotta)!important;color:var(--retro-espresso)!important; }
.st-key-login_card [data-testid="stFormSubmitButton"] button:hover { background:#ed9276!important; }
.st-key-login_card .stButton > button { background:#3C2A21!important;border-color:#F2CC8F50!important;color:var(--retro-parchment)!important; }
.st-key-login_card .stButton > button:hover { background:#52382b!important;border-color:var(--retro-terracotta)!important; }
.st-key-login_card [data-testid="stAlert"] { background:#3C2A21!important;border-color:#F2CC8F55!important;color:var(--retro-parchment)!important; }
</style>
""", unsafe_allow_html=True)

    if background_image_uri.startswith("data:image/"):
        background_style = """
<style>
[data-testid="stAppViewContainer"] {
  background-image:
    linear-gradient(90deg, rgba(26, 18, 11, .9), rgba(26, 18, 11, .5)),
    linear-gradient(0deg, rgba(26, 18, 11, .88), rgba(26, 18, 11, .3)),
    url("__BACKGROUND_IMAGE__") !important;
  background-size: cover, cover, cover !important;
  background-position: center, center, center !important;
  background-attachment: fixed !important;
}
</style>
""".replace("__BACKGROUND_IMAGE__", background_image_uri)
        st.markdown(background_style, unsafe_allow_html=True)

    # ── Center the card while keeping a narrow, responsive form ──────────
    _, col, _ = st.columns([1, 1.4, 1], gap="large")

    with col:
        with st.container(key="login_card"):
            st.markdown("""
<div class="login-logo-wrap">
  <div class="login-logo-mark">M</div>
  <div>
    <div class="login-logo-name">MovieMind</div>
    <div class="login-logo-sub">AI-POWERED DISCOVERY</div>
  </div>
</div>
""", unsafe_allow_html=True)

            if auth_state == "password":
                st.markdown('<div class="login-heading">Welcome back</div>', unsafe_allow_html=True)
                st.markdown('<div class="login-sub">Sign in with your MovieMind password</div>', unsafe_allow_html=True)
                with st.form("login_password_form", clear_on_submit=False):
                    email_input = st.text_input("Gmail address", placeholder="you@gmail.com", key="login_password_email")
                    password_input = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password_value")
                    submitted = st.form_submit_button("Sign in →", type="primary", use_container_width=True)
                if submitted:
                    email_clean = email_input.strip().lower()
                    if not _is_gmail(email_clean):
                        st.error("Enter a valid @gmail.com address.")
                    elif not password_input:
                        st.error("Enter your password.")
                    else:
                        try:
                            profile = verify_account_password(email_clean, password_input)
                        except RuntimeError:
                            st.error("The account database is unavailable or not configured. Check the app's Supabase settings.")
                            profile = None
                            database_error = True
                        else:
                            database_error = False
                        if profile:
                            _complete_password_authentication(profile)
                            st.rerun()
                        elif not database_error:
                            st.error("Email or password is incorrect. If you have not set a password yet, use email verification once, then set one in Account settings.")
                if st.button("Use email verification instead", key="switch_to_otp"):
                    st.session_state["_auth_state"] = "email"
                    st.rerun()

            elif auth_state == "password_setup":
                st.markdown('<div class="login-heading">Create your password</div>', unsafe_allow_html=True)
                st.markdown('<div class="login-sub">Email verified. Choose a password for future sign-ins.</div>', unsafe_allow_html=True)
                setup_email = st.session_state.get("_auth_setup_email", "")
                setup_username = st.session_state.get("_auth_setup_username", "")
                st.caption(f"Verified email: {setup_email}")
                with st.form("first_password_setup_form", clear_on_submit=True):
                    new_password = st.text_input("Password", type="password", placeholder="At least 8 characters")
                    confirm_password = st.text_input("Confirm password", type="password")
                    setup_submitted = st.form_submit_button("Save password and continue →", type="primary", use_container_width=True)
                if setup_submitted:
                    if len(new_password) < 8:
                        st.error("Use at least 8 characters for your password.")
                    elif len(new_password) > 128:
                        st.error("Password must be 128 characters or fewer.")
                    elif new_password != confirm_password:
                        st.error("The passwords do not match.")
                    else:
                        try:
                            if _complete_password_setup(setup_email, setup_username, new_password):
                                st.rerun()
                            st.error("Could not save your password. Please try again.")
                        except RuntimeError:
                            st.error("The account database is unavailable or not configured. Check the app's Supabase settings.")

            elif auth_state == "email":
                st.markdown('<div class="login-heading">Welcome back</div>', unsafe_allow_html=True)

                with st.form("login_email_form", clear_on_submit=False):
                    username_input = st.text_input(
                        "Username",
                        placeholder="Enter your username",
                        max_chars=40,
                        key="login_username_field",
                    )
                    email_input = st.text_input(
                        "Gmail address",
                        placeholder="you@gmail.com",
                        key="login_email_field",
                    )
                    submitted = st.form_submit_button(
                        "Continue →",
                        type="primary",
                        use_container_width=True,
                    )

                if submitted:
                    username = normalize_username(username_input)
                    email_clean = email_input.strip().lower()
                    if not username:
                        st.error("Please enter a username using letters, numbers, spaces, or . _ ' - (up to 40 characters).")
                    elif not email_clean:
                        st.error("Please enter your Gmail address.")
                    elif not _is_gmail(email_clean):
                        st.error("Only @gmail.com addresses are supported. Please check your email.")
                    else:
                        try:
                            _start_otp_challenge(email_clean, username)
                        except EmailConfigurationError as exc:
                            st.error(str(exc))
                        except Exception as exc:
                            st.error(_email_send_error(exc))
                        else:
                            st.rerun()

            elif auth_state == "otp":
                email_display = st.session_state.get("_auth_email", "")
                otp_ts = st.session_state.get("_auth_otp_ts", time.time())
                elapsed = time.time() - otp_ts
                remaining = max(0, int(_OTP_TTL_SECONDS - elapsed))

                if remaining == 0:
                    _expire_otp_challenge()
                    st.warning("Your OTP has expired. Please go back and try again.")
                    if st.button("← Back to login"):
                        st.session_state["_auth_state"] = "password"
                        st.rerun()
                else:
                    st.markdown('<div class="login-heading">Verification code</div>', unsafe_allow_html=True)
                    if st.session_state.pop("_auth_email_sent_notice", False):
                        st.success("Brevo accepted the email request. Check your Inbox and Spam folders.")
                    st.caption(f"Sent to {email_display}")

                    _render_otp_countdown()

                    with st.form("login_otp_form", clear_on_submit=True):
                        otp_input = st.text_input(
                            "Verification code",
                            placeholder="Enter OTP",
                            max_chars=6,
                            key="login_otp_field",
                        )
                        verify_btn = st.form_submit_button(
                            "Verify OTP",
                            type="primary",
                            use_container_width=True,
                        )

                    if verify_btn:
                        if _otp_attempts_exhausted():
                            st.error("Too many incorrect attempts. Please restart.")
                            st.rerun()
                        elif otp_input.strip() == st.session_state.get("_auth_otp", ""):
                            username = st.session_state.get("_auth_pending_username", "")
                            try:
                                _complete_otp_authentication(email_display, username)
                            except RuntimeError:
                                st.error("The account database is unavailable or not configured. Check the app's Supabase settings.")
                            else:
                                st.rerun()
                        else:
                            remaining_attempts = _record_failed_otp_attempt()
                            st.error(f"Incorrect OTP. {remaining_attempts} attempts remaining.")

                    col_back, col_resend = st.columns(2)
                    if col_back.button("← Back", use_container_width=True):
                        _clear_otp_challenge(clear_email=True)
                        st.session_state["_auth_state"] = "email"
                        st.rerun()
                    if col_resend.button("Resend OTP", use_container_width=True):
                        otp = _generate_otp()
                        username = st.session_state.get("_auth_pending_username", "")
                        try:
                            _send_otp(email_display, otp, username)
                        except EmailConfigurationError as exc:
                            st.error(str(exc))
                        except Exception as exc:
                            st.error(_email_send_error(exc))
                        else:
                            st.session_state["_auth_otp"] = otp
                            st.session_state["_auth_otp_ts"] = time.time()
                            st.session_state["_otp_attempts"] = 0
                            st.session_state.pop("_otp_demo_value", None)
                            st.session_state["_auth_email_sent_notice"] = True
                            st.rerun()

            st.markdown('<hr class="login-divider">', unsafe_allow_html=True)
            st.markdown(
                '<div class="login-footer">MovieMind · B.Tech CSE Project · ABES Engineering College<br>'
                'Yagyansh Verma · Roll No. 2400320101287</div>',
                unsafe_allow_html=True,
            )
    return False
