import base64
import importlib
import json
import html
import re
import uuid
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import tmdb_helper as _tmdb_helper
# Streamlit can retain imported helper modules between script reruns. Reload the
# helper if a public API used below is missing from that cached module object.
_required_tmdb_exports = (
    "get_tmdb_similar_movies",
    "get_tmdb_genre_movies",
    "get_movie_poster_url",
    "preload_movie_media",
)
if any(not hasattr(_tmdb_helper, name) for name in _required_tmdb_exports):
    _tmdb_helper = importlib.reload(_tmdb_helper)
try:
    from gemini_helper import parse_movie_intent, transcribe_speech_to_text
except Exception:
    parse_movie_intent = None
    transcribe_speech_to_text = None
from tmdb_helper import (
    POSTER_PLACEHOLDER,
    get_movie_backdrop_image,
    get_movie_details,
    get_movie_poster_image,
    get_movie_trailer,
    get_movie_poster_url,
    get_discover_movies,
    preload_movie_media,
    get_tmdb_genre_movies,
    get_tmdb_similar_movies,
)
from intent_match import get_intent_match_details
from watch_later import (
    DEFAULT_WATCH_LATER_PROFILE_ID,
    toggle_watch_later_entry,
    watch_later_key,
    watch_later_movie_entry,
)
from movie_details import select_movie_details
from carousel_component import render_circular_carousel
from login import clear_auth_session, render_login_page
from account_profile import (
    load_profiles as load_profile_records,
    save_profiles as save_profile_records,
    set_account_password,
)


from scipy.sparse import csr_matrix, lil_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from surprise import Dataset, Reader, SVD

PROFILE_PATH = Path(__file__).resolve().parent / "data" / "user_profiles.json"
CONTEXT_WEIGHT = 0.15


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MovieMind",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# AUTHENTICATION GATE
# ============================================================

# Use a real TMDB backdrop behind the sign-in card when credentials and image
# data are available. The login page keeps its branded gradient as fallback.
login_background_uri = ""
if st.session_state.get("_auth_state") != "done":
    try:
        login_backdrops = get_discover_movies("Trending")
        if login_backdrops and login_backdrops[0].get("backdrop_path"):
            login_backdrop = get_movie_backdrop_image(login_backdrops[0]["backdrop_path"])
            if login_backdrop:
                login_background_uri = "data:image/jpeg;base64," + base64.b64encode(login_backdrop).decode("ascii")
    except Exception:
        login_background_uri = ""

# The login gate also upgrades older OTP-only sessions into password setup.
st.session_state["_login_background_uri"] = login_background_uri
if not render_login_page():
    st.stop()

st.session_state.setdefault("ui_theme", "dark")


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>
.block-container { max-width: 1500px; padding-top: 1.3rem; padding-bottom: 3rem; scroll-behavior: smooth; }
[data-testid="stAppViewContainer"] { background: radial-gradient(ellipse at 76% -18%, #202a3b 0%, #0b0e13 45%, #090b0f 100%); }
section[data-testid="stSidebar"] { width: 260px !important; min-width: 260px !important; background: linear-gradient(165deg, #0f1520 0%, #0a0e16 55%, #07090d 100%); border-right: 1px solid rgba(255,255,255,.06); backdrop-filter: blur(20px); }
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 0 0 1.5rem; overflow-x: hidden; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #9aa4b3; }
.sb-header { padding: 20px 16px 16px; border-bottom: 1px solid rgba(255,255,255,.06); margin-bottom: 8px; position: relative; }
.sb-header::after { content:''; position:absolute; bottom:0; left:16px; right:16px; height:1px; background: linear-gradient(90deg,transparent,rgba(224,166,109,.3),transparent); }
.sb-logo { display:flex; align-items:center; gap:11px; margin-bottom: 16px; }
.sb-logo-mark { width:38px;height:38px;border-radius:12px;background:linear-gradient(145deg,#e0a66d,#b76575);display:grid;place-items:center;font-size:18px;font-weight:900;color:#13111a;box-shadow:0 6px 20px rgba(183,101,117,.35),0 0 0 1px rgba(224,166,109,.2); }
.sb-logo-text { display:flex;flex-direction:column; }
.sb-logo-name { font-size:18px;font-weight:780;letter-spacing:-.5px;color:#f5f7fb;line-height:1.1; }
.sb-logo-caption { font-size:10px;letter-spacing:1.4px;text-transform:uppercase;color:#7a8698;margin-top:3px; }
.sb-user-card { background: linear-gradient(135deg, rgba(224,166,109,.1), rgba(183,101,117,.06)); border: 1px solid rgba(224,166,109,.18); border-radius: 12px; padding: 10px 12px; display:flex; align-items:center; gap:10px; }
.sb-user-avatar { width:32px;height:32px;border-radius:50%;background:linear-gradient(145deg,#8b5e9e,#4a6eb5);display:grid;place-items:center;font-size:13px;font-weight:700;color:#fff;flex-shrink:0;border:2px solid rgba(224,166,109,.3); }
.sb-user-name { font-size:13px;font-weight:700;color:#f0f2f6;line-height:1.2; }
.sb-user-email { font-size:11px;color:#7a8698;margin-top:1px; }
.sb-user-badge { display:inline-block;background:rgba(132,200,173,.15);color:#84c8ad;border:1px solid rgba(132,200,173,.3);border-radius:999px;font-size:9px;font-weight:800;letter-spacing:.6px;padding:2px 6px;margin-top:3px;text-transform:uppercase; }
.sb-section { padding: 0 10px; margin-top: 6px; }
.sb-section-label { font-size:9px;font-weight:800;letter-spacing:2px;text-transform:uppercase;color:#4e5a6b;padding: 10px 6px 4px; }
section[data-testid="stSidebar"] .stButton > button { width:100%;justify-content:flex-start;text-align:left;border:none;background:transparent;padding:7px 10px;min-height:40px;color:#8fa0b5;font-size:13px;font-weight:600;border-radius:10px;transition:all .15s ease;position:relative; }
section[data-testid="stSidebar"] .stButton > button:hover { background:rgba(255,255,255,.06);color:#e8edf4; }
section[data-testid="stSidebar"] .stButton > button[kind="primary"] { background:linear-gradient(90deg,rgba(224,166,109,.12),rgba(183,101,117,.06))!important;color:#f5d0a0!important;box-shadow:inset 3px 0 0 #d6a36b; }
.sidebar-profile { margin-top:17px;padding:13px 12px 10px;border:1px solid #ffffff12;border-radius:13px;background:linear-gradient(145deg,#ffffff08,#ffffff03); }
.sidebar-profile-title { color:#7e8a9a;font-size:9px;font-weight:800;letter-spacing:1.4px;margin-bottom:8px; }
.sidebar-profile-name { color:#f4f6f9;font-size:14px;font-weight:700; }
.sidebar-profile-meta { color:#9aa5b4;font-size:11px;margin:3px 0 10px; }
.brand-bar { display:flex;align-items:center;gap:11px;margin:0 0 17px;color:#f5f7fb; }
.brand-mark { width:32px;height:32px;border-radius:10px;display:grid;place-items:center;background:linear-gradient(145deg,#dca66d,#b86479);color:#161216;font-size:16px;font-weight:900; }
.brand-name { font-size:20px;font-weight:760;letter-spacing:-.65px;line-height:1.1; }
.brand-sub { color:#929eaf;font-size:12px;margin-top:3px; }
.brand-ai { margin-left:auto;color:#c9d2dc;font-size:10px;letter-spacing:.25px;padding:6px 9px;border:1px solid #ffffff17;border-radius:999px;background:#ffffff08;white-space:nowrap; }
.brand-ai-dot { color:#84c8ad;margin-right:5px; }
.ai-panel { padding:20px 22px 10px;border:1px solid #ffffff17;border-bottom:0;border-radius:18px 18px 0 0;background:linear-gradient(120deg,rgba(24,31,43,.96),rgba(16,20,28,.95));box-shadow:0 14px 40px #0004;margin:0; }
.ai-kicker { color:#d9a978;text-transform:uppercase;letter-spacing:1.55px;font-size:10px;font-weight:800; }
.ai-heading { color:#f7f8fb;font-size:24px;font-weight:720;letter-spacing:-.55px;margin:5px 0; }
.ai-sub { color:#aeb7c4;font-size:13px;line-height:1.5;margin-bottom:2px; }
div[data-testid="stForm"] { padding:5px 20px 4px;border:1px solid #ffffff17;border-top:0;border-bottom:0;border-radius:0;background:linear-gradient(120deg,rgba(24,31,43,.96),rgba(16,20,28,.95)); }
div[data-testid="stForm"] input { min-height:52px!important;border:1px solid #ffffff20!important;border-radius:13px!important;background:#0e131b!important;color:#f4f6f9!important;padding:0 16px!important;font-size:15px!important;transition:border-color .16s ease,box-shadow .16s ease!important; }
div[data-testid="stForm"] input:focus { border-color:#c9956a!important;box-shadow:0 0 0 3px #d6a36b1a!important; }
div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button { min-height:50px;border-radius:12px;background:linear-gradient(110deg,#d3a06c,#bd7182);border:0;color:#171316;font-weight:750;transition:filter .16s ease,transform .16s ease; }
div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover { filter:brightness(1.08);transform:translateY(-1px); }
.ai-examples { color:#8f9aab;font-size:11px;line-height:1.8;padding:2px 22px 13px;border:1px solid #ffffff17;border-top:0;border-radius:0 0 18px 18px;background:linear-gradient(120deg,rgba(24,31,43,.96),rgba(16,20,28,.95));margin:0 0 12px; }
.ai-example-label { color:#c5ced8;font-weight:700;letter-spacing:.4px;margin-right:5px; }
.ai-status { margin:8px 0 15px;padding:11px 14px;border:1px solid #ffffff14;border-radius:12px;background:#111720;color:#cbd3dd;font-size:12px; }
.ai-status-label { display:block;color:#8190a1;font-size:9px;font-weight:800;letter-spacing:1.3px;margin-bottom:5px; }
.ai-error-state { margin:8px 0 15px;padding:11px 14px;border:1px solid #c5756c50;border-radius:12px;background:#492b2a2e;color:#edc9c4;font-size:12px; }
.ai-429-state { border-color:#d4a15e55;background:#57402438;color:#f2d7b3; }
.manual-refine-label { color:#909bac;font-size:10px;font-weight:800;letter-spacing:1.4px;text-transform:uppercase; }
.sidebar-profile div[data-testid="stSelectbox"] label { display:none; }
.cinema-hero { min-height:min(72vh,760px);position:relative;overflow:hidden;border-radius:10px;margin:0 0 28px;padding:clamp(32px,6vw,86px);display:flex;align-items:center;background-color:#13192a;background-position:center 38%;background-size:cover;border:1px solid #ffffff18;box-shadow:0 22px 65px #0007; }
.cinema-hero::before { content:'';position:absolute;inset:0;background:linear-gradient(90deg,rgba(5,7,11,.97) 0%,rgba(5,7,11,.82) 39%,rgba(5,7,11,.16) 100%),linear-gradient(0deg,rgba(5,7,11,.82),transparent 65%);z-index:0; }
.cinema-copy { max-width:720px;position:relative;z-index:1; }
.cinema-kicker { font-size:11px;text-transform:uppercase;letter-spacing:1.5px;font-weight:800;color:#ffbd79; }
.cinema-title { color:#fff;font-size:clamp(38px,5.2vw,72px);line-height:1.02;letter-spacing:-1.8px;font-weight:820;margin:10px 0; }
.cinema-meta { color:#e5e8ed;font-size:13px;font-weight:650;margin:10px 0; }
.cinema-overview { color:#d2d6df;font-size:14px;line-height:1.55;max-width:580px; }
.cinema-why { color:#f0c7a1;font-size:13px;margin-top:13px; }
.match-pill { display:inline-block;padding:7px 11px;border-radius:999px;background:#ffffff1c;border:1px solid #ffffff2b;color:#fff;font-size:13px;font-weight:750;backdrop-filter:blur(8px); }
.hero-signal-chip { display:inline-block;padding:6px 11px;border-radius:999px;background:rgba(224,166,109,0.18);border:1px solid rgba(224,166,109,0.4);color:#ffc686;font-size:12px;font-weight:750; }
.movie-card { background:linear-gradient(160deg,#171c25,#101319);border:1px solid #ffffff0e;border-radius:14px;overflow:hidden;transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease;margin-bottom:8px;position:relative; }
.movie-card:hover { transform:translateY(-5px);border-color:#eab5738c;box-shadow:0 14px 32px #0009, 0 0 16px rgba(224,166,109,0.15); }
.card-poster-wrap { position:relative;width:100%;aspect-ratio:2/3;background:#131720;overflow:hidden; }
.movie-poster { width:100%;height:100%;object-fit:cover;display:block;transition:transform .3s ease; }
.movie-card:hover .movie-poster { transform:scale(1.03); }
.card-poster-gradient { position:absolute;bottom:0;left:0;right:0;height:45px;background:linear-gradient(0deg,rgba(16,19,25,0.95) 0%,rgba(16,19,25,0) 100%); }
.card-badge { position:absolute;top:8px;left:8px;padding:3px 7px;border-radius:999px;font-size:10px;font-weight:800;background:rgba(12,16,22,0.85);color:#ffc686;border:1px solid rgba(224,166,109,0.5);backdrop-filter:blur(6px); }
.card-saved-badge { position:absolute;top:8px;right:8px;padding:3px 7px;border-radius:999px;font-size:9px;font-weight:750;background:rgba(132,200,173,0.22);color:#84c8ad;border:1px solid rgba(132,200,173,0.45);backdrop-filter:blur(6px); }
.movie-card-info { padding:10px 11px 11px; }
.movie-card-title { color:#f3f5f8;font-size:15px;font-weight:730;line-height:1.3;min-height:38px; }
.movie-card-meta { color:#9ca6b5;font-size:12px;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }
.movie-card-match { color:#ffc27e;font-size:12px;font-weight:750;margin-top:8px; }
.movie-card-intent { color:#aeb9c9;font-size:11px;line-height:1.4;margin-top:4px; }
.row-heading { color:#f4f5f8;font-size:22px;font-weight:760;letter-spacing:-.35px;margin:28px 0 2px; }
.row-caption { color:#8994a4;font-size:13px;margin-bottom:10px; }
.intent-chip { display:inline-block;background:#ffffff10;border:1px solid #ffffff18;color:#dce2eb;border-radius:999px;padding:5px 10px;margin:3px 5px 3px 0;font-size:11px; }
.voice-input-card { background:linear-gradient(135deg,rgba(224,166,109,0.08),rgba(183,101,117,0.06));border:1px solid rgba(224,166,109,0.2);border-radius:14px;padding:14px 18px;margin:10px 0 14px; }
.voice-input-title { color:#ffc686;font-size:14px;font-weight:750;display:flex;align-items:center;gap:6px; }
.voice-input-sub { color:#a6b2c2;font-size:12px;margin-top:4px; }
.empty-state-box { text-align:center;padding:48px 24px;border:1px dashed rgba(255,255,255,0.12);border-radius:18px;background:rgba(18,23,32,0.6);margin:20px 0; }
.empty-state-icon { font-size:38px;margin-bottom:10px; }
.empty-state-title { font-size:18px;font-weight:750;color:#f3f5f8;margin-bottom:6px; }
.empty-state-sub { font-size:13px;color:#8a96a7;max-width:500px;margin:0 auto;line-height:1.5; }
.sidebar-student-info { margin-top:14px;padding:12px 11px;border-radius:12px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);font-size:12px;color:#9aa5b4;line-height:1.6; }
.brand-ai { font-size:11px; }
.card-badge { font-size:11px; }
[data-testid="stWidgetLabel"] p { font-size:14px; }
[data-testid="stCaptionContainer"] p { font-size:13px;line-height:1.5; }
.section-gap {height:10px;}
.stButton > button { border-radius:10px;min-height:38px;font-weight:650;border-color:#ffffff1c;background:#161b24;color:#e9edf3;transition:all .16s ease; }
@media (max-width: 768px) {
  [class*="st-key-movie-row-"] [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important; }
  [class*="st-key-movie-row-"] [data-testid="stHorizontalBlock"] > [data-testid="column"] { flex:1 1 calc(50% - 0.5rem) !important; width:calc(50% - 0.5rem) !important; min-width:0 !important; }
  [class*="st-key-movie-actions-"] [data-testid="stHorizontalBlock"] { flex-direction:column !important; gap:0.25rem !important; }
  [class*="st-key-movie-actions-"] [data-testid="stHorizontalBlock"] > [data-testid="column"] { flex:1 1 100% !important; width:100% !important; }
}
.stButton > button:hover { border-color:#e4ac6d;background:#222733;color:#fff; }
.stButton > button[kind="primary"] { background:linear-gradient(110deg,#d94b63,#9b4e9b);border:0;color:white; }
button[data-baseweb="tab"] { font-weight:650;color:#aab3c0; }
button[data-baseweb="tab"][aria-selected="true"] { color:#fff; }
div[data-testid="stExpander"] { border-radius:13px;border-color:#ffffff15;background:#10141b; }
div[data-testid="stMetric"] { background:#121720;border-radius:12px;padding:10px; }
.footer-text { color:#687385;font-size:12px;text-align:center;margin-top:22px; }
button, input, [data-testid="stSelectbox"], [data-testid="stMultiSelect"] { scroll-margin-top: 1rem; }
button:focus-visible, input:focus-visible { outline:2px solid #e0a66d!important;outline-offset:2px!important; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior:auto!important;animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important; }
}
@media (max-width: 760px) { .block-container{padding-top:1rem}.cinema-hero{min-height:56vh;padding:25px 22px;background-position:62% center}.ai-panel{padding:17px 16px 8px}.ai-heading{font-size:21px}.ai-examples{padding-left:16px;padding-right:16px}.brand-sub{display:none}.row-heading{font-size:18px}section[data-testid="stSidebar"]{width:min(250px,82vw)!important;min-width:min(250px,82vw)!important} }

/* MovieMind's streaming-inspired visual system: deep charcoal, cinematic red, and clear type. */
[data-testid="stAppViewContainer"] { background:radial-gradient(ellipse at 72% -12%,#40141b 0%,#160d12 32%,#09090b 72%)!important; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#17090d 0%,#0b0a0c 60%,#080809 100%)!important;border-right:1px solid #ffffff12; }
.sb-header::after { background:linear-gradient(90deg,transparent,#e50914aa,transparent); }
.sb-logo-mark,.brand-mark { background:#e50914!important;color:#fff!important;box-shadow:0 6px 20px #e5091440!important;border-radius:8px!important; }
.sb-logo-name,.brand-name { color:#fff!important; }
.sb-logo-caption { color:#ef7c82!important; }
section[data-testid="stSidebar"] .stButton > button[kind="primary"] { background:linear-gradient(90deg,#e50914,#b80710)!important;color:#fff!important;box-shadow:inset 3px 0 0 #ff535a; }
.brand-bar { padding:8px 0 18px;border-bottom:1px solid #ffffff16;margin-bottom:20px; }
.brand-mark { width:38px;height:38px;font-size:19px; }
.brand-name { font-size:24px;letter-spacing:.6px;text-transform:uppercase; }
.brand-sub { color:#c0b7bb; }
.brand-ai { border-color:#e5091460;background:#e5091412;color:#f1d8db; }
.brand-ai-dot { color:#ff515a; }
.ai-panel { min-height:260px;padding:34px 36px 24px;border:1px solid #ffffff17;border-radius:14px 14px 0 0;background:linear-gradient(115deg,rgba(43,12,17,.97),rgba(15,12,15,.91)),radial-gradient(ellipse at 85% 20%,#9d142255,transparent 45%);box-shadow:0 24px 60px #0008;display:flex;flex-direction:column;justify-content:center;position:relative;isolation:isolate;overflow:hidden; }
.hero-poster-mosaic { position:absolute;inset:-15% -5%;display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;opacity:.36;transform:rotate(-5deg) scale(1.1);z-index:-2;pointer-events:none; }
.hero-poster-mosaic img { width:100%;height:100%;min-height:280px;object-fit:cover;border-radius:6px;filter:saturate(.72); }
.ai-panel::before { content:"";position:absolute;inset:0;background:linear-gradient(90deg,rgba(11,8,10,.97) 0%,rgba(15,9,12,.88) 42%,rgba(13,8,10,.56) 100%),linear-gradient(0deg,rgba(8,7,9,.75),transparent 70%);z-index:-1; }
.hero-content { position:relative;z-index:1; }
.ai-kicker { color:#ff5961;font-size:11px;letter-spacing:2px; }
.ai-heading { font-size:clamp(30px,4vw,46px);line-height:1.07;letter-spacing:-1.2px;font-weight:820;max-width:780px;margin:10px 0; }
.ai-sub { color:#d3c8cc;font-size:15px;max-width:800px; }
.ai-examples { border-radius:0 0 14px 14px;background:#120e11;border-color:#ffffff17;color:#b9afb3;font-size:12px;padding:10px 36px 15px; }
.ai-status { background:#171014;border-color:#e5091440; }
.ai-status-label { color:#ff6a70; }
.intent-chip { background:#e5091414;border-color:#e509144f;color:#ffe4e5; }
.cinema-hero { border-radius:14px;border-color:#ffffff20;box-shadow:0 24px 70px #0009; }
.cinema-hero::before { background:linear-gradient(90deg,rgba(5,5,7,.97) 0%,rgba(5,5,7,.75) 42%,rgba(5,5,7,.12) 100%),linear-gradient(0deg,rgba(5,5,7,.82),transparent 65%); }
.cinema-kicker,.cinema-why { color:#ff777c; }
.match-pill,.hero-signal-chip { background:#e50914d9;border-color:#ff646b;color:#fff; }
.row-heading { color:#fff;font-size:24px;letter-spacing:-.3px; }
.row-caption { color:#b6aeb2; }
.movie-card { background:linear-gradient(160deg,#1c171a,#100e10);border-color:#ffffff18;border-radius:8px; }
.movie-card:hover { border-color:#e50914aa;box-shadow:0 14px 32px #000a,0 0 16px #e5091425; }
.card-badge { background:#e50914;color:#fff;border-color:#ff5961; }
.stButton > button { border-radius:6px;min-height:42px;border-color:#ffffff25;background:#191417;color:#f4eff0; }
.stButton > button:hover { border-color:#ff535a;background:#2b171b;color:#fff; }
.stButton > button[kind="primary"] { background:#e50914!important;border-color:#e50914!important;color:#fff!important; }
.stButton > button[kind="primary"]:hover { background:#c11119!important;filter:brightness(1.08); }
button[data-baseweb="tab"] { color:#b9afb3; }
button[data-baseweb="tab"][aria-selected="true"] { color:#fff;border-bottom-color:#e50914!important; }
div[data-testid="stExpander"] { background:#130f12;border-color:#ffffff20;border-radius:8px; }
div[data-testid="stMetric"] { background:#171114;border:1px solid #ffffff12;border-radius:8px; }
input,textarea,[data-baseweb="select"] > div { background-color:#191518!important;border-color:#ffffff3b!important;color:#fff!important;border-radius:5px!important; }
input:focus,textarea:focus { border-color:#ff4b55!important;box-shadow:0 0 0 2px #e509142c!important; }
@media (max-width:760px) {
  .brand-bar { padding:4px 0 13px; }
  .brand-name { font-size:20px; }
  .brand-ai { font-size:9px;padding:5px 7px; }
  .ai-panel { min-height:220px;padding:24px 18px 14px; }
  .hero-poster-mosaic { grid-template-columns:repeat(3,minmax(0,1fr));inset:-10% -18%;gap:6px;opacity:.26; }
  .hero-poster-mosaic img { min-height:230px; }
  .ai-heading { font-size:30px; }
  .ai-sub { font-size:13px; }
  .ai-examples { padding:9px 18px 12px; }
  .row-heading { font-size:20px; }
}
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display:none!important; }
[data-testid="stMainBlockContainer"] { max-width:1600px!important;padding-top:0!important; }
.block-container { max-width:1600px!important;padding-top:.45rem!important;padding-left:clamp(14px,3vw,52px)!important;padding-right:clamp(14px,3vw,52px)!important; }
.st-key-streaming-navbar { position:sticky;top:calc(var(--header-height,3.75rem) + 8px);z-index:1000;background:rgba(20,15,12,.98);border:1px solid rgba(242,204,143,.38);border-radius:10px;padding:8px 12px 6px;margin-top:calc(var(--header-height,3.75rem) + 8px);margin-bottom:16px;box-shadow:0 10px 28px rgba(0,0,0,.42),0 2px 12px rgba(224,122,95,.12);backdrop-filter:blur(14px); }
.streaming-brand { height:42px;display:flex;align-items:center;gap:8px;color:#f6121d;font-size:18px;font-weight:900;letter-spacing:.8px;white-space:nowrap; }
.streaming-brand-mark { display:grid;place-items:center;width:31px;height:31px;border-radius:5px;background:#e50914;color:#fff;font-size:17px; }
.st-key-streaming-navbar .stButton > button { min-height:40px;padding:0 7px;background:transparent!important;border:0!important;border-radius:0!important;color:#d3cdd0!important;font-size:13px;font-weight:600;white-space:nowrap; }
.st-key-streaming-navbar .stButton > button:hover { color:#fff!important;background:#ffffff10!important; }
.st-key-streaming-navbar .stButton > button[kind="primary"] { color:#fff!important;box-shadow:inset 0 -2px #e50914!important; }
.st-key-streaming-navbar [data-testid="stTextInput"] input { min-height:38px!important;padding:0 12px!important;background:#171416!important;border:1px solid #ffffff30!important;border-radius:5px!important;color:#fff!important;font-size:13px!important; }
.st-key-streaming-navbar [data-testid="stTextInput"] input:focus { border-color:#e50914!important;box-shadow:0 0 0 2px #e5091425!important; }
.st-key-ai_search_submit > div > button { min-height:38px!important;width:40px!important;padding:0!important;background:#e50914!important;border:0!important;border-radius:5px!important;color:#fff!important;font-size:21px!important; }
.st-key-ai_search_submit > div > button:hover { background:#c11119!important; }
.st-key-streaming-navbar [data-testid="stPopover"] > button { background:#171214!important;border:1px solid #ffffff22!important;border-radius:6px!important;color:#fff!important;min-height:38px; }
.st-key-streaming-navbar [data-testid="stPopover"] > button:hover { border-color:#e50914!important; }
.st-key-home-category-bar { margin:-8px 0 18px;padding:0 2px 8px;border-bottom:1px solid #ffffff16; }
.st-key-home-category-bar .stButton > button { min-height:36px;padding:0 14px;background:transparent!important;border:0!important;border-radius:0!important;color:#c8c2c5!important;font-size:13px;font-weight:650;white-space:nowrap; }
.st-key-home-category-bar .stButton > button:hover { color:#fff!important;background:#ffffff0d!important; }
.st-key-home-category-bar .stButton > button[kind="primary"] { color:#fff!important;box-shadow:inset 0 -2px #e50914!important; }
.st-key-discover_genre [role="radiogroup"] { gap:clamp(10px,2vw,28px);flex-wrap:nowrap;overflow-x:auto;padding:2px 0 8px;scrollbar-width:none; }
.st-key-discover_genre [role="radiogroup"]::-webkit-scrollbar { display:none; }
.st-key-discover_genre [role="radiogroup"] label { white-space:nowrap;padding:5px 0 9px;border-bottom:2px solid transparent;color:#c9c4c4;font-weight:600; }
.st-key-discover_genre [role="radiogroup"] label:has(input:checked) { color:#fff;border-bottom-color:#e50914; }
.discover-divider { height:1px;background:#ffffff1c;margin:3px 0 20px; }
.discover-page-count { text-align:center;padding:9px 2px;color:#b8b3b6;white-space:nowrap; }
.st-key-movie-rail- [data-testid="stHorizontalBlock"] { flex-wrap:nowrap!important;overflow-x:auto!important;overflow-y:hidden!important;gap:12px!important;padding:2px 2px 14px!important;scroll-snap-type:x mandatory;scrollbar-width:thin;scrollbar-color:#5b3338 transparent; }
.st-key-movie-rail- [data-testid="stHorizontalBlock"] > [data-testid="column"] { flex:0 0 224px!important;width:224px!important;min-width:224px!important;scroll-snap-align:start; }
.movie-card-wide .card-poster-wrap { aspect-ratio:16/9; }
.movie-card-wide .movie-card-info { padding:8px 10px 10px; }
.movie-card-wide .movie-card-title { min-height:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:14px; }
.movie-card-wide .movie-card-meta { margin-top:3px; }
.streaming-feature { position:relative;min-height:390px;display:flex;align-items:flex-end;overflow:hidden;margin:18px 0 4px;padding:34px 38px;border-radius:10px;background-color:#171214;background-size:cover;background-position:center 35%;border:1px solid #ffffff14;box-shadow:0 20px 55px #0009; }
.streaming-feature::before { content:"";position:absolute;inset:0;background:linear-gradient(90deg,rgba(7,7,8,.96) 0%,rgba(7,7,8,.77) 38%,rgba(7,7,8,.05) 100%),linear-gradient(0deg,rgba(7,7,8,.8),transparent 70%); }
.streaming-feature-copy { position:relative;z-index:1;max-width:650px; }
.streaming-feature-kicker { color:#ff6067;font-size:11px;font-weight:800;letter-spacing:1.8px; }
.streaming-feature-title { color:#fff;font-size:clamp(32px,4vw,54px);line-height:1.05;font-weight:850;letter-spacing:-1px;margin:8px 0; }
.streaming-feature-meta { color:#e9dfe2;font-size:14px;font-weight:650;margin-bottom:10px; }
.streaming-feature-overview { color:#e0d9db;font-size:14px;line-height:1.5;max-width:560px;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden; }
@media (max-width:760px) {
  .st-key-streaming-navbar { overflow-x:auto; }
  .st-key-streaming-navbar [data-testid="stHorizontalBlock"] { min-width:1040px; }
  .st-key-movie-rail- [data-testid="stHorizontalBlock"] > [data-testid="column"] { flex-basis:176px!important;width:176px!important;min-width:176px!important; }
  .streaming-feature { min-height:300px;padding:22px 18px;background-position:60% center; }
  .streaming-feature-title { font-size:32px; }
  .streaming-feature-overview { font-size:13px; }
}

/* Retro 35mm palette */
:root { --retro-espresso:#1A120B; --retro-leather:#3C2A21; --retro-parchment:#F5EBE6; --retro-terracotta:#E07A5F; --retro-gold:#F2CC8F; }
[data-testid="stAppViewContainer"] { background:radial-gradient(ellipse at 74% -12%,#3C2A21 0%,#241910 36%,#1A120B 78%)!important;color:var(--retro-parchment)!important; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#2c2018 0%,#21170f 58%,#1A120B 100%)!important;border-right:1px solid #F2CC8F25!important; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,[data-testid="stCaptionContainer"] p { color:#d7c4b9!important; }
.streaming-brand,.sb-logo-name,.brand-name { color:var(--retro-gold)!important; }
.streaming-brand-mark,.sb-logo-mark,.brand-mark { background:var(--retro-terracotta)!important;color:var(--retro-espresso)!important;box-shadow:0 5px 20px #e07a5f45!important; }
.streaming-navbar,.st-key-streaming-navbar { background:rgba(26,18,11,.98)!important;border-color:#F2CC8F68!important;box-shadow:0 10px 30px #100a06c7,0 0 18px #E07A5F20!important; }
.st-key-streaming-navbar .stButton > button,.st-key-home-category-bar .stButton > button { color:#e8d8cf!important; }
.st-key-streaming-navbar .stButton > button[kind="primary"],.st-key-home-category-bar .stButton > button[kind="primary"] { color:var(--retro-gold)!important;box-shadow:inset 0 -2px var(--retro-terracotta)!important; }
.st-key-streaming-navbar [data-testid="stTextInput"] input,.st-key-streaming-navbar [data-testid="stPopover"] > button { background:#302219!important;border-color:#F2CC8F40!important;color:var(--retro-parchment)!important; }
.ai-panel,.ai-examples,div[data-testid="stForm"] { background:linear-gradient(125deg,rgba(60,42,33,.97),rgba(37,26,19,.98))!important;border-color:#F2CC8F30!important; }
.ai-heading,.cinema-title,.streaming-feature-title,.row-heading,.movie-card-title,.empty-state-title { color:var(--retro-parchment)!important; }
.ai-kicker,.cinema-kicker,.streaming-feature-kicker,.ai-status-label { color:var(--retro-gold)!important; }
.cinema-hero,.streaming-feature { border-color:#F2CC8F38!important;box-shadow:0 24px 70px #100a06aa!important; }
.cinema-hero::before,.streaming-feature::before { background:linear-gradient(90deg,rgba(26,18,11,.96) 0%,rgba(26,18,11,.76) 42%,rgba(26,18,11,.12) 100%),linear-gradient(0deg,rgba(26,18,11,.84),transparent 68%)!important; }
.cinema-kicker,.cinema-why,.movie-card-match,.card-badge { color:var(--retro-gold)!important; }
.match-pill,.hero-signal-chip,.card-badge { background:#3C2A21e8!important;border-color:#F2CC8F88!important; }
.movie-card,div[data-testid="stMetric"],div[data-testid="stExpander"] { background:linear-gradient(160deg,#3C2A21,#2b1e17)!important;border-color:#F2CC8F2b!important; }
.movie-card:hover { border-color:#E07A5Fbb!important;box-shadow:0 14px 32px #100a06aa,0 0 16px #E07A5F30!important; }
.card-poster-wrap { background:#291d16!important; }
.card-poster-gradient { background:linear-gradient(0deg,rgba(26,18,11,.96),transparent)!important; }
.row-caption,.movie-card-meta,.movie-card-intent,.empty-state-sub { color:#cbb8ad!important; }
.intent-chip,.ai-status,.voice-input-card,.empty-state-box { background:#3C2A213b!important;border-color:#F2CC8F40!important;color:var(--retro-parchment)!important; }
.stButton > button { background:#3C2A21!important;border-color:#F2CC8F40!important;color:var(--retro-parchment)!important; }
.stButton > button:hover { background:#52382b!important;border-color:var(--retro-terracotta)!important;color:#fff!important; }
.stButton > button[kind="primary"],div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button { background:var(--retro-terracotta)!important;border-color:var(--retro-terracotta)!important;color:var(--retro-espresso)!important; }
.stButton > button[kind="primary"]:hover,div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover { background:#ed9276!important; }
input,textarea,[data-baseweb="select"] > div,div[data-testid="stMetric"] { background-color:#3C2A21!important;border-color:#F2CC8F45!important;color:var(--retro-parchment)!important; }
input:focus,textarea:focus { border-color:var(--retro-gold)!important;box-shadow:0 0 0 2px #F2CC8F35!important; }
button:focus-visible,input:focus-visible { outline:2px solid var(--retro-gold)!important; }
.footer-text { color:#bda89b!important; }

</style>

""",
    unsafe_allow_html=True
)

if st.session_state.get("ui_theme") == "light":
    st.markdown(
        """
<style>
:root { --retro-espresso:#f4f1ee; --retro-leather:#ffffff; --retro-parchment:#27211d; --retro-terracotta:#bd5d43; --retro-gold:#885a26; }
[data-testid="stAppViewContainer"],[data-testid="stMainBlockContainer"] { background:#f4f1ee!important;color:#27211d!important; }
[data-testid="stHeader"] { background:rgba(244,241,238,.96)!important; }
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,[data-testid="stMarkdownContainer"] h1,[data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3,[data-testid="stMarkdownContainer"] h4,[data-testid="stWidgetLabel"] p { color:#302720!important; }
[data-testid="stCaptionContainer"] p,.row-caption,.footer-text { color:#74685f!important; }
.st-key-streaming-navbar,.streaming-navbar,.st-key-home-category-bar { background:#fff!important;border-color:#e6ddd5!important; }
[data-baseweb="popover"],[data-testid="stPopoverBody"] { background:#fff!important;color:#302720!important;border-color:#e3d9d1!important; }
[data-baseweb="popover"] [data-testid="stMarkdownContainer"] p { color:#74685f!important; }
.streaming-brand,.sb-logo-name,.brand-name { color:#8a4b2e!important; }
.st-key-streaming-navbar .stButton > button,.st-key-home-category-bar .stButton > button { color:#453a33!important; }
.st-key-streaming-navbar .stButton > button[kind="primary"],.st-key-home-category-bar .stButton > button[kind="primary"] { color:#8a4b2e!important;box-shadow:inset 0 -2px #bd5d43!important; }
.st-key-streaming-navbar [data-testid="stTextInput"] input,.st-key-streaming-navbar [data-testid="stPopover"] > button { background:#f7f4f1!important;border-color:#ded3ca!important;color:#302720!important; }
.ai-panel,.ai-examples,div[data-testid="stForm"] { background:#fff!important;border-color:#e5dcd5!important; }
div[data-testid="stForm"] input { background:#fff!important;color:#302720!important;border-color:#ddd1c8!important; }
.ai-heading,.row-heading,.movie-card-title,.empty-state-title,.streaming-feature-title,.cinema-title { color:#302720!important; }
.ai-kicker,.cinema-kicker,.streaming-feature-kicker,.ai-status-label { color:#885a26!important; }
.ai-status,.intent-chip,.voice-input-card,.empty-state-box { background:#fff!important;border-color:#e6ddd5!important;color:#40352e!important; }
div[data-testid="stExpander"] { background:#fff!important;border-color:#e3d9d1!important; }
.stButton > button { background:#fff!important;border-color:#ded3ca!important;color:#392f29!important; }
.stButton > button:hover { background:#f5ede7!important;border-color:#bd5d43!important;color:#34251e!important; }
.stButton > button[kind="primary"],div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button { background:#bd5d43!important;border-color:#bd5d43!important;color:#fff!important; }
input,textarea,[data-baseweb="select"] > div,div[data-testid="stMetric"] { background:#fff!important;border-color:#ded3ca!important;color:#302720!important; }
[data-testid="stRadio"] label,[data-testid="stCheckbox"] label { color:#40352e!important; }
.st-key-discover_genre [role="radiogroup"] label { color:#6b5e55!important; }
.st-key-discover_genre [role="radiogroup"] label:has(input:checked) { color:#302720!important;border-bottom-color:#bd5d43!important; }
.discover-divider { background:#ded5ce!important; }
.discover-page-count { color:#74685f!important; }
button:focus-visible,input:focus-visible { outline:2px solid #bd5d43!important; }
</style>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():

    ratings = pd.read_csv(
        "data/ratings_clean.csv"
    )

    movies = pd.read_csv(
        "data/movies_clean.csv"
    )

    links = pd.read_csv(
        "data/links.csv"
    )

    return ratings, movies, links


ratings, movies, links = load_data()


# ============================================================
# MOVIE LOOKUP
# ============================================================

movie_lookup = (
    movies
    .merge(
        links,
        on="movieId",
        how="left"
    )
    .set_index("movieId")
)


# ============================================================
# MOVIE OPTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def create_movie_options(movies):

    title_counts = (
        movies["title"]
        .value_counts()
    )

    movie_label_to_id = {}

    for _, row in movies.sort_values(
        "title"
    ).iterrows():

        movie_id = int(
            row["movieId"]
        )

        title = str(
            row["title"]
        )

        display_title = title
        if title_counts[title] > 1:
            label = f"{display_title} | Movie ID {movie_id}"
        else:
            label = display_title

        movie_label_to_id[
            label
        ] = movie_id

    return movie_label_to_id


movie_label_to_id = (
    create_movie_options(movies)
)


def movie_title_year(title):
    """Return a local dataset title and its year, when present."""
    title = str(title)
    match = re.search(r"\((\d{4})\)\s*$", title)
    return (title[:match.start()].strip(), match.group(1)) if match else (title, "Year unavailable")


def filter_by_release_decade(frame, decade):
    """Filter local catalog recommendations by the year embedded in movie titles."""
    if frame.empty or decade == "Any era":
        return frame
    years = pd.to_numeric(frame["title"].map(lambda title: movie_title_year(title)[1]), errors="coerce")
    if decade == "Before 1980":
        return frame.loc[years < 1980].copy()
    start_year = int(decade[:-1])
    return frame.loc[(years >= start_year) & (years < start_year + 10)].copy()


def get_similar_movies(movie_id, limit=8):
    """Use the fitted item-CF model to return cosine-nearest local movies."""
    movie_id = int(movie_id)
    if movie_id not in movie_to_index:
        return pd.DataFrame()
    movie_index = movie_to_index[movie_id]
    neighbor_count = min(21, normalized_item_matrix.shape[0])
    distances, indices = item_model.kneighbors(
        normalized_item_matrix[movie_index], n_neighbors=neighbor_count
    )
    rows = []
    for distance, neighbor_index in zip(distances[0], indices[0]):
        similar_id = index_to_movie[int(neighbor_index)]
        similarity = float(1.0 - distance)
        if similar_id == movie_id or similarity <= 0 or similar_id not in movie_lookup.index:
            continue
        movie = movie_lookup.loc[similar_id]
        rows.append({
            "movieId": int(similar_id), "title": str(movie["title"]),
            "genres": "" if pd.isna(movie["genres"]) else str(movie["genres"]),
            "imdbId": movie.get("imdbId"), "tmdbId": movie.get("tmdbId"),
            "similarity": similarity,
        })
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows)


# ============================================================
# PREPARE SPARSE MATRIX
# ============================================================

@st.cache_resource(show_spinner=False)
def prepare_matrices(_ratings):

    user_ids = np.sort(
        _ratings["userId"].unique()
    )

    movie_ids = np.sort(
        _ratings["movieId"].unique()
    )

    user_to_index = {
        int(user_id): index
        for index, user_id
        in enumerate(user_ids)
    }

    movie_to_index = {
        int(movie_id): index
        for index, movie_id
        in enumerate(movie_ids)
    }

    index_to_movie = {
        index: int(movie_id)
        for movie_id, index
        in movie_to_index.items()
    }

    user_indices = (
        _ratings["userId"]
        .map(user_to_index)
        .to_numpy(dtype=np.int32)
    )

    movie_indices = (
        _ratings["movieId"]
        .map(movie_to_index)
        .to_numpy(dtype=np.int32)
    )

    rating_values = (
        _ratings["rating"]
        .to_numpy(dtype=np.float32)
    )

    matrix = csr_matrix(
        (
            rating_values,
            (
                user_indices,
                movie_indices
            )
        ),
        shape=(
            len(user_ids),
            len(movie_ids)
        ),
        dtype=np.float32
    )

    return (
        matrix,
        user_ids,
        movie_ids,
        user_to_index,
        movie_to_index,
        index_to_movie
    )


(
    user_movie_matrix,
    user_ids,
    movie_ids,
    user_to_index,
    movie_to_index,
    index_to_movie
) = prepare_matrices(ratings)


# ============================================================
# USER MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def train_user_model(_matrix):

    n_neighbors = min(
        21,
        _matrix.shape[0]
    )

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=n_neighbors
    )

    model.fit(_matrix)

    return model


# ============================================================
# ITEM MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def train_item_model(_matrix):

    movie_user_matrix = (
        _matrix.T.tocsr()
    )

    normalized_matrix = normalize(
        movie_user_matrix,
        axis=1
    )

    n_neighbors = min(
        21,
        normalized_matrix.shape[0]
    )

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=n_neighbors
    )

    model.fit(
        normalized_matrix
    )

    return (
        model,
        normalized_matrix
    )


# ============================================================
# SVD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def train_svd_model(_ratings):

    sample_size = min(
        1_000_000,
        len(_ratings)
    )

    sample = _ratings.sample(
        n=sample_size,
        random_state=42
    )

    reader = Reader(
        rating_scale=(0.5, 5.0)
    )

    data = Dataset.load_from_df(
        sample[
            [
                "userId",
                "movieId",
                "rating"
            ]
        ],
        reader
    )

    trainset = (
        data.build_full_trainset()
    )

    model = SVD(
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42
    )

    model.fit(trainset)

    return model


# ============================================================
# TRAIN MODELS
# ============================================================

with st.spinner(
    "Preparing recommendation models..."
):

    user_model = train_user_model(
        user_movie_matrix
    )

    (
        item_model,
        normalized_item_matrix
    ) = train_item_model(
        user_movie_matrix
    )

    svd_model = train_svd_model(
        ratings
    )


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def normalize_scores(scores):

    if not scores:
        return {}

    scores = {
        key: float(value)
        for key, value in scores.items()
        if np.isfinite(value)
    }
    if not scores:
        return {}

    values = np.array(list(scores.values()), dtype=float)

    minimum = values.min()
    maximum = values.max()

    if np.isclose(
        minimum,
        maximum
    ):

        return {
            key: 1.0
            for key in scores
        }

    return {
        key: float(
            (value - minimum)
            / (maximum - minimum)
        )
        for key, value in scores.items()
    }


# ============================================================
# USER-BASED COLLABORATIVE FILTERING
# ============================================================

@st.cache_data(show_spinner=False, max_entries=128)
def get_user_based_scores(
    profile_ratings
):

    profile_vector = lil_matrix(
        (
            1,
            user_movie_matrix.shape[1]
        ),
        dtype=np.float32
    )

    rated_movie_ids = set()

    for movie_id, rating in profile_ratings:

        if movie_id not in movie_to_index:
            continue

        column = movie_to_index[
            movie_id
        ]

        profile_vector[
            0,
            column
        ] = rating

        rated_movie_ids.add(
            movie_id
        )

    profile_vector = (
        profile_vector.tocsr()
    )

    neighbor_count = min(
        20,
        user_movie_matrix.shape[0]
    )

    distances, indices = (
        user_model.kneighbors(
            profile_vector,
            n_neighbors=neighbor_count
        )
    )

    similarities = (
        1.0 - distances[0]
    )

    neighbors = indices[0]

    profile_mean = np.mean(
        [
            rating
            for _, rating
            in profile_ratings
        ]
    )

    weighted_scores = {}
    similarity_weights = {}

    for similarity, neighbor_index in zip(
        similarities,
        neighbors
    ):

        if similarity <= 0:
            continue

        neighbor_row = (
            user_movie_matrix
            .getrow(neighbor_index)
        )

        if neighbor_row.nnz == 0:
            continue

        neighbor_mean = (
            neighbor_row.data.mean()
        )

        for column, rating in zip(
            neighbor_row.indices,
            neighbor_row.data
        ):

            movie_id = (
                index_to_movie[column]
            )

            if movie_id in rated_movie_ids:
                continue

            deviation = (
                float(rating)
                - float(neighbor_mean)
            )

            weighted_scores[
                movie_id
            ] = (
                weighted_scores.get(
                    movie_id,
                    0.0
                )
                + similarity * deviation
            )

            similarity_weights[
                movie_id
            ] = (
                similarity_weights.get(
                    movie_id,
                    0.0
                )
                + similarity
            )

    final_scores = {}

    for movie_id, value in (
        weighted_scores.items()
    ):

        weight = (
            similarity_weights.get(
                movie_id,
                0.0
            )
        )

        if weight <= 0:
            continue

        prediction = (
            profile_mean
            + value / weight
        )

        final_scores[movie_id] = float(
            np.clip(
                prediction,
                0.5,
                5.0
            )
        )

    return final_scores


# ============================================================
# ITEM-BASED COLLABORATIVE FILTERING
# ============================================================

@st.cache_data(show_spinner=False, max_entries=128)
def get_item_based_scores(
    profile_ratings
):

    rated_movie_ids = {
        movie_id
        for movie_id, _
        in profile_ratings
    }

    weighted_scores = {}
    similarity_weights = {}

    neighbor_count = min(
        21,
        normalized_item_matrix.shape[0]
    )

    for movie_id, rating in profile_ratings:

        if movie_id not in movie_to_index:
            continue

        movie_index = (
            movie_to_index[movie_id]
        )

        query_vector = (
            normalized_item_matrix[
                movie_index
            ]
        )

        distances, indices = (
            item_model.kneighbors(
                query_vector,
                n_neighbors=neighbor_count
            )
        )

        similarities = (
            1.0 - distances[0]
        )

        neighbors = indices[0]

        preference = (
            float(rating) - 3.0
        )

        for similarity, neighbor_index in zip(
            similarities,
            neighbors
        ):

            if similarity <= 0:
                continue

            candidate_movie_id = (
                index_to_movie[
                    neighbor_index
                ]
            )

            if (
                candidate_movie_id
                in rated_movie_ids
            ):
                continue

            weighted_scores[
                candidate_movie_id
            ] = (
                weighted_scores.get(
                    candidate_movie_id,
                    0.0
                )
                + similarity * preference
            )

            similarity_weights[
                candidate_movie_id
            ] = (
                similarity_weights.get(
                    candidate_movie_id,
                    0.0
                )
                + similarity
            )

    final_scores = {}
    final_support = {}

    for movie_id, value in (
        weighted_scores.items()
    ):

        weight = (
            similarity_weights.get(
                movie_id,
                0.0
            )
        )

        if weight <= 0:
            continue

        # Keep how much similarity evidence supports each candidate.
        # This prevents equal profile ratings from making every item
        # score identical after the weighted average cancels similarity.
        confidence = weight / (weight + 1.0)
        final_scores[movie_id] = (value / weight) * confidence
        final_support[movie_id] = float(weight)

    return final_scores, final_support


# ============================================================
# SVD COLD-START PROFILE
# ============================================================

@st.cache_data(show_spinner=False, max_entries=128)
def get_svd_cold_start_scores(
    profile_ratings
):

    trainset = svd_model.trainset

    raw_to_inner = (
        trainset._raw2inner_id_items
    )

    known_items = []
    known_ratings = []

    for movie_id, rating in profile_ratings:

        movie_id = int(movie_id)

        if movie_id not in raw_to_inner:
            continue

        known_items.append(
            raw_to_inner[movie_id]
        )

        known_ratings.append(
            float(rating)
        )

    if not known_items:
        return {}

    known_items = np.array(
        known_items,
        dtype=np.int32
    )

    known_ratings = np.array(
        known_ratings,
        dtype=float
    )

    global_mean = float(
        trainset.global_mean
    )

    item_biases = np.asarray(
        svd_model.bi,
        dtype=float
    )

    item_factors = np.asarray(
        svd_model.qi,
        dtype=float
    )

    X = np.column_stack(
        [
            np.ones(
                len(known_items)
            ),
            item_factors[
                known_items
            ]
        ]
    )

    y = (
        known_ratings
        - global_mean
        - item_biases[
            known_items
        ]
    )

    regularization = 1.0

    A = (
        X.T @ X
        + regularization
        * np.eye(
            X.shape[1]
        )
    )

    A[0, 0] -= regularization

    b = X.T @ y

    try:

        beta = np.linalg.solve(
            A,
            b
        )

    except np.linalg.LinAlgError:

        beta = (
            np.linalg.pinv(A)
            @ b
        )

    user_bias = float(
        beta[0]
    )

    user_factors = beta[1:]

    rated_movie_ids = {
        int(movie_id)
        for movie_id, _
        in profile_ratings
    }

    scores = {}

    for movie_id in movie_ids:

        movie_id = int(movie_id)

        if movie_id in rated_movie_ids:
            continue

        if movie_id not in raw_to_inner:
            continue

        inner_id = (
            raw_to_inner[movie_id]
        )

        prediction = (
            global_mean
            + user_bias
            + item_biases[inner_id]
            + np.dot(
                item_factors[inner_id],
                user_factors
            )
        )

        scores[movie_id] = float(
            np.clip(
                prediction,
                0.5,
                5.0
            )
        )

    return scores


# ============================================================
# HYBRID RECOMMENDATION ENGINE
# ============================================================

@st.cache_data(show_spinner=False, max_entries=128)
def get_recommendations(
    profile_ratings,
    selected_genres=None,
    selected_mood=None,
    selected_release_decade="Any era",
):

    user_raw = (
        get_user_based_scores(
            profile_ratings
        )
    )

    item_raw, item_support = (
        get_item_based_scores(
            profile_ratings
        )
    )

    svd_raw = (
        get_svd_cold_start_scores(
            profile_ratings
        )
    )

    user_scores = normalize_scores(
        user_raw
    )

    item_scores = normalize_scores(
        item_raw
    )

    svd_scores = normalize_scores(
        svd_raw
    )

    rated_movies = {
        movie_id
        for movie_id, _
        in profile_ratings
    }

    candidate_movies = (
        set(user_scores)
        | set(item_scores)
        | set(svd_scores)
    )

    candidate_movies -= rated_movies

    results = []

    for movie_id in candidate_movies:

        user_score = (
            user_scores.get(
                movie_id,
                0.0
            )
        )

        item_score = (
            item_scores.get(
                movie_id,
                0.0
            )
        )

        svd_score = (
            svd_scores.get(
                movie_id,
                0.0
            )
        )

        base_hybrid_score = (
            0.30 * user_score
            + 0.30 * item_score
            + 0.40 * svd_score
        )

        if movie_id not in movie_lookup.index:
            continue

        imdb_id = movie_lookup.loc[movie_id, "imdbId"]
        tmdb_id = movie_lookup.loc[movie_id, "tmdbId"]
        raw_genres = movie_lookup.loc[movie_id, "genres"]
        genre_value = "" if pd.isna(raw_genres) else str(raw_genres)
        context_score = get_context_score(
            genre_value,
            selected_genres or [],
            selected_mood
        )
        results.append(
            {
                "movieId": int(movie_id),
                "title": str(
                    movie_lookup.loc[
                        movie_id,
                        "title"
                    ]
                ),
                "genres": str(
                    movie_lookup.loc[
                        movie_id,
                        "genres"
                    ]
                ),
                "imdbId": imdb_id,
                "tmdbId": tmdb_id,
                "user_score": user_score,
                "item_score": item_score,
                "svd_score": svd_score,
                "item_support": item_support.get(movie_id, 0.0),
                "base_hybrid_score": float(base_hybrid_score),
                "context_score": context_score,
                "hybrid_score": float(base_hybrid_score)
            }
        )

    result_df = pd.DataFrame(
        results
    )

    if result_df.empty:
        return result_df

    result_df = filter_by_release_decade(result_df, selected_release_decade)
    if result_df.empty:
        return result_df

    has_context = bool(selected_genres) or bool(selected_mood)
    if has_context:
        context_matches = result_df["context_score"].fillna(0) > 0
        if context_matches.any():
            # Explicit viewing preferences define the eligible pool; the
            # recommendation model personalizes ordering within that pool.
            result_df = result_df.loc[context_matches].copy()
        else:
            return result_df.iloc[0:0].copy()
    context_available = (
        has_context
        and result_df["context_score"].notna().any()
        and result_df["context_score"].fillna(0).max() > 0
    )
    result_df["context_applied"] = bool(context_available)
    if context_available:
        result_df["final_score"] = (
            (1.0 - CONTEXT_WEIGHT) * result_df["base_hybrid_score"]
            + CONTEXT_WEIGHT * result_df["context_score"]
        )
    else:
        # Preserve the original ranking exactly when no context is selected
        # or no candidate has any meaningful contextual match.
        result_df["final_score"] = result_df["base_hybrid_score"]
    result_df["hybrid_score"] = result_df["final_score"]

    result_df = (
        result_df
        .sort_values(
            "final_score",
            ascending=False
        )
        .head(20)
        .reset_index(drop=True)
    )

    result_df.insert(
        0,
        "rank",
        range(
            1,
            len(result_df) + 1
        )
    )

    return result_df


@st.cache_data(show_spinner=False, max_entries=128)
def get_current_viewing_recommendations(selected_genres=None, selected_mood=None, selected_release_decade="Any era"):
    """Recommend from metadata for a current-viewing request without a taste profile."""
    selected_genres = list(dict.fromkeys(selected_genres or []))
    mood_genres = MOOD_GENRES.get(selected_mood, set()) if selected_mood else set()
    catalog = movie_lookup
    genre_text = catalog["genres"].fillna("").astype(str)
    genre_score = pd.Series(0.0, index=catalog.index)
    if selected_genres:
        matches = []
        for genre in selected_genres:
            if genre == "Rom-Com":
                mask = genre_text.str.contains(r"(?:^|\|)Romance(?:\||$)", regex=True) & genre_text.str.contains(r"(?:^|\|)Comedy(?:\||$)", regex=True)
            else:
                mask = genre_text.str.contains(rf"(?:^|\|){re.escape(genre)}(?:\||$)", regex=True)
            matches.append(mask.astype(float))
        genre_score = sum(matches) / len(matches)
    mood_score = pd.Series(0.0, index=catalog.index)
    if mood_genres:
        for genre in mood_genres:
            mood_score += genre_text.str.contains(rf"(?:^|\|){re.escape(genre)}(?:\||$)", regex=True).astype(float)
        mood_score /= len(mood_genres)

    if selected_genres and mood_genres:
        score = 0.70 * genre_score + 0.30 * mood_score
    elif selected_genres:
        score = genre_score
    else:
        score = mood_score
    eligible = score > 0 if selected_genres else pd.Series(True, index=catalog.index)
    result_df = catalog.loc[eligible, ["title", "genres", "imdbId", "tmdbId"]].copy()
    if result_df.empty:
        return result_df
    result_df["movieId"] = result_df.index.astype(int)
    result_df["user_score"] = 0.0
    result_df["item_score"] = 0.0
    result_df["svd_score"] = 0.0
    result_df["item_support"] = 0.0
    result_df["base_hybrid_score"] = None
    result_df["context_score"] = score.loc[result_df.index].astype(float)
    result_df["hybrid_score"] = result_df["context_score"]
    result_df["final_score"] = result_df["context_score"]
    result_df["context_applied"] = True
    result_df = filter_by_release_decade(result_df, selected_release_decade)
    if result_df.empty:
        return result_df

    result_df = (
        result_df.sort_values(["final_score", "title"], ascending=[False, True], kind="stable")
        .head(20)
        .reset_index(drop=True)
    )
    result_df.insert(0, "rank", range(1, len(result_df) + 1))
    return result_df


# ============================================================
# GENRE PROFILE
# ============================================================

def get_genre_profile(
    profile_ratings
):

    genre_data = {}

    for movie_id, rating in profile_ratings:

        if movie_id not in movie_lookup.index:
            continue

        genres = str(
            movie_lookup.loc[
                movie_id,
                "genres"
            ]
        )

        if genres == "(no genres listed)":
            continue

        for genre in genres.split("|"):

            if genre not in genre_data:
                genre_data[genre] = []

            genre_data[genre].append(
                float(rating)
            )

    rows = []

    for genre, values in genre_data.items():

        rows.append(
            {
                "Genre": genre,
                "Movies": len(values),
                "Average Rating": round(
                    np.mean(values),
                    2
                )
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            "Average Rating",
            ascending=False
        )
        .reset_index(drop=True)
    )


# Mood is not a MovieLens field; these associations interpret mood through
# genres that exist in the current MovieMind dataset.
MOOD_GENRES = {
    "Feel-Good": {"Comedy", "Children", "Animation"},
    "Funny": {"Comedy"},
    "Romantic": {"Romance"},
    "Emotional": {"Drama", "Romance"},
    "Relaxing": {"Animation", "Comedy", "Documentary"},
    "Exciting": {"Action", "Adventure", "Thriller"},
    "Dark": {"Horror", "Crime", "Thriller"},
    "Suspenseful": {"Thriller", "Mystery", "Crime"},
}

PRACTICAL_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Horror", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
DATASET_GENRES = {
    genre
    for genres in movies["genres"].fillna("").astype(str)
    for genre in genres.split("|")
    if genre and genre != "(no genres listed)"
}
GENRE_OPTIONS = [genre for genre in PRACTICAL_GENRES if genre in DATASET_GENRES]
if {"Romance", "Comedy"}.issubset(DATASET_GENRES):
    # Convenience context for a combined romance/comedy request; the dataset
    # stores these as separate genre tags.
    GENRE_OPTIONS.append("Rom-Com")
DECADE_OPTIONS = ["Any era", "Before 1980", *[f"{year}s" for year in range(1980, 2030, 10)]]


def get_context_score(movie_genres, selected_genres=None, selected_mood=None):
    """Return a semantic 0-1 genre/mood relevance score, or None without context."""
    selected_genres = set(selected_genres or [])
    mood_genres = MOOD_GENRES.get(selected_mood, set()) if selected_mood else set()
    if not selected_genres and not mood_genres:
        return None

    if not isinstance(movie_genres, str) or not movie_genres.strip():
        actual = set()
    else:
        actual = {
            genre.strip()
            for genre in movie_genres.split("|")
            if genre.strip() and genre.strip() != "(no genres listed)"
        }

    component_scores = []
    if selected_genres:
        genre_matches = []
        for genre in selected_genres:
            if genre == "Rom-Com":
                genre_matches.append({"Romance", "Comedy"}.issubset(actual))
            else:
                genre_matches.append(genre in actual)
        component_scores.append(sum(genre_matches) / len(genre_matches))
    if mood_genres:
        component_scores.append(len(actual & mood_genres) / len(mood_genres))
    return float(np.mean(component_scores)) if component_scores else 0.0


# ============================================================
# ============================================================
# LOCAL PROFILE STORAGE
# ============================================================

def load_profiles():
    return load_profile_records(PROFILE_PATH)


def save_profiles(profiles):
    save_profile_records(profiles, PROFILE_PATH)


def get_active_watch_later_entries(profiles=None):
    profiles = load_profiles() if profiles is None else profiles
    profile_id = st.session_state.get("profile_id")
    if not profile_id and any(
        profile.get("profile_id") == DEFAULT_WATCH_LATER_PROFILE_ID
        for profile in profiles
    ):
        profile_id = DEFAULT_WATCH_LATER_PROFILE_ID
    profile = next(
        (item for item in profiles if item.get("profile_id") == profile_id),
        None,
    )
    entries = (profile or {}).get("watch_later", [])
    return list(entries) if isinstance(entries, list) else []


def toggle_active_watch_later(movie):
    """Persist one add/remove operation through the configured profile store."""
    profiles = load_profiles()
    profile_id = st.session_state.get("profile_id")
    if not profile_id:
        profile_id = DEFAULT_WATCH_LATER_PROFILE_ID
        st.session_state["profile_id"] = profile_id

    profile = next(
        (item for item in profiles if item.get("profile_id") == profile_id),
        None,
    )
    if profile is None:
        profile = {
            "profile_id": profile_id,
            "name": st.session_state.get("profile_name") or "My MovieMind",
            "movies": [],
            "watch_later": [],
        }
        profiles.append(profile)

    watch_later, saved = toggle_watch_later_entry(
        profile.get("watch_later", []),
        movie,
    )
    profile["watch_later"] = watch_later
    save_profiles(profiles)
    st.session_state["profile_id"] = profile_id
    return saved


def watch_later_frame(entries):
    rows = []
    for entry in entries:
        if entry.get("external_tmdb"):
            tmdb_id = int(entry["tmdbId"])
            rows.append({
                "movieId": -tmdb_id,
                "tmdbId": tmdb_id,
                "imdbId": None,
                "title": str(entry.get("title", "")),
                "genres": str(entry.get("genres", "")),
                "release_date": str(entry.get("release_date", "")),
                "poster_path": entry.get("poster_path"),
                "backdrop_path": entry.get("backdrop_path"),
                "vote_average": entry.get("vote_average"),
                "_external_tmdb": True,
            })
            continue
        try:
            movie_id = int(entry.get("movieId"))
        except (TypeError, ValueError, OverflowError):
            continue
        if movie_id not in movie_lookup.index:
            continue
        movie = movie_lookup.loc[movie_id]
        rows.append({
            "movieId": movie_id,
            "tmdbId": movie.get("tmdbId"),
            "imdbId": movie.get("imdbId"),
            "title": str(movie.get("title", "")),
            "genres": "" if pd.isna(movie.get("genres")) else str(movie.get("genres")),
        })
    return pd.DataFrame(rows)


def profile_rows_to_pairs(profile):
    unique = {}
    for row in profile.get("movies", []):
        movie_id = int(row["movieId"])
        rating = float(row["rating"])
        if movie_id in movie_lookup.index and np.isfinite(rating):
            unique[movie_id] = float(np.clip(rating, 0.5, 5.0))
    return list(unique.items())


def apply_profile(profile):
    pairs = profile_rows_to_pairs(profile)
    id_to_label = {movie_id: label for label, movie_id in movie_label_to_id.items()}
    st.session_state["profile_name"] = profile.get("name", "")
    st.session_state["profile_id"] = profile.get("profile_id")
    st.session_state["profile"] = pairs
    st.session_state["selected_movies"] = [id_to_label[mid] for mid, _ in pairs if mid in id_to_label]
    for movie_id, rating in pairs:
        st.session_state[f"taste_rating_{movie_id}"] = rating
    st.session_state.pop("recommendations", None)


def reset_profile_state():
    for key in list(st.session_state):
        if key.startswith("taste_rating_"):
            del st.session_state[key]
    st.session_state["selected_movies"] = []
    st.session_state["profile_name"] = ""
    st.session_state.pop("profile_id", None)
    st.session_state.pop("profile", None)
    st.session_state.pop("_last_autosave_key", None)
    st.session_state.pop("recommendations", None)
    clear_ai_request_state()


def clear_ai_request_state(clear_recommendations=False):
    """Remove only submitted AI state; manual genre and mood widgets remain untouched."""
    for key in (
        "ai_intent_summary",
        "ai_reference_movie_id",
        "ai_reference_results",
        "ai_reference_missing",
        "ai_request_succeeded",
        "ai_request_error",
    ):
        st.session_state.pop(key, None)
    if st.session_state.get("recommendation_source") == "ai":
        st.session_state["recommendation_source"] = "manual"
        clear_recommendations = True
    if clear_recommendations:
        st.session_state["recommendations"] = pd.DataFrame()
        st.session_state["surprise_result"] = False


def clear_ai_state_for_manual_context():
    """Keep AI summaries from being presented as the current manual request."""
    clear_ai_request_state(clear_recommendations=True)


def set_active_view(view):
    st.session_state["active_view"] = view


def set_home_category(category, view, discover_category=None, discover_genre=None):
    st.session_state["home_category"] = category
    st.session_state["active_view"] = view
    if discover_category:
        st.session_state["discover_category"] = discover_category
    if discover_genre:
        st.session_state["discover_genre"] = discover_genre
        st.session_state["discover_page"] = 1
        st.session_state["discover_mode"] = "genres"
    elif view == "Discover":
        st.session_state["discover_mode"] = "genres"


def open_genre_discovery():
    st.session_state["active_view"] = "Discover"
    st.session_state["discover_mode"] = "genres"


def open_movie_category(category):
    st.session_state["active_view"] = "Discover"
    st.session_state["discover_mode"] = "category"
    st.session_state["discover_category"] = category
    st.session_state["discover_page"] = 1


@st.dialog("Watch movie trailer", width="large")
def show_trailer_dialog(movie):
    """Play a movie trailer in a large, widescreen in-app dialog."""
    movie = movie if isinstance(movie, dict) else {}
    title = str(movie.get("title") or "Movie trailer")
    try:
        tmdb_id = int(movie.get("tmdbId"))
    except (TypeError, ValueError, OverflowError):
        tmdb_id = 0
    try:
        movie_id = int(movie.get("movieId"))
    except (TypeError, ValueError, OverflowError):
        movie_id = 0

    st.subheader(title)
    trailer = get_movie_trailer(tmdb_id)
    if trailer:
        st.video(f"https://www.youtube.com/watch?v={trailer['key']}", width="stretch")
        st.caption(trailer.get("name") or "Official trailer")
        st.caption("If the embedded player is unavailable, open the trailer directly on YouTube.")
        st.link_button("Open trailer on YouTube", f"https://www.youtube.com/watch?v={trailer['key']}")
    else:
        st.info("No embeddable trailer is listed for this movie yet. Try a YouTube search or browse similar movies below.")
        st.link_button(
            "Search YouTube for the trailer",
            f"https://www.youtube.com/results?search_query={quote_plus(title + ' official trailer')}",
        )
        if tmdb_id > 0:
            st.markdown(f"[View this movie on TMDB](https://www.themoviedb.org/movie/{tmdb_id})")

    if movie_id > 0 and movie_id in movie_lookup.index:
        similar_frame = get_similar_movies(movie_id, limit=8)
        similar_mode = "similarity"
    elif tmdb_id > 0:
        similar_movies = get_tmdb_similar_movies(tmdb_id)
        similar_frame = make_discover_frame(similar_movies) if similar_movies else pd.DataFrame()
        similar_mode = "discover"
    else:
        similar_frame = pd.DataFrame()
        similar_mode = "similarity"
    if not similar_frame.empty:
        st.markdown(f"### More Like {movie_title_year(title)[0]}")
        render_circular_carousel(
            similar_frame,
            similar_mode,
            {},
            max_cards=min(6, len(similar_frame)),
            height=420,
            carousel_id=f"trailer_similar_{tmdb_id or movie_id}",
        )


def render_play_trailer(movie, key_prefix, label="▶  Play Trailer"):
    """Open the selected movie's YouTube trailer in a large in-app dialog."""
    tmdb_id = movie.get("tmdbId")
    try:
        tmdb_id = int(tmdb_id)
    except (TypeError, ValueError, OverflowError):
        tmdb_id = None
    if not tmdb_id or tmdb_id <= 0:
        st.caption("Trailer unavailable for this title.")
        return

    if st.button(label, type="primary", key=f"{key_prefix}_play_trailer_{tmdb_id}"):
        movie_data = movie.to_dict() if hasattr(movie, "to_dict") else dict(movie)
        show_trailer_dialog(movie_data)


def is_gemini_rate_limit_error(error):
    """Recognize SDK and HTTP 429 errors without exposing the exception text."""
    response = getattr(error, "response", None)
    return (
        getattr(error, "code", None) == 429
        or getattr(error, "status_code", None) == 429
        or getattr(response, "status_code", None) == 429
        or "429" in str(error)
        or "rate limit" in str(error).casefold()
    )


def normalize_ai_genres(genres):
    """Map Gemini's genre labels onto values understood by local MovieMind context."""
    aliases = {
        "science fiction": "Sci-Fi", "sci fi": "Sci-Fi", "sci-fi": "Sci-Fi",
        "family": "Children", "children's": "Children", "film noir": "Film-Noir",
        "rom com": "Rom-Com", "rom-com": "Rom-Com", "romcom": "Rom-Com",
        "romantic comedy": "Rom-Com", "romantic comedies": "Rom-Com",
        "romance comedy": "Rom-Com",
    }
    by_casefold = {genre.casefold(): genre for genre in DATASET_GENRES}
    normalized = []
    for value in genres or []:
        candidate = str(value).strip()
        if not candidate:
            continue
        canonical = aliases.get(candidate.casefold(), by_casefold.get(candidate.casefold(), candidate))
        if canonical == "Rom-Com" and canonical in GENRE_OPTIONS and canonical not in normalized:
            normalized.append(canonical)
        elif canonical in DATASET_GENRES and canonical not in normalized:
            normalized.append(canonical)
    if {"Romance", "Comedy"}.issubset(normalized) and "Rom-Com" in GENRE_OPTIONS:
        normalized = [genre for genre in normalized if genre not in {"Romance", "Comedy"}]
        normalized.insert(0, "Rom-Com")
    return normalized


def normalize_ai_mood(mood):
    """Map supported intent words to the existing mood-to-genre mapping."""
    if not mood:
        return None
    key = re.sub(r"[^a-z]+", " ", str(mood).casefold()).strip()
    aliases = {
        "funny": "Funny", "humorous": "Funny", "comedic": "Funny",
        "romantic": "Romantic", "love": "Romantic",
        "feel good": "Feel-Good", "uplifting": "Feel-Good",
        "emotional": "Emotional", "moving": "Emotional",
        "relaxing": "Relaxing", "calm": "Relaxing", "cozy": "Relaxing",
        "exciting": "Exciting", "adventurous": "Exciting",
        "dark": "Dark", "suspenseful": "Suspenseful", "tense": "Suspenseful",
    }
    return aliases.get(key)


def find_local_movie_id(title):
    """Resolve a reference title against the local catalog without external search."""
    if not title:
        return None

    def normalize_title(value):
        value = re.sub(r"\s*\(\d{4}\)\s*$", "", str(value))
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    requested = normalize_title(title)
    if not requested:
        return None
    matches = movies.loc[movies["title"].map(normalize_title) == requested, "movieId"]
    return int(matches.iloc[0]) if not matches.empty else None


def parse_movie_intent_fast(user_text):
    """Resolve short, explicit genre or mood requests without a network round-trip."""
    text = re.sub(r"[^a-z0-9' -]+", " ", str(user_text).casefold())
    text = re.sub(r"\s+", " ", text).strip()
    if not text or len(text.split()) > 10 or re.search(r"\b(no|not|without|except|avoid|exclude)\b", text):
        return None

    genre_patterns = [
        (r"\b(?:rom\s*com|romcom|romantic comedy|romance comedy)\b", "Rom-Com"),
        (r"\bscience fiction\b|\bsci fi\b|\bsci-fi\b", "Sci-Fi"),
        (r"\baction\b", "Action"), (r"\badventure\b", "Adventure"),
        (r"\banimation\b|\banimated\b", "Animation"), (r"\bcomedy\b|\bcomedies\b", "Comedy"),
        (r"\bcrime\b", "Crime"), (r"\bdocumentary\b|\bdocumentaries\b", "Documentary"),
        (r"\bdrama\b|\bdramatic\b", "Drama"), (r"\bfantasy\b", "Fantasy"),
        (r"\bhorror\b|\bscary\b", "Horror"), (r"\bmystery\b", "Mystery"),
        (r"\bromance\b|\bromantic\b", "Romance"), (r"\bthriller\b|\bthrillers\b", "Thriller"),
        (r"\bwar\b", "War"), (r"\bwestern\b", "Western"),
    ]
    genres, consumed = [], []
    for pattern, label in genre_patterns:
        if re.search(pattern, text) and label not in genres:
            genres.append(label)
            consumed.append(pattern)

    mood_patterns = [
        (r"\bfunny\b|\bhumorous\b|\bcomedic\b", "Funny"),
        (r"\bfeel good\b|\buplifting\b", "Feel-Good"),
        (r"\bemotional\b|\bmoving\b", "Emotional"),
        (r"\brelaxing\b|\bcalm\b|\bcozy\b", "Relaxing"),
        (r"\bexciting\b|\badventurous\b", "Exciting"),
        (r"\bdark\b", "Dark"), (r"\bsuspenseful\b|\btense\b", "Suspenseful"),
    ]
    mood = None
    for pattern, label in mood_patterns:
        if re.search(pattern, text):
            mood = label
            consumed.append(pattern)
            break

    reference_title = None
    reference_match = re.search(r"\b(?:like|similar to)\s+(.+)$", text)
    if reference_match:
        candidate = re.sub(r"\b(?:movie|film)$", "", reference_match.group(1)).strip()
        reference_id = find_local_movie_id(candidate)
        if reference_id is None:
            return None
        reference_title = str(movie_lookup.loc[reference_id, "title"])
        consumed.append(re.escape(reference_match.group(0)))

    remaining = text
    for pattern in consumed:
        remaining = re.sub(pattern, " ", remaining)
    stop_words = {
        "i", "want", "would", "love", "to", "watch", "looking", "for", "something", "anything",
        "me", "please", "recommend", "suggest", "find", "show", "give", "a", "an", "the",
        "movie", "movies", "film", "films", "tonight", "weekend", "this", "of", "with", "that",
        "have", "has", "more", "make", "feel", "good", "interesting", "kind", "sort", "one",
        "can", "you", "some", "on", "about", "new", "for", "date", "night",
    }
    leftovers = [word for word in re.findall(r"[a-z]+", remaining) if word not in stop_words]
    mapped_genres = normalize_ai_genres(genres)
    if leftovers or not (mapped_genres or mood or reference_title):
        return None
    return SimpleNamespace(genres=mapped_genres, mood=mood, reference_movie=reference_title, keywords=[])


def signal_explanation(movie, mode, context):
    """Build a restrained explanation from the recommendation's stored signals."""
    if mode == "current":
        labels = list(context.get("genres", []))
        mood = context.get("mood")
        intent = " + ".join(labels) if labels else "your selected mood"
        if mood:
            intent = f"{intent} + {mood}" if labels else mood
        return f"Its local genre metadata aligns with your current {intent} viewing intent."

    weighted = {
        "Taste Match": .30 * float(movie.get("user_score", 0) or 0),
        "Movie Similarity": .30 * float(movie.get("item_score", 0) or 0),
        "Preference Learning": .40 * float(movie.get("svd_score", 0) or 0),
    }
    strongest = max(weighted, key=weighted.get)
    explanations = {
        "Taste Match": "User-based collaborative filtering found rating patterns similar to your profile.",
        "Movie Similarity": "Item-based collaborative filtering found similarity to movies in your profile.",
        "Preference Learning": "Preference Learning found a strong signal from the ratings in your profile.",
    }
    explanation = explanations[strongest]
    if mode == "combined" and bool(movie.get("context_applied", False)) and float(movie.get("context_score", 0) or 0) > 0:
        labels = list(context.get("genres", []))
        mood = context.get("mood")
        intent = " + ".join(labels) if labels else "your selected mood"
        if mood:
            intent = f"{intent} + {mood}" if labels else mood
        explanation = f"Its genre metadata aligns with your current {intent} viewing intent; {explanation[0].lower() + explanation[1:]}"
    return f"{explanation} Main signal: {strongest}."


def strongest_signal_label(movie):
    """Return the strongest contributing signal for concise badge display."""
    weighted = {
        "Taste Match (30%)": .30 * float(movie.get("user_score", 0) or 0),
        "Movie Similarity (30%)": .30 * float(movie.get("item_score", 0) or 0),
        "Preference Learning (40%)": .40 * float(movie.get("svd_score", 0) or 0),
    }
    return max(weighted, key=weighted.get)


@st.cache_data(ttl=60 * 60, show_spinner=False, max_entries=32)
def image_data_uri(image, fallback_svg=POSTER_PLACEHOLDER):
    """Create a self-contained image URI for compact, styled movie cards."""
    if image:
        raw = image if isinstance(image, bytes) else str(image).encode("utf-8")
        mime = "image/jpeg" if isinstance(image, bytes) else "image/svg+xml"
    else:
        raw = fallback_svg.encode("utf-8")
        mime = "image/svg+xml"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def with_intent_match_columns(frame, genres, mood, reference_similarities=None, reference_title=None):
    """Add presentation-only intent columns without changing any ranking fields."""
    if frame is None or frame.empty:
        return frame
    frame = frame.copy()
    reference_similarities = reference_similarities or {}
    details = [
        get_intent_match_details(
            movie.get("genres", ""),
            genres,
            mood,
            MOOD_GENRES,
            reference_similarities.get(int(movie["movieId"])),
            reference_title,
        )
        for _, movie in frame.iterrows()
    ]
    frame["intent_match"] = [item[0] for item in details]
    frame["intent_explanation"] = [item[1] for item in details]
    return frame


def render_movie_card(movie, mode, context, row_key, wide=False):
    movie_id_value = movie.get("movieId")
    movie_id = int(movie_id_value) if movie_id_value is not None else 0
    is_external = bool(movie.get("_external_tmdb")) or movie_id < 0
    title, year = movie_title_year(movie["title"])
    metadata = None
    image_field = "backdrop_path" if wide else "poster_path"
    if not movie.get(image_field):
        metadata = get_movie_details(movie.get("tmdbId"))
    poster_path = (metadata or {}).get(image_field) or movie.get(image_field)
    if not poster_path and wide:
        poster_path = (metadata or {}).get("poster_path") or movie.get("poster_path")
    poster_uri = get_movie_poster_url(poster_path) or image_data_uri(None)
    is_intent = mode == "current" and movie.get("intent_match") is not None and pd.notna(movie.get("intent_match"))
    score = (float(movie.get("intent_match")) / 100) if is_intent else float(movie.get("hybrid_score", movie.get("similarity", 0)) or 0)
    score_label = "Intent Match" if is_intent else ("Similarity" if "similarity" in movie else "Match")
    browse_mode = mode in {"search", "discover", "watch_later"}
    watch_entry = watch_later_movie_entry(movie)
    watch_key = watch_later_key(watch_entry)
    already_saved = watch_key in st.session_state.get("_watch_later_keys", set())
    genres = str(movie.get("genres", "")).replace("|", " · ")
    trailer = get_movie_trailer(movie.get("tmdbId"))

    badge_html = f'<span class="card-badge">{score:.0%} {html.escape(score_label)}</span>' if not browse_mode else ''
    saved_html = '<span class="card-saved-badge">✓ Saved</span>' if already_saved else ''

    trailer_key = trailer.get("key", "") if trailer else ""
    light_theme = st.session_state.get("ui_theme", "dark") == "light"
    card_background = "linear-gradient(160deg,#ffffff,#f2eeea)" if light_theme else "linear-gradient(160deg,#3C2A21,#1A120B)"
    poster_background = "#e9e3dd" if light_theme else "#1A120B"
    card_text = "#2d241e" if light_theme else "#F5EBE6"
    card_muted = "#75685e" if light_theme else "#cbb8ad"
    card_border = "rgba(75,55,42,.20)" if light_theme else "rgba(242,204,143,.18)"
    card_accent = "#885a26" if light_theme else "#F2CC8F"
    preview_frame = (
        '<iframe class="movie-trailer-preview" title="Muted trailer preview" allow="autoplay; encrypted-media; picture-in-picture" '
        f'data-trailer-key="{html.escape(trailer_key, quote=True)}"></iframe>'
        if trailer_key else '<div class="movie-preview-unavailable">Trailer preview unavailable</div>'
    )
    card_height = 205 if wide else (520 if mode == "discover" else 365)
    poster_height = "auto"
    poster_ratio = "16 / 9" if wide else "2 / 3"
    card_markup = f"""
<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box}} body{{margin:0;background:transparent;color:{card_text};font-family:Arial,sans-serif}}
.movie-card{{overflow:hidden;border:1px solid {card_border};border-radius:10px;background:{card_background};height:100%;}}
.poster-wrap{{position:relative;width:100%;aspect-ratio:{poster_ratio};height:{poster_height};overflow:hidden;background:{poster_background};}}
.poster{{width:100%;height:100%;object-fit:{'cover' if wide else 'cover'};display:block;}}
.preview{{position:absolute;inset:0;width:100%;height:100%;border:0;display:none;pointer-events:none;background:{poster_background};}}
.movie-card.previewing .preview{{display:block;z-index:4;}}
.preview-label{{position:absolute;z-index:5;left:8px;bottom:8px;padding:4px 7px;border-radius:20px;background:rgba(26,18,11,.8);color:{card_accent};font-size:10px;display:none;}}
.movie-card.previewing .preview-label{{display:block;}}
.movie-preview-unavailable{{position:absolute;inset:0;display:none;align-items:center;justify-content:center;background:rgba(26,18,11,.9);color:{card_accent};font-size:12px;text-align:center;padding:12px;}}
.movie-card:hover .movie-preview-unavailable{{display:flex;}}
.poster-gradient{{position:absolute;inset:auto 0 0;height:45px;background:linear-gradient(0deg,rgba(26,18,11,.96),transparent);pointer-events:none;}}
.badge{{position:absolute;z-index:2;top:8px;left:8px;padding:3px 7px;border-radius:99px;background:#3C2A21e8;border:1px solid #F2CC8F80;color:{card_accent};font-size:10px;font-weight:700;}}
.saved{{position:absolute;z-index:2;right:8px;top:8px;padding:3px 7px;border-radius:99px;background:#3C2A21e8;border:1px solid #F2CC8F70;color:#F2CC8F;font-size:10px;}}
.info{{padding:9px 10px;}}
.title{{font-size:14px;line-height:1.25;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:{card_text};}}
.meta{{font-size:11px;color:{card_muted};margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.intent{{font-size:10px;color:{card_muted};margin-top:5px;line-height:1.25;max-height:26px;overflow:hidden;}}
</style></head><body>
<div class="movie-card" id="movieCard">
  <div class="poster-wrap"><img class="poster" src="{poster_uri}" alt="{html.escape(title, quote=True)}">{preview_frame}
  <div class="preview-label">Muted trailer preview</div><div class="poster-gradient"></div>{badge_html}{saved_html}</div>
  <div class="info"><div class="title" title="{html.escape(title, quote=True)}">{html.escape(title)}</div>
  <div class="meta">{html.escape(year if year != 'Year unavailable' else genres)}</div>
  {'<div class="intent">' + html.escape(str(movie.get('intent_explanation', ''))) + '</div>' if is_intent and movie.get('intent_explanation') else ''}</div>
</div>
<script>
(()=>{{const card=document.getElementById('movieCard');const frame=card.querySelector('.preview');if(!frame)return;
card.addEventListener('mouseenter',()=>{{if(!frame.dataset.trailerKey)return;card.classList.add('previewing');const key=frame.dataset.trailerKey;frame.src=`https://www.youtube-nocookie.com/embed/${{key}}?autoplay=1&mute=1&controls=0&playsinline=1&rel=0&loop=1&playlist=${{key}}`;}});
card.addEventListener('mouseleave',()=>{{card.classList.remove('previewing');frame.src='about:blank';}});}})();
</script></body></html>
"""
    components.html(card_markup, height=card_height, scrolling=False)
    with st.container(key=f"movie-actions-{row_key}-{movie_id}"):
        details_col, trailer_col = st.columns(2)
        similar_col, later_col = st.columns(2)
        if details_col.button("Details", key=f"details_{row_key}_{movie_id}", use_container_width=True):
            select_movie_details(st.session_state, movie)
            if not browse_mode and movie_id > 0:
                st.session_state["selected_hero_movie_id"] = movie_id
            st.rerun()
        if trailer_col.button("▶ Trailer", key=f"card_trailer_{row_key}_{movie_id}", use_container_width=True):
            movie_data = movie.to_dict() if hasattr(movie, "to_dict") else dict(movie)
            show_trailer_dialog(movie_data)
        if similar_col.button("More Like", key=f"similar_{row_key}_{movie_id}", use_container_width=True, disabled=is_external or movie_id <= 0):
            st.session_state["more_like_movie_id"] = movie_id
            st.session_state["more_like_results"] = get_similar_movies(movie_id)
            st.session_state["more_like_source_title"] = str(movie["title"])
        if later_col.button(
            "✓ In List" if already_saved else "+ My List",
            key=f"watch_later_{row_key}_{movie_id}",
            use_container_width=True,
            disabled=watch_entry is None,
        ):
            toggle_active_watch_later(movie)
            st.rerun()


def render_movie_row(title, frame, mode, context, row_key, caption=None):
    if frame is None or frame.empty:
        return
    st.markdown(f'<div class="row-heading">{html.escape(title)}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="row-caption">{html.escape(caption)}</div>', unsafe_allow_html=True)
    rows = [row for _, row in frame.iterrows()]
    for start in range(0, len(rows), 5):
        visible_rows = rows[start:start + 5]
        preload_movie_media([movie.get("tmdbId") for movie in visible_rows])
        with st.container(key=f"movie-row-{row_key}-{start}"):
            columns = st.columns(5)
            for column, movie in zip(columns, visible_rows):
                with column:
                    render_movie_card(movie, mode, context, row_key)


def render_movie_rail(title, frame, mode, context, row_key, limit=8):
    """Render a continuously looping, circular movie carousel."""
    if frame is None or frame.empty:
        return
    st.markdown(f'<div class="row-heading">{html.escape(title)}</div>', unsafe_allow_html=True)
    render_circular_carousel(
        frame.head(limit),
        mode,
        context,
        max_cards=limit,
        height=500,
        carousel_id=f"movie_carousel_{row_key}",
    )


def render_because_you_watched(context, row_key):
    """Show similar titles based on the highest-rated movie in the profile."""
    profile = st.session_state.get("profile", [])
    valid_ratings = []
    for item in profile:
        try:
            movie_id, rating = int(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if movie_id in movie_lookup.index:
            valid_ratings.append((movie_id, rating))
    if not valid_ratings:
        return

    watched_id, _ = max(valid_ratings, key=lambda pair: pair[1])
    watched_title = str(movie_lookup.loc[watched_id, "title"])
    similar = get_similar_movies(watched_id, limit=10)
    render_movie_rail(
        f"Because you watched {movie_title_year(watched_title)[0]}",
        similar,
        "similarity",
        context,
        row_key,
        limit=10,
    )


def render_streaming_feature(movie):
    """Render a full-width featured movie banner for the browse landing page."""
    movie_id = int(movie.get("movieId", 0) or 0)
    metadata = None
    backdrop_path = movie.get("backdrop_path")
    if not backdrop_path:
        metadata = get_movie_details(movie.get("tmdbId"))
        backdrop_path = (metadata or {}).get("backdrop_path")
    backdrop_uri = get_movie_poster_url(backdrop_path, "w1280") if backdrop_path else ""
    style = f"background-image:url('{backdrop_uri}');" if backdrop_uri else ""
    title, year = movie_title_year(movie.get("title", "Featured movie"))
    year = str(movie.get("release_date") or year)[:4]
    genres = str(movie.get("genres", "")).replace("|", " · ")
    overview = str(movie.get("overview") or (metadata or {}).get("overview") or "Discover a popular movie selected for your next watch.")
    st.markdown(
        '<section class="streaming-feature" style="' + style + '">'
        '<div class="streaming-feature-copy">'
        '<div class="streaming-feature-kicker">FEATURED · MOVIEMIND PICKS</div>'
        '<div class="streaming-feature-title">' + html.escape(title) + '</div>'
        '<div class="streaming-feature-meta">' + html.escape(" · ".join(part for part in (year, genres) if part)) + '</div>'
        '<div class="streaming-feature-overview">' + html.escape(overview) + '</div>'
        '</div></section>',
        unsafe_allow_html=True,
    )
    info_col, trailer_col, save_col, _ = st.columns([1, 1.2, 1.1, 4.8])
    if info_col.button("More Info", type="primary", key=f"stream_feature_info_{movie_id}"):
        select_movie_details(st.session_state, movie)
        st.rerun()
    with trailer_col:
        render_play_trailer(movie, "stream_feature")
    already_saved = watch_later_key(watch_later_movie_entry(movie)) in st.session_state.get("_watch_later_keys", set())
    if save_col.button("✓ In My List" if already_saved else "+ My List", key=f"stream_feature_save_{movie_id}"):
        toggle_active_watch_later(movie)
        st.rerun()
    render_movie_details()


def render_hero(movie, mode, context):
    movie_id = int(movie["movieId"])
    preload_movie_media([movie.get("tmdbId")], include_trailers=False)
    metadata = get_movie_details(movie.get("tmdbId"))
    backdrop_uri = get_movie_poster_url((metadata or {}).get("backdrop_path"), "w1280") or "linear-gradient(125deg,#19283b,#21172b 58%,#11151d)"
    title, year = movie_title_year(movie["title"])
    genres = str(movie.get("genres", "")).replace("|", " · ")
    is_intent = mode == "current" and movie.get("intent_match") is not None and pd.notna(movie.get("intent_match"))
    score = (float(movie.get("intent_match")) / 100) if is_intent else float(movie.get("hybrid_score", 0) or 0)
    score_title = "VIEWING INTENT" if is_intent else "MOVIEMIND MATCH"
    release_year = year if year != "Year unavailable" else "Release year unavailable"
    overview = (metadata or {}).get("overview", "").strip()
    why = movie.get("intent_explanation") if is_intent else signal_explanation(movie, mode, context)
    signal_chip_html = ""
    if mode != "current" and "user_score" in movie:
        signal_chip_html = '<span class="hero-signal-chip">' + html.escape(strongest_signal_label(movie)) + '</span>'

    score_label = "Intent Match" if is_intent else "MovieMind Match"
    score_pct = f"{score:.0%}"

    # Build background style separately to avoid f-string issues with data URIs
    if backdrop_uri.startswith("data:") or backdrop_uri.startswith("http"):
        bg_style = f"background-image:url('{backdrop_uri}');"
    else:
        # fallback gradient
        bg_style = f"background:{backdrop_uri};"

    hero_html = (
        '<section class="cinema-hero" style="' + bg_style + '">'
        '<div class="cinema-copy">'
        '<div class="cinema-kicker">' + html.escape(score_title) + ' · FEATURED SPOTLIGHT</div>'
        '<div style="margin:10px 0;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">'
        '<span class="match-pill">' + score_pct + ' ' + score_label + '</span>'
        + signal_chip_html +
        '</div>'
        '<div class="cinema-title">' + html.escape(title) + '</div>'
        '<div class="cinema-meta">' + html.escape(release_year) + ' &nbsp;·&nbsp; ' + html.escape(genres) + '</div>'
        '<div class="cinema-overview">' + (html.escape(overview) if overview else "Overview unavailable from TMDB.") + '</div>'
        '<div class="cinema-why"><b>Why MovieMind picked this</b><br>' + html.escape(why) + '</div>'
        '</div>'
        '</section>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)
    already_saved = watch_later_key({"movieId": movie_id}) in st.session_state.get("_watch_later_keys", set())
    trailer_col, view_col, similar_col, later_col, surprise_col, spacer = st.columns([1.2, 1.2, 1.3, 1.45, 1.25, 2.3])
    with trailer_col:
        render_play_trailer(movie, "hero", label="▶  Play Now")
    if view_col.button("Details  →", key=f"hero_details_{movie_id}"):
        select_movie_details(st.session_state, movie)
    if similar_col.button("✦  More Like This", key=f"hero_similar_{movie_id}"):
        st.session_state["more_like_movie_id"] = movie_id
        st.session_state["more_like_results"] = get_similar_movies(movie_id)
        st.session_state["more_like_source_title"] = str(movie["title"])
    if later_col.button(
        "✓ Saved · Remove" if already_saved else "+ Watch Later",
        key=f"hero_watch_later_{movie_id}",
        use_container_width=True,
    ):
        toggle_active_watch_later(movie)
        st.rerun()
    if surprise_col.button("Surprise Pick", key=f"hero_surprise_{movie_id}"):
        st.session_state["surprise_result"] = True
        clear_ai_state_for_manual_context()
        st.rerun()



TMDB_GENRE_NAMES = {
    12: "Adventure", 14: "Fantasy", 16: "Animation", 18: "Drama",
    27: "Horror", 28: "Action", 35: "Comedy", 36: "History",
    37: "Western", 53: "Thriller", 80: "Crime", 99: "Documentary",
    10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Sci-Fi",
    10751: "Children", 10752: "War", 10770: "TV Movie",
}


def make_discover_frame(tmdb_movies):
    rows = []
    for item in tmdb_movies:
        genre_names = [
            TMDB_GENRE_NAMES[genre_id]
            for genre_id in item.get("genre_ids", [])
            if genre_id in TMDB_GENRE_NAMES
        ]
        rows.append({
            "movieId": -int(item["tmdbId"]),
            "tmdbId": int(item["tmdbId"]),
            "imdbId": None,
            "title": (
                f"{item.get('title', '')} ({str(item.get('release_date', ''))[:4]})"
                if str(item.get("release_date", ""))[:4].isdigit()
                else str(item.get("title", ""))
            ),
            "genres": "|".join(genre_names),
            "release_date": str(item.get("release_date", "")),
            "overview": str(item.get("overview", "")),
            "poster_path": item.get("poster_path"),
            "backdrop_path": item.get("backdrop_path"),
            "vote_average": item.get("vote_average"),
            "_external_tmdb": True,
        })
    return pd.DataFrame(rows)


def render_movie_details():
    detail = st.session_state.get("selected_movie_details")
    if not detail:
        return
    tmdb_id = detail.get("tmdbId")
    metadata = get_movie_details(tmdb_id) if tmdb_id else None
    if not metadata:
        metadata = {}
    title, catalog_year = movie_title_year(detail.get("title", "Movie details"))
    st.markdown(f"### {html.escape(title)}")
    detail_left, detail_right = st.columns([1, 3])
    with detail_left:
        poster = get_movie_poster_image(metadata.get("poster_path") or detail.get("poster_path"))
        if poster:
            st.image(poster, use_container_width=True)
        else:
            st.markdown(POSTER_PLACEHOLDER, unsafe_allow_html=True)
    with detail_right:
        genres = str(detail.get("genres", "")).replace("|", " · ")
        release_year = str(metadata.get("release_date") or detail.get("release_date") or catalog_year or "")[:4]
        st.caption(" · ".join(value for value in (release_year, genres) if value))
        backdrop_path = metadata.get("backdrop_path") or detail.get("backdrop_path")
        if backdrop_path:
            backdrop = get_movie_backdrop_image(backdrop_path)
            if backdrop:
                st.image(backdrop, use_container_width=True)
        st.write(metadata.get("overview") or "Overview unavailable from TMDB.")
        with st.expander("▶ Watch trailer", expanded=False):
            trailer = get_movie_trailer(tmdb_id) if tmdb_id else None
            if trailer:
                trailer_url = f"https://www.youtube.com/watch?v={trailer['key']}"
                st.video(trailer_url)
                st.caption(trailer.get("name") or "Trailer")
            else:
                st.info("No YouTube trailer is listed for this movie yet.")
        if tmdb_id:
            st.markdown(f"[TMDB](https://www.themoviedb.org/movie/{int(tmdb_id)})")
        imdb_id = detail.get("imdbId")
        if imdb_id:
            try:
                imdb_id = int(imdb_id)
                if imdb_id > 0:
                    st.markdown(f"[IMDb](https://www.imdb.com/title/tt{imdb_id:07d}/)")
            except (TypeError, ValueError, OverflowError):
                pass
        recommendation = detail.get("_recommendation")
        if recommendation:
            mode = st.session_state.get("recommendation_mode", "personalized")
            context = st.session_state.get("recommendation_context", {"genres": [], "mood": None})
            if mode == "current" and recommendation.get("intent_match") is not None and pd.notna(recommendation.get("intent_match")):
                st.metric("Viewing Intent Match", f"{int(recommendation['intent_match'])}%")
                if recommendation.get("intent_explanation"):
                    st.caption(str(recommendation["intent_explanation"]))
            elif recommendation.get("hybrid_score") is not None:
                st.metric("MovieMind Match", f"{float(recommendation['hybrid_score']):.0%}")
                st.markdown(f"**Why MovieMind picked this**  \n{signal_explanation(recommendation, mode, context)}")
                if mode != "current":
                    with st.expander("View model details"):
                        st.write(f"Taste Match contribution: {0.30 * float(recommendation['user_score']):.1%}")
                        st.write(f"Movie Similarity contribution: {0.30 * float(recommendation['item_score']):.1%}")
                        st.write(f"Preference Learning contribution: {0.40 * float(recommendation['svd_score']):.1%}")
    action_cols = st.columns([1, 1, 4])
    movie_id = detail.get("movieId")
    external = detail.get("_external_tmdb")
    if action_cols[0].button("+ More Like This", key=f"selected_more_like_{tmdb_id or movie_id}", disabled=external or not movie_id or movie_id <= 0):
        st.session_state["more_like_movie_id"] = movie_id
        st.session_state["more_like_results"] = get_similar_movies(movie_id)
        st.session_state["more_like_source_title"] = detail["title"]
    entry = watch_later_movie_entry(detail)
    watch_key = watch_later_key(entry)
    already_saved = watch_key in st.session_state.get("_watch_later_keys", set())
    if entry and action_cols[1].button("✓ Saved · Remove" if already_saved else "+ Watch Later", key=f"selected_watch_later_{watch_key}"):
        toggle_active_watch_later(detail)
        st.rerun()
    if st.button("Close details", key="close_selected_movie_details"):
        st.session_state.pop("selected_movie_details", None)
        st.rerun()


# ============================================================
# APPLICATION
# ============================================================

profiles = load_profiles()

# Bind the verified account to the MovieMind profile loaded from the same store.
_auth_email = st.session_state.get("_auth_user", "").strip().lower()
_auth_profile_id = st.session_state.get("_auth_profile_id")
if _auth_email and _auth_profile_id and st.session_state.get("_auth_profile_loaded_email") != _auth_email:
    _account_profile = next(
        (profile for profile in profiles if profile.get("profile_id") == _auth_profile_id),
        None,
    )
    if _account_profile is not None:
        apply_profile(_account_profile)
        st.session_state["_auth_profile_loaded_email"] = _auth_email

# Handle quick navigation and actions from 3D carousel
query_action = st.query_params.get("action")
query_movie_id = st.query_params.get("movie_id")
query_tmdb_id = st.query_params.get("tmdb_id")
query_title = st.query_params.get("title", "")
if query_action and query_movie_id:
    try:
        qid = int(query_movie_id)
        st.query_params.clear()
        target_movie = None
        if qid in movie_lookup.index:
            target_movie = movie_lookup.loc[qid].to_dict()
            target_movie["movieId"] = qid
        elif qid < 0:
            try:
                external_tmdb_id = int(query_tmdb_id) if query_tmdb_id else -qid
            except (TypeError, ValueError, OverflowError):
                external_tmdb_id = -qid
            target_movie = {
                "movieId": qid,
                "tmdbId": external_tmdb_id,
                "_external_tmdb": True,
                "title": str(query_title or "Featured Movie"),
            }
        if target_movie:
            if query_action == "trailer":
                st.session_state["_trailer_dialog_movie"] = target_movie
            elif query_action == "details":
                select_movie_details(st.session_state, target_movie)
                if qid > 0:
                    st.session_state["selected_hero_movie_id"] = qid
            elif query_action == "watch_later":
                toggle_active_watch_later(target_movie)
            elif query_action == "more_like" and qid > 0:
                st.session_state["more_like_movie_id"] = qid
                st.session_state["more_like_results"] = get_similar_movies(qid)
                st.session_state["more_like_source_title"] = str(target_movie.get("title", "this movie"))
            st.rerun()
    except Exception:
        pass

if not st.session_state.get("profile_id") and not st.session_state.get("selected_movies"):
    default_watch_profile = next(
        (profile for profile in profiles if profile.get("profile_id") == DEFAULT_WATCH_LATER_PROFILE_ID),
        None,
    )
    if default_watch_profile:
        apply_profile(default_watch_profile)
view_names = {
    "home": "Home",
    "search": "Search",
    "discover": "Discover",
    "my-taste": "My Taste",
    "watch-later": "Watch Later",
    "model-performance": "Model Performance",
    "about": "About MovieMind",
}
active_view = st.session_state.get("active_view", "Home")
if active_view not in view_names.values():
    active_view = "Home"
    st.session_state["active_view"] = active_view


def switch_selected_profile():
    """Switch profiles as soon as the single profile picker changes."""
    selected_id = (
        st.session_state.get("topnav_profile_selector")
        or st.session_state.get("sidebar_profile_selector")
    )
    profile = next((item for item in profiles if item.get("profile_id") == selected_id), None)
    if profile:
        apply_profile(profile)


def queue_ai_prompt():
    prompt = str(st.session_state.get("ai_prompt", "")).strip()
    if prompt:
        st.session_state["_pending_ai_prompt"] = prompt
        st.session_state["active_view"] = "Home"


def open_home_for_voice():
    st.session_state["active_view"] = "Home"


def toggle_ui_theme():
    st.session_state["ui_theme"] = "light" if st.session_state.get("ui_theme", "dark") == "dark" else "dark"

_auth_user_email = st.session_state.get("_auth_user", "")
_auth_username = st.session_state.get("_auth_username") or (
    _auth_user_email.split("@")[0].replace(".", " ").title() if _auth_user_email else "Guest"
)
_auth_initial = _auth_username[:1].upper() if _auth_username else "M"
with st.container(key="streaming-navbar"):
    nav_columns = st.columns([1.75, .58, .62, .72, .85, .68, 3.1, .52, .65, .52, 1.0])
    with nav_columns[0]:
        st.markdown(
            '<div class="streaming-brand"><span class="streaming-brand-mark">M</span>'
            '<span>MOVIEMIND</span></div>',
            unsafe_allow_html=True,
        )
    nav_items = [
        ("Home", "Home", "top_nav_home"),
        ("Movies", "_movies_menu", "top_nav_movies"),
        ("Discover", "Discover", "top_nav_discover"),
        ("My Taste", "My Taste", "top_nav_taste"),
        ("My List", "Watch Later", "top_nav_watch_later"),
    ]
    for column, (label, view, key) in zip(nav_columns[1:6], nav_items):
        with column:
            if view == "_movies_menu":
                with st.popover("Movies  ⌄", help="Browse movie collections"):
                    st.markdown("### Movies")
                    st.caption("Explore a collection or find your next favorite film.")
                    movie_menu_items = [
                        ("↗  Trending Movies", "Trending", "See what films are making waves."),
                        ("✦  Discover", None, "Explore movies by genre and find hidden gems."),
                        ("♡  Popular", "Popular", "Browse movies audiences are watching."),
                        ("▷  Now Playing", "Now Playing", "Catch movies currently in theaters."),
                        ("▣  Upcoming", "Upcoming", "See what is coming to theaters soon."),
                        ("☆  Top Rated", "Highest Rated", "Explore the highest-rated movies."),
                    ]
                    menu_columns = st.columns(2)
                    for index, (item_label, category, description) in enumerate(movie_menu_items):
                        with menu_columns[index % 2]:
                            if category is None:
                                st.button(
                                    item_label,
                                    key=f"movies_menu_genres_{index}",
                                    use_container_width=True,
                                    on_click=open_genre_discovery,
                                )
                            else:
                                st.button(
                                    item_label,
                                    key=f"movies_menu_{category.casefold().replace(' ', '_')}",
                                    use_container_width=True,
                                    on_click=open_movie_category,
                                    args=(category,),
                                )
                            st.caption(description)
            else:
                st.button(
                    label,
                    key=key,
                    type="primary" if active_view == view else "secondary",
                    use_container_width=True,
                    on_click=open_genre_discovery if view == "Discover" else set_active_view,
                    args=() if view == "Discover" else (view,),
                )
    with nav_columns[6]:
        st.text_input(
            "Search by movie, mood, or genre",
            placeholder="Search by movie, mood, or genre…",
            label_visibility="collapsed",
            key="ai_prompt",
            on_change=queue_ai_prompt,
        )
    with nav_columns[7]:
        ai_submit = st.button(
            "⌕",
            key="ai_search_submit",
            type="primary",
            help="Find recommendations",
            on_click=queue_ai_prompt,
        )
    with nav_columns[8]:
        is_light_theme = st.session_state.get("ui_theme", "dark") == "light"
        st.button(
            "🌙" if is_light_theme else "☀️",
            key="toggle_ui_theme",
            help="Switch to dark mode" if is_light_theme else "Switch to light mode",
            on_click=toggle_ui_theme,
            use_container_width=True,
        )
    with nav_columns[9]:
        voice_audio = None
        with st.popover("🎙", help="Speak a movie request"):
            st.caption("Tap the microphone and say what you want to watch.")
            voice_audio = st.audio_input(
                "Record a request",
                key="ai_voice_audio_record",
                on_change=open_home_for_voice,
            )
    with nav_columns[10]:
        with st.popover(f"{_auth_initial}  ▾", help="Profile and account"):
            st.markdown(f"**{html.escape(_auth_username)}**")
            st.caption(_auth_user_email)
            profile_labels = {
                p.get("profile_id"): p.get("name", "Unnamed")
                for p in profiles if p.get("profile_id")
            }
            st.selectbox(
                "Switch profile",
                [""] + list(profile_labels),
                format_func=lambda value: "Choose profile…" if not value else profile_labels.get(value, value),
                key="topnav_profile_selector",
                on_change=switch_selected_profile,
            )
            with st.expander("Account password"):
                st.caption("Set a password to sign in without an email code next time.")
                with st.form("topnav_account_password_setup", clear_on_submit=True):
                    new_password = st.text_input("New password", type="password", key="topnav_account_new_password")
                    confirm_password = st.text_input("Confirm password", type="password", key="topnav_account_confirm_password")
                    password_submit = st.form_submit_button("Save password", use_container_width=True)
                if password_submit:
                    if len(new_password) < 8:
                        st.error("Use at least 8 characters for your password.")
                    elif len(new_password) > 128:
                        st.error("Password must be 128 characters or fewer.")
                    elif new_password != confirm_password:
                        st.error("The passwords do not match.")
                    elif set_account_password(_auth_user_email, new_password):
                        st.success("Password saved for your next sign-in.")
                    else:
                        st.error("Could not save your password. Please try again.")
            if st.button("Sign Out", key="topnav_sign_out", use_container_width=True):
                clear_auth_session(st.session_state)
                st.rerun()

if active_view == "Home":
    category_choices = [
        ("For You", "Home", None),
        ("Action", "Discover", None, "Action"),
        ("Comedy", "Discover", None, "Comedy"),
        ("Thriller", "Discover", None, "Thriller"),
        ("All Genres", "Discover", None, None),
    ]
    with st.container(key="home-category-bar"):
        category_columns = st.columns([1, 1, 1, 1.2, 1.3, 4])
        active_home_category = st.session_state.get("home_category", "For You")
        for column, (label, destination, discover_category, *genre_args) in zip(category_columns, category_choices):
            with column:
                st.button(
                    label,
                    key=f"home_category_{label.casefold().replace(' ', '_')}",
                    type="primary" if active_home_category == label else "secondary",
                    use_container_width=True,
                    on_click=set_home_category,
                    args=(label, destination, discover_category, genre_args[0] if genre_args else None),
                )

pending_trailer_movie = st.session_state.pop("_trailer_dialog_movie", None)
if pending_trailer_movie:
    show_trailer_dialog(pending_trailer_movie)

with st.sidebar:
    # ── User info from auth ──────────────────────────────────────────────
    _auth_user_email = st.session_state.get("_auth_user", "")
    _auth_username = st.session_state.get("_auth_username") or (
        _auth_user_email.split("@")[0].replace(".", " ").title() if _auth_user_email else "Guest"
    )
    _auth_initial = _auth_username[0].upper() if _auth_username else "G"

    # ── Header: Logo + User card ─────────────────────────────────────────
    st.markdown(
        '<div class="sb-header">'
        '<div class="sb-logo">'
        '<div class="sb-logo-mark">M</div>'
        '<div class="sb-logo-text">'
        '<div class="sb-logo-name">MovieMind</div>'
        '<div class="sb-logo-caption">AI Movie Discovery</div>'
        '</div></div>'
        f'<div class="sb-user-card">'
        f'<div class="sb-user-avatar">{html.escape(_auth_initial)}</div>'
        f'<div><div class="sb-user-name">{html.escape(_auth_username)}</div>'
        f'<div class="sb-user-email">{html.escape(_auth_user_email)}</div>'
        f'<span class="sb-user-badge">● Verified</span>'
        f'</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    nav_icon = {
        "Home": "⌂",
        "Search": "⌕",
        "Discover": "✦",
        "My Taste": "♡",
        "Watch Later": "＋",
        "Model Performance": "▤",
        "About MovieMind": "ⓘ",
    }

    def sidebar_nav_item(view, key):
        button_type = "primary" if active_view == view else "secondary"
        st.button(
            f"  {nav_icon[view]}   {view}",
            key=key,
            type=button_type,
            use_container_width=True,
            on_click=set_active_view,
            args=(view,),
        )

    st.markdown('<div class="sb-section"><div class="sb-section-label">DISCOVER</div></div>', unsafe_allow_html=True)
    sidebar_nav_item("Home", "nav_home")
    sidebar_nav_item("Search", "nav_search")
    sidebar_nav_item("Discover", "nav_discover")
    st.markdown('<div class="sb-section"><div class="sb-section-label">MY MOVIEMIND</div></div>', unsafe_allow_html=True)
    sidebar_nav_item("My Taste", "nav_my_taste")
    sidebar_nav_item("Watch Later", "nav_watch_later")
    st.markdown('<div class="sb-section"><div class="sb-section-label">INSIGHTS</div></div>', unsafe_allow_html=True)
    sidebar_nav_item("Model Performance", "nav_model_performance")
    st.markdown('<div class="sb-section"><div class="sb-section-label">ABOUT</div></div>', unsafe_allow_html=True)
    sidebar_nav_item("About MovieMind", "nav_about")

    # ── Profile status card ───────────────────────────────────────────────
    current_profile_pairs = st.session_state.get("profile", [])
    current_profile_record = next(
        (profile for profile in profiles if profile.get("profile_id") == st.session_state.get("profile_id")),
        None,
    )
    current_profile_name = (
        current_profile_record.get("name", "Personal taste")
        if current_profile_record
        else "Personal taste" if current_profile_pairs else "No active profile"
    )
    saved_watch_later_count = (
        len(current_profile_record.get("watch_later", []))
        if current_profile_record and isinstance(current_profile_record.get("watch_later"), list)
        else 0
    )
    profile_meta = (
        f"{len(current_profile_pairs)} rated · {saved_watch_later_count} saved"
        if current_profile_record
        else "Rate movies to personalize"
    )
    st.markdown(
        f'<div class="sidebar-profile"><div class="sidebar-profile-title">ACTIVE PROFILE</div>'
        f'<div class="sidebar-profile-name">{html.escape(current_profile_name)}</div>'
        f'<div class="sidebar-profile-meta">{html.escape(profile_meta)}</div></div>',
        unsafe_allow_html=True,
    )

    profile_labels = {p.get("profile_id"): p.get("name", "Unnamed") for p in profiles if p.get("profile_id")}
    st.selectbox(
        "Switch Profile",
        [""] + list(profile_labels),
        format_func=lambda value: "Switch profile…" if not value else profile_labels.get(value, value),
        label_visibility="collapsed",
        key="sidebar_profile_selector",
        on_change=switch_selected_profile,
    )

    with st.expander("Account password"):
        st.caption("Set a password to sign in without an email code next time.")
        with st.form("account_password_setup", clear_on_submit=True):
            new_password = st.text_input("New password", type="password", key="account_new_password")
            confirm_password = st.text_input("Confirm password", type="password", key="account_confirm_password")
            password_submit = st.form_submit_button("Save password", use_container_width=True)
        if password_submit:
            if len(new_password) < 8:
                st.error("Use at least 8 characters for your password.")
            elif len(new_password) > 128:
                st.error("Password must be 128 characters or fewer.")
            elif new_password != confirm_password:
                st.error("The passwords do not match.")
            elif not _auth_user_email:
                st.error("Sign in before setting an account password.")
            elif set_account_password(_auth_user_email, new_password):
                st.success("Password saved. You can use it the next time you sign in.")
            else:
                st.error("Could not save the password for this account. Please try again.")

    # ── Sign out button ───────────────────────────────────────────────────
    if st.button("⏻  Sign Out", use_container_width=True):
        clear_auth_session(st.session_state)
        st.rerun()

    # ── Student project card ──────────────────────────────────────────────
    st.markdown(
        '<div class="sidebar-student-info">'
        '<div style="color:#d9a978;font-weight:800;letter-spacing:1px;font-size:10px;text-transform:uppercase;margin-bottom:6px;">Project Details</div>'
        '<b>Student:</b> Yagyansh Verma<br>'
        '<b>Roll No:</b> 2400320101287<br>'
        '<b>Course:</b> B.Tech CSE · 3rd Year<br>'
        '<b>College:</b> ABES Engineering College<br>'
        '<b>Guide:</b> Mr. Vivek Chaudhary'
        '</div>',
        unsafe_allow_html=True,
    )


active_watch_later_entries = get_active_watch_later_entries(profiles)
st.session_state["_watch_later_keys"] = {
    watch_later_key(entry)
    for entry in active_watch_later_entries
    if watch_later_key(entry)
}

if active_view == "Search":
    st.markdown("### Search the MovieMind catalog")
    catalog_query = st.text_input(
        "Search movies",
        placeholder="Search by movie title",
        key="catalog_search_query",
    ).strip()
    render_movie_details()
    if catalog_query:
        matches = movies.loc[
            movies["title"].astype(str).str.contains(catalog_query, case=False, regex=False, na=False)
        ].head(30)
        search_rows = []
        for _, item in matches.iterrows():
            movie_id = int(item["movieId"])
            try:
                lookup_row = movie_lookup.loc[movie_id]
                tmdb_id = lookup_row.get("tmdbId") if hasattr(lookup_row, "get") else lookup_row["tmdbId"] if "tmdbId" in lookup_row.index else None
                imdb_id = lookup_row.get("imdbId") if hasattr(lookup_row, "get") else lookup_row["imdbId"] if "imdbId" in lookup_row.index else None
            except KeyError:
                tmdb_id = None
                imdb_id = None
            genres_val = item.get("genres") if hasattr(item, "get") else item["genres"]
            search_rows.append({
                "movieId": movie_id,
                "tmdbId": tmdb_id,
                "imdbId": imdb_id,
                "title": str(item["title"]),
                "genres": "" if pd.isna(genres_val) else str(genres_val),
            })
        if search_rows:
            render_movie_row("Catalog results", pd.DataFrame(search_rows), "search", {}, "catalog_search")
        else:
            st.info("No movies in the local catalog match that title.")
    else:
        st.caption("Searches MovieMind's local movie catalog.")
    if st.session_state.get("more_like_results") is not None:
        render_movie_row(
            f"More Like {st.session_state.get('more_like_source_title', 'this movie')}",
            st.session_state["more_like_results"],
            "similarity",
            {},
            "search_more_like",
        )
    st.markdown('<div class="footer-text">MovieMind | Personalized Hybrid Movie Recommendation System</div>', unsafe_allow_html=True)
    st.stop()

if active_view == "Discover":
    if st.session_state.get("discover_mode", "genres") == "category":
        selected_category = st.session_state.get("discover_category", "Trending")
        category_title = "Top Rated" if selected_category == "Highest Rated" else selected_category
        st.markdown("## Movies")
        st.caption("Explore current movie collections")
        title_col, page_col = st.columns([4, 1.1], vertical_alignment="center")
        with title_col:
            st.markdown(f"### {html.escape(category_title)} Movies")
        with page_col:
            category_page = int(st.session_state.get("discover_page", 1))
            previous_col, count_col, next_col = st.columns([1, 1.2, 1])
            if previous_col.button("←", key="movie_category_prev", disabled=category_page <= 1, use_container_width=True):
                st.session_state["discover_page"] = max(1, category_page - 1)
                st.rerun()
            count_col.markdown(f'<div class="discover-page-count"><b>{category_page}</b> / 6</div>', unsafe_allow_html=True)
            if next_col.button("→", key="movie_category_next", disabled=category_page >= 6, use_container_width=True):
                st.session_state["discover_page"] = min(6, category_page + 1)
                st.rerun()
        render_movie_details()
        with st.spinner(f"Loading {category_title.lower()} movies..."):
            category_movies = get_discover_movies(selected_category, category_page)
        if category_movies:
            category_frame = make_discover_frame(category_movies)
            preload_movie_media(category_frame["tmdbId"].tolist())
            movie_rows = list(category_frame.iterrows())
            for start in range(0, len(movie_rows), 5):
                with st.container(key=f"movie-category-row-{selected_category.casefold().replace(' ', '-')}-{category_page}-{start}"):
                    card_columns = st.columns(5)
                    for column, (_, movie) in zip(card_columns, movie_rows[start:start + 5]):
                        with column:
                            render_movie_card(movie, "discover", {}, f"category_{selected_category}_{category_page}")
        else:
            st.info("These movies are temporarily unavailable. Check your TMDB configuration and try again.")
        st.markdown('<div class="footer-text">MovieMind | Personalized Movie Discovery</div>', unsafe_allow_html=True)
        st.stop()

    genre_names = ["Action", "Comedy", "Drama", "Romance", "Science Fiction", "Thriller", "Horror", "Animation", "Crime"]
    st.markdown("## Genres")
    st.caption("Explore movies by genre")

    selected_genre = st.radio(
        "Choose a genre",
        genre_names,
        horizontal=True,
        key="discover_genre",
        label_visibility="collapsed",
        on_change=lambda: st.session_state.update(discover_page=1),
    )
    st.markdown('<div class="discover-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="row-caption">Browse top movies in <b>{html.escape(selected_genre)}</b></div>', unsafe_allow_html=True)
    heading_col, pager_col = st.columns([4, 1.1], vertical_alignment="center")
    with heading_col:
        st.markdown(f"### {html.escape(selected_genre)} Movies")
    with pager_col:
        page = int(st.session_state.get("discover_page", 1))
        prev_col, count_col, next_col = st.columns([1, 1.2, 1])
        if prev_col.button("←", key="discover_prev", disabled=page <= 1, use_container_width=True):
            st.session_state["discover_page"] = max(1, page - 1)
            st.rerun()
        count_col.markdown(f'<div class="discover-page-count"><b>{page}</b> / 6</div>', unsafe_allow_html=True)
        if next_col.button("→", key="discover_next", disabled=page >= 6, use_container_width=True):
            st.session_state["discover_page"] = min(6, page + 1)
            st.rerun()

    with st.expander("⚑  Filters", expanded=False):
        filter_cols = st.columns([1.2, 1.2, 2])
        with filter_cols[0]:
            min_rating = st.slider("Minimum rating", 0.0, 10.0, 0.0, 0.5, key="discover_min_rating")
        with filter_cols[1]:
            year_range = st.slider("Release years", 1950, datetime.now().year + 2, (1950, datetime.now().year + 2), key="discover_year_range")
        with filter_cols[2]:
            st.caption("Filters apply to the selected genre and page.")

    render_movie_details()
    with st.spinner(f"Loading {selected_genre.lower()} movies..."):
        discover_movies, _total_pages = get_tmdb_genre_movies(selected_genre, page)
    if discover_movies:
        genre_frame = make_discover_frame(discover_movies)
        if min_rating > 0:
            genre_frame = genre_frame.loc[pd.to_numeric(genre_frame["vote_average"], errors="coerce").fillna(0) >= min_rating]
        years = pd.to_numeric(genre_frame["title"].map(lambda value: movie_title_year(value)[1]), errors="coerce")
        genre_frame = genre_frame.loc[years.between(year_range[0], year_range[1])]
        if genre_frame.empty:
            st.info("No movies on this page match those filters. Try a lower rating or a wider year range.")
        else:
            preload_movie_media(genre_frame["tmdbId"].tolist())
            rows = list(genre_frame.iterrows())
            for start in range(0, len(rows), 5):
                with st.container(key=f"discover-genre-row-{selected_genre.casefold().replace(' ', '-')}-{page}-{start}"):
                    cards = st.columns(5)
                    for column, (_, movie) in zip(cards, rows[start:start + 5]):
                        with column:
                            render_movie_card(movie, "discover", {}, f"genre_{selected_genre}_{page}")
    else:
        st.info("Genre movies are temporarily unavailable. Check your TMDB configuration and try again.")
    st.markdown('<div class="footer-text">MovieMind | Personalized Movie Discovery</div>', unsafe_allow_html=True)
    st.stop()

if active_view == "Watch Later":
    st.markdown("### Watch Later")
    render_movie_details()
    watch_frame = watch_later_frame(active_watch_later_entries)
    if watch_frame.empty:
        st.markdown(
            '<div class="empty-state-box">'
            '<div class="empty-state-icon">🍿</div>'
            '<div class="empty-state-title">Your Watch Later List is Empty</div>'
            '<div class="empty-state-sub">Movies you save from recommendations, search, discover, or the 3D arc carousel will appear here. Click <b>+ Watch Later</b> on any title to save it.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        render_movie_row(
            "Saved for later",
            watch_frame,
            "watch_later",
            {},
            "watch_later",
        )
    if st.session_state.get("more_like_results") is not None:
        render_movie_row(
            f"More Like {st.session_state.get('more_like_source_title', 'this movie')}",
            st.session_state["more_like_results"],
            "similarity",
            {},
            "watch_later_more_like",
        )
    st.markdown('<div class="footer-text">MovieMind | Personalized Movie Discovery</div>', unsafe_allow_html=True)
    st.stop()

if active_view == "My Taste":
    current_profile = st.session_state.get("profile", [])
    taste_rows = [
        {"Movie": movie_lookup.loc[mid, "title"], "Rating": rating, "Genres": str(movie_lookup.loc[mid, "genres"]).replace("|", " · ")}
        for mid, rating in current_profile if mid in movie_lookup.index
    ]
    taste_frame = pd.DataFrame(taste_rows)
    st.markdown("### My Taste")
    if taste_frame.empty:
        st.info("Your rating history will appear here after you rate movies from Home.")
    else:
        summary_col, genre_col = st.columns([1, 2])
        with summary_col:
            st.metric("Movies rated", len(taste_frame))
            st.metric("Average rating", f"{taste_frame['Rating'].mean():.2f} / 5")
        with genre_col:
            st.markdown("#### Your genre profile")
            genre_frame = get_genre_profile(current_profile)
            if not genre_frame.empty:
                st.dataframe(genre_frame, use_container_width=True, hide_index=True)
        st.markdown("#### Rating history")
        st.dataframe(taste_frame, use_container_width=True, hide_index=True)
        st.download_button("Download rating history as CSV", taste_frame.to_csv(index=False).encode("utf-8"), "moviemind_rating_history.csv", "text/csv")
    current_context = st.session_state.get("recommendation_context", {"genres": [], "mood": None})
    with st.expander("Current viewing preferences"):
        st.write(f"Genres: {', '.join(current_context.get('genres', [])) or 'Any'}")
        st.write(f"Mood: {current_context.get('mood') or 'Any'}")
    st.stop()

if active_view == "Model Performance":
    st.markdown("### How MovieMind Recommends")
    signal_columns = st.columns(3)
    for column, (name, weight, description) in zip(signal_columns, [
        ("Taste Match", "30%", "Patterns from users with similar rating behavior."),
        ("Movie Similarity", "30%", "Relationships between movies based on rating patterns."),
        ("Preference Learning", "40%", "Latent user and movie preferences learned with SVD."),
    ]):
        with column:
            with st.container(border=True):
                st.metric(name, weight)
                st.caption(description)
    st.caption("Evaluation results come from a separate experiment; they are not live accuracy measurements for your profile.")
    for column, (name, value, description) in zip(st.columns(4), [
        ("RMSE", "0.8352", "Prediction error; lower is better."),
        ("MAE", "0.6372", "Average absolute prediction error."),
        ("Precision@10", "53.37%", "Relevant movies within the top 10."),
        ("Recall@10", "74.77%", "Relevant movies retrieved in the top 10."),
    ]):
        with column:
            with st.container(border=True):
                st.metric(name, value)
                st.caption(description)
    with st.expander("Evaluation setup"):
        st.write("A reproducible sample of 1,000,000 ratings was split 80/20 into training and held-out test data. Ratings of 4.0 or higher counted as relevant. Surprise SVD used 100 factors, 20 epochs, learning rate 0.005, regularization 0.02, and random_state 42.")
    st.stop()

if active_view == "About MovieMind":
    st.markdown("### About MovieMind")
    st.write("MovieMind offers mood and genre discovery, taste-based recommendations from your ratings, or a combination of both.")
    st.markdown("#### Recommendation approach")
    st.write("The personalized hybrid combines Taste Match (30%), Movie Similarity (30%), and Preference Learning (40%). Gemini maps natural-language requests to supported genre and mood signals; the existing recommendation engine ranks the results. TMDB provides discovery metadata and images.")
    st.markdown("#### Your data")
    st.write(f"The local movie dataset contains {len(user_ids):,} users, {len(movies):,} movies, and {len(ratings):,} ratings. Account profiles and saved ratings are stored in the configured account database.")
    st.markdown("#### Project")
    st.write("MovieMind · B.Tech Computer Science & Engineering · ABES Engineering College")
    st.write("Movie metadata and images are provided by TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.")
    st.stop()

active_intent_text = None
pending_prompt = st.session_state.pop("_pending_ai_prompt", None)
if ai_submit:
    active_intent_text = str(st.session_state.get("ai_prompt", "")).strip()
    st.session_state["active_view"] = "Home"
    active_view = "Home"
elif pending_prompt:
    active_intent_text = pending_prompt

if voice_audio is not None:
    raw_audio_bytes = voice_audio.getvalue() if hasattr(voice_audio, "getvalue") else voice_audio.read()
    if raw_audio_bytes:
        audio_hash = f"{len(raw_audio_bytes)}_{hash(raw_audio_bytes[:256])}"
        if audio_hash != st.session_state.get("_last_processed_voice_hash"):
            st.session_state["_last_processed_voice_hash"] = audio_hash
            if transcribe_speech_to_text is None:
                st.markdown('<div class="ai-error-state">Voice transcription is temporarily unavailable. Please try a typed request.</div>', unsafe_allow_html=True)
            else:
                try:
                    with st.spinner("Listening to your request..."):
                        spoken_text = transcribe_speech_to_text(raw_audio_bytes, mime_type=getattr(voice_audio, "type", "audio/wav"))
                    if not spoken_text:
                        st.warning("No clear speech was detected. Please try again.")
                    else:
                        st.session_state["_last_spoken_prompt"] = spoken_text
                        active_intent_text = spoken_text.strip()
                        st.session_state["active_view"] = "Home"
                        active_view = "Home"
                except Exception as exc:
                    if is_gemini_rate_limit_error(exc):
                        st.markdown('<div class="ai-error-state ai-429-state">MovieMind AI has temporarily reached its API request limit. Please try again later.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="ai-error-state">Voice transcription encountered an issue. Please try a typed request.</div>', unsafe_allow_html=True)

ai_context = None
if active_intent_text is not None:
    clear_ai_request_state(clear_recommendations=True)
    st.session_state["recommendation_source"] = "manual"
    st.session_state["ai_request_succeeded"] = False
    if not active_intent_text.strip():
        st.markdown('<div class="ai-error-state">Enter or speak a movie request to ask MovieMind AI.</div>', unsafe_allow_html=True)
    else:
        fast_intent = parse_movie_intent_fast(active_intent_text.strip())
    if active_intent_text.strip() and fast_intent is None and parse_movie_intent is None:
        st.session_state["ai_request_error"] = "unavailable"
        st.markdown('<div class="ai-error-state">MovieMind AI is temporarily unavailable. You can continue using manual preferences below.</div>', unsafe_allow_html=True)
    elif active_intent_text.strip():
        try:
            if fast_intent is not None:
                parsed_intent = fast_intent
            else:
                with st.spinner("Understanding your movie request..."):
                    parsed_intent = parse_movie_intent(active_intent_text.strip())
            raw_genres = list(parsed_intent.genres or [])
            mapped_genres = normalize_ai_genres(raw_genres)
            if re.search(
                r"\b(?:rom[\s-]*com|romantic[\s-]+comed(?:y|ies))\b",
                active_intent_text,
                flags=re.IGNORECASE,
            ):
                mapped_genres = [
                    genre for genre in mapped_genres
                    if genre not in {"Comedy", "Romance", "Rom-Com"}
                ]
                mapped_genres.insert(0, "Rom-Com")
            mapped_mood = normalize_ai_mood(parsed_intent.mood)
            unsupported_genres = [genre for genre in raw_genres if not normalize_ai_genres([genre])]
            reference_movie = (parsed_intent.reference_movie or "").strip() or None
            reference_id = find_local_movie_id(reference_movie)
            ai_context = {"genres": mapped_genres, "mood": mapped_mood}
            summary = {
                **ai_context,
                "reference_movie": reference_movie,
                "keywords": [str(word) for word in (parsed_intent.keywords or []) if str(word).strip()],
                "unsupported_genres": unsupported_genres,
                "unsupported_mood": bool(parsed_intent.mood and not mapped_mood),
                "source_text": active_intent_text,
            }
            st.session_state["ai_intent_summary"] = summary
            st.session_state["ai_request_succeeded"] = True
            if reference_id is not None:
                st.session_state["ai_reference_movie_id"] = reference_id
                st.session_state["ai_reference_results"] = get_similar_movies(reference_id)
                st.session_state["ai_reference_missing"] = False
            else:
                st.session_state.pop("ai_reference_movie_id", None)
                st.session_state.pop("ai_reference_results", None)
                st.session_state["ai_reference_missing"] = bool(reference_movie)
            intent_chips = [*mapped_genres]
            if mapped_mood:
                intent_chips.append(mapped_mood)
            intent_chips.extend(summary["keywords"])
            if reference_movie:
                intent_chips.append(f"Reference: {reference_movie}")
            status_markup = "".join(
                f'<span class="intent-chip">{html.escape(str(chip))}</span>'
                for chip in intent_chips
            ) or '<span class="intent-chip">No supported genre or mood signal</span>'
            st.markdown(
                f'<div class="ai-status"><span class="ai-status-label">MOVIEMIND UNDERSTANDS</span>{status_markup}</div>',
                unsafe_allow_html=True,
            )
            if summary["unsupported_genres"] or summary["unsupported_mood"]:
                st.caption("Some intent terms were not available in MovieMind's context and were skipped.")
        except Exception as exc:
            ai_context = None
            clear_ai_request_state(clear_recommendations=True)
            st.session_state["recommendation_source"] = "manual"
            st.session_state["ai_request_succeeded"] = False
            error_markup = (
                "MovieMind AI has temporarily reached its API request limit. You can continue using manual preferences below."
                if is_gemini_rate_limit_error(exc)
                else "MovieMind AI is temporarily unavailable. You can continue using manual preferences below."
            )
            st.session_state["ai_request_error"] = "rate_limit" if is_gemini_rate_limit_error(exc) else "unavailable"
            error_class = "ai-error-state ai-429-state" if is_gemini_rate_limit_error(exc) else "ai-error-state"
            st.markdown(f'<div class="{error_class}">{html.escape(error_markup)}</div>', unsafe_allow_html=True)

elif st.session_state.get("ai_request_succeeded") and st.session_state.get("recommendation_source") == "ai":
    summary = st.session_state.get("ai_intent_summary", {})
    intent_chips = [*summary.get("genres", [])]
    if summary.get("mood"):
        intent_chips.append(summary["mood"])
    intent_chips.extend(summary.get("keywords", []))
    if summary.get("reference_movie"):
        intent_chips.append(f"Reference: {summary['reference_movie']}")
    status_markup = "".join(
        f'<span class="intent-chip">{html.escape(str(chip))}</span>'
        for chip in intent_chips
    ) or '<span class="intent-chip">No supported genre or mood signal</span>'
    st.markdown(
        f'<div class="ai-status"><span class="ai-status-label">MOVIEMIND UNDERSTANDS</span>{status_markup}</div>',
        unsafe_allow_html=True,
    )

with st.expander("Refine your search", expanded=False):
    st.markdown('<div class="manual-refine-label">REFINE YOUR SEARCH · MANUAL PREFERENCES</div>', unsafe_allow_html=True)
    context_genre_col, context_mood_col, context_era_col = st.columns([2, 1, 1])
    with context_genre_col:
        selected_genres = st.multiselect("Genre Preference", options=GENRE_OPTIONS, placeholder="Any", key="context_genres", on_change=clear_ai_state_for_manual_context)
    with context_mood_col:
        selected_mood_option = st.selectbox("Mood", options=["Any", *MOOD_GENRES], key="context_mood", on_change=clear_ai_state_for_manual_context)
    with context_era_col:
        selected_release_decade = st.selectbox("Release era", DECADE_OPTIONS, key="context_release_decade", on_change=clear_ai_state_for_manual_context)
    result_control_col, sort_control_col = st.columns([1, 2])
    with result_control_col:
        recommendation_limit = st.select_slider("Recommendations", options=[5, 10, 20], value=10, key="recommendation_limit")
    with sort_control_col:
        recommendation_sort = st.selectbox("Sort results", ["Best match", "Newest first", "Title A–Z"], key="recommendation_sort")
selected_mood = None if selected_mood_option == "Any" else selected_mood_option

with st.expander("Personalize using my taste", expanded=False):
    st.caption("Rate at least 1 movie (up to 10). Your profile uses the existing 30/30/40 hybrid recommender.")
    st.text_input("Profile name", key="profile_name", placeholder="Enter your name")
    selected_movies = st.multiselect("Search the MovieMind catalog", options=list(movie_label_to_id), max_selections=10, placeholder="Type a title, for example Interstellar (2014)...", key="selected_movies", help="Searches the local MovieMind movie dataset. Movie IDs distinguish duplicate titles.")
    if selected_movies:
        st.markdown("#### Rate your movies")
        st.caption("Each rating uses half-star steps from 0.5 to 5.0.")
        for start in range(0, len(selected_movies), 2):
            columns = st.columns(2)
            for pos, column in enumerate(columns):
                idx = start + pos
                if idx >= len(selected_movies):
                    continue
                label = selected_movies[idx]
                movie_id = movie_label_to_id[label]
                movie_info = movie_lookup.loc[movie_id]
                with column:
                    poster_col, details_col = st.columns([1, 2])
                    metadata = get_movie_details(movie_info.get("tmdbId"))
                    poster = get_movie_poster_image((metadata or {}).get("poster_path"))
                    poster_col.image(poster or POSTER_PLACEHOLDER, use_container_width=True)
                    title, year = movie_title_year(movie_info["title"])
                    details_col.markdown(f"**{title}**")
                    details_col.caption(year)
                    details_col.caption(str(movie_info["genres"]).replace("|", " · "))
                    st.slider("Your rating (0.5-5.0)", 0.5, 5.0, 4.0, 0.5, key=f"taste_rating_{movie_id}")

profile_preview = [(movie_label_to_id[label], float(st.session_state.get(f"taste_rating_{movie_label_to_id[label]}", 4.0))) for label in selected_movies]

# Auto-save profile on every change to prevent data loss on tab switch
_autosave_name = st.session_state.get("profile_name", "").strip()
_autosave_key = None
if profile_preview and _autosave_name:
    _pid = st.session_state.get("profile_id") or f"MM-{uuid.uuid4().hex[:6].upper()}"
    _autosave_key = (_pid, _autosave_name, tuple(sorted(profile_preview)))
    if st.session_state.get("_last_autosave_key") != _autosave_key:
        _now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _profs = load_profiles()
        _prev = next((p for p in _profs if p.get("profile_id") == _pid), None)
        _rec = {
            "profile_id": _pid,
            "name": _autosave_name,
            "movies": [{"movieId": mid, "rating": rating} for mid, rating in profile_preview],
            "watch_later": list(_prev.get("watch_later", [])) if _prev else [],
            "created_at": _prev.get("created_at", _now) if _prev else _now,
            "updated_at": _now,
        }
        if _prev and _prev.get("email"):
            _rec["email"] = _prev["email"]
        _profs = [p for p in _profs if p.get("profile_id") != _pid] + [_rec]
        save_profiles(_profs)
        st.session_state["profile_id"] = _pid
        st.session_state["profile"] = profile_preview
        st.session_state["_last_autosave_key"] = _autosave_key
elif profile_preview:
    st.session_state["profile"] = profile_preview

if profile_preview:

    genre_preview = get_genre_profile(profile_preview)
    leading_genre = genre_preview.iloc[0]["Genre"] if not genre_preview.empty else "Not available"
    m1, m2, m3 = st.columns(3)
    m1.metric("Movies Selected", len(profile_preview))
    m2.metric("Average Rating", f"{np.mean([r for _, r in profile_preview]):.1f} / 5")
    m3.metric("Leading Genre", leading_genre)

action_cols = st.columns([2, 1, 1, 1])
generate = action_cols[0].button("Show Recommendations", type="primary", use_container_width=True, on_click=clear_ai_state_for_manual_context)
surprise = action_cols[1].button("Surprise Me", use_container_width=True, on_click=clear_ai_state_for_manual_context)
_is_saved = bool(_autosave_key) and st.session_state.get("_last_autosave_key") == _autosave_key
save_profile = action_cols[2].button("✓ Saved" if _is_saved else "Save Profile", use_container_width=True)
reset = action_cols[3].button("Reset Profile", use_container_width=True, on_click=reset_profile_state)

if save_profile:
    if not st.session_state.get("profile_name", "").strip():
        st.warning("Enter a profile name before saving.")
    elif not 1 <= len(profile_preview) <= 10:
        st.warning("Select between 1 and 10 movies before saving.")
    else:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        profile_id = st.session_state.get("profile_id") or f"MM-{uuid.uuid4().hex[:6].upper()}"
        record = {"profile_id": profile_id, "name": st.session_state["profile_name"].strip(), "movies": [{"movieId": mid, "rating": rating} for mid, rating in profile_preview]}
        previous = next((p for p in profiles if p.get("profile_id") == profile_id), None)
        if previous:
            record["watch_later"] = list(previous.get("watch_later", []))
            if previous.get("email"):
                record["email"] = previous["email"]
        record["created_at"] = previous.get("created_at", now) if previous else now
        record["updated_at"] = now
        profiles = [p for p in profiles if p.get("profile_id") != profile_id] + [record]
        save_profiles(profiles)
        st.session_state["profile_id"] = profile_id
        st.session_state["profile"] = profile_preview
        st.session_state["_last_autosave_key"] = (
            profile_id,
            record["name"],
            tuple(sorted(profile_preview)),
        )
        st.success("Profile saved to your account.")

effective_genres = ai_context["genres"] if ai_context is not None else list(selected_genres)
effective_mood = ai_context["mood"] if ai_context is not None else selected_mood

if generate or surprise or ai_context is not None:
    active_saved_profile = st.session_state.get("profile", [])
    use_profile = profile_preview if 1 <= len(profile_preview) <= 10 else active_saved_profile
    valid_profile = 1 <= len(use_profile) <= 10
    has_context = bool(effective_genres or effective_mood)
    source = "ai" if ai_context is not None else "manual"
    if not valid_profile and not has_context:
        st.session_state["recommendations"] = pd.DataFrame()
        st.session_state["recommendation_source"] = source
        st.session_state["recommendation_context"] = {"genres": [], "mood": None}
        if st.session_state.get("ai_reference_movie_id") is None:
            st.warning("Choose a genre or mood, rate at least 1 movie, or describe a movie intent for MovieMind AI.")
        else:
            st.info("No local genre or mood signal was found. MovieMind can still show catalog similarity for a recognized reference movie.")
    else:
        if valid_profile:
            mode = "combined" if has_context else "personalized"
            with st.spinner("Finding movies with your MovieMind profile..."):
                result_frame = get_recommendations(
                    use_profile,
                    selected_genres=effective_genres,
                    selected_mood=effective_mood,
                    selected_release_decade=selected_release_decade,
                )
            st.session_state["profile"] = use_profile
        else:
            mode = "current"
            with st.spinner("Finding movies for your viewing intent..."):
                result_frame = get_current_viewing_recommendations(
                    selected_genres=effective_genres,
                    selected_mood=effective_mood,
                    selected_release_decade=selected_release_decade,
                )
            st.session_state["profile"] = []
        st.session_state["recommendations"] = result_frame.head(1) if surprise else result_frame
        st.session_state["recommendation_context"] = {"genres": list(effective_genres), "mood": effective_mood}
        st.session_state["recommendation_release_decade"] = selected_release_decade
        st.session_state["recommendation_mode"] = mode
        st.session_state["recommendation_source"] = source
        st.session_state["recommendation_profile"] = list(use_profile) if valid_profile else []
        st.session_state["surprise_result"] = bool(surprise)
        st.session_state.pop("more_like_movie_id", None)
        st.session_state.pop("more_like_results", None)
        st.session_state.pop("selected_movie_details", None)
        st.session_state.pop("selected_hero_movie_id", None)
        if result_frame.empty:
            st.info("No recommendations matched that release era. Try another era or choose Any era.")
        elif surprise:
            st.success("Here's a MovieMind recommendation based on your current profile and viewing intent.")
        elif source == "ai":
            st.success("MovieMind translated your request into its existing viewing context and recommendation system.")
        else:
            st.success("Your recommendations are ready.")

stored_profile = st.session_state.get("profile", [])
recommendations = st.session_state.get("recommendations", pd.DataFrame())
generated_context = st.session_state.get("recommendation_context", {"genres": [], "mood": None})
generated_mode = st.session_state.get("recommendation_mode", "personalized")
current_profile_for_recommendations = (
    profile_preview if 1 <= len(profile_preview) <= 10 else stored_profile
)
if (
    (generated_mode != "current" and (
        st.session_state.get("recommendation_profile", [])
        != list(current_profile_for_recommendations)
    ))
    or (st.session_state.get("recommendation_source") != "ai" and (
        generated_context.get("genres", []) != list(selected_genres)
        or generated_context.get("mood") != selected_mood
    ))
    or st.session_state.get("recommendation_release_decade", "Any era") != selected_release_decade
):
    recommendations = pd.DataFrame()
if not recommendations.empty and st.session_state.get("surprise_result"):
    recommendations = recommendations.head(1)
elif not recommendations.empty:
    if recommendation_sort == "Newest first":
        recommendations = recommendations.assign(
            _release_year=pd.to_numeric(
                recommendations["title"].map(lambda title: movie_title_year(title)[1]),
                errors="coerce",
            )
        ).sort_values(["_release_year", "final_score"], ascending=[False, False], na_position="last").drop(columns="_release_year")
    elif recommendation_sort == "Title A–Z":
        recommendations = recommendations.sort_values("title", key=lambda values: values.str.casefold())
    recommendations = recommendations.head(int(recommendation_limit))
if active_view == "Home":
    current_mode = st.session_state.get("recommendation_mode", "personalized")
    recommendation_source = st.session_state.get("recommendation_source", "manual")
    current_context = st.session_state.get("recommendation_context", {"genres": [], "mood": None})
    context_genres = current_context.get("genres", [])
    context_mood = current_context.get("mood")
    has_context = bool(context_genres or context_mood)
    reference_results = st.session_state.get("ai_reference_results", pd.DataFrame())
    reference_similarities = (
        dict(zip(reference_results["movieId"].astype(int), reference_results["similarity"].astype(float)))
        if isinstance(reference_results, pd.DataFrame)
        and not reference_results.empty
        and {"movieId", "similarity"}.issubset(reference_results.columns)
        else {}
    )
    if not recommendations.empty and (has_context or reference_similarities):
        reference_title = st.session_state.get("ai_intent_summary", {}).get("reference_movie")
        recommendations = with_intent_match_columns(
            recommendations,
            context_genres,
            context_mood,
            reference_similarities,
            reference_title,
        )
    if recommendation_source == "ai":
        st.markdown("### MovieMind understood")
        summary = st.session_state.get("ai_intent_summary", {})
        chips = [*summary.get("genres", [])]
        if summary.get("mood"):
            chips.append(summary["mood"])
        if summary.get("reference_movie"):
            chips.append(f"Reference: {summary['reference_movie']}")
        chips.extend(summary.get("keywords", []))
        st.markdown("".join(f'<span class="intent-chip">{html.escape(str(chip))}</span>' for chip in chips) or '<span class="intent-chip">No supported genre or mood signal</span>', unsafe_allow_html=True)
        if summary.get("unsupported_genres") or summary.get("unsupported_mood"):
            st.caption("Some intent terms were not available in the local MovieMind context and were skipped.")
        if st.session_state.get("ai_reference_missing"):
            st.info("Reference movie not found in the MovieMind catalog. The rest of your request is still active.")
    if recommendations.empty:
        st.markdown('<div class="row-heading">Your movie night, your way</div><div class="row-caption">Start with a natural-language request, choose a genre or mood, or build a taste profile.</div>', unsafe_allow_html=True)
        if (
            not st.session_state.get("recommendations", pd.DataFrame()).empty
            and st.session_state.get("recommendation_release_decade", "Any era") != selected_release_decade
        ):
            st.info("Your release-era filter changed. Choose Show Recommendations to refresh the list.")
        # Streaming-style browse page with a featured film and horizontal rails.
        trending = get_discover_movies("Trending")
        if trending:
            trending_frame = make_discover_frame(trending)
            render_streaming_feature(trending_frame.iloc[0])
            render_movie_details()
        render_because_you_watched(current_context, "because_you_watched_browse")

        if recommendation_source == "ai" and st.session_state.get("ai_reference_movie_id") is not None:
            reference_id = int(st.session_state["ai_reference_movie_id"])
            reference_title = movie_lookup.loc[reference_id, "title"]
            reference_results = st.session_state.get("ai_reference_results", pd.DataFrame())
            reference_results = with_intent_match_columns(
                reference_results,
                [],
                None,
                dict(zip(reference_results["movieId"].astype(int), reference_results["similarity"].astype(float)))
                if not reference_results.empty else {},
                str(reference_title),
            )
            render_movie_rail(f"Because you mentioned {reference_title}", reference_results, "current", current_context, "ai_reference_empty")
    else:
        st.caption("MovieMind Match is a normalized recommendation score, not a probability." if current_mode != "current" else "Viewing Intent Match summarizes verified catalog signals. It is separate from recommendation ranking.")
        if current_context.get("genres") or current_context.get("mood"):
            st.markdown("".join(f'<span class="intent-chip">{html.escape(str(chip))}</span>' for chip in [*context_genres, *([context_mood] if context_mood else [])]), unsafe_allow_html=True)
        hero_id = st.session_state.get("selected_hero_movie_id")
        hero_rows = recommendations.loc[recommendations["movieId"].astype(int) == int(hero_id)] if hero_id is not None else pd.DataFrame()
        hero_movie = hero_rows.iloc[0] if not hero_rows.empty else recommendations.iloc[0]
        render_hero(hero_movie, current_mode, current_context)

        render_movie_details()
        render_because_you_watched(current_context, "because_you_watched_home")


        if recommendation_source == "ai" and st.session_state.get("ai_reference_movie_id") is not None:
            reference_id = int(st.session_state["ai_reference_movie_id"])
            reference_title = movie_lookup.loc[reference_id, "title"]
            reference_results = st.session_state.get("ai_reference_results", pd.DataFrame())
            reference_results = with_intent_match_columns(
                reference_results,
                [],
                None,
                dict(zip(reference_results["movieId"].astype(int), reference_results["similarity"].astype(float)))
                if not reference_results.empty else {},
                str(reference_title),
            )
            render_movie_rail(f"Because you mentioned {reference_title}", reference_results, "current", current_context, "ai_reference")

        if st.session_state.get("surprise_result"):
            main_title = "Your Surprise Pick"
        elif recommendation_source == "ai":
            main_title = "Top Picks for Your Request"
        elif current_mode == "current":
            main_title = "Current Viewing"
        else:
            main_title = "Top Picks For You"
        additional_picks = recommendations.iloc[1:].copy()
        if not additional_picks.empty:
            render_movie_rail(
                main_title,
                additional_picks,
                current_mode,
                current_context,
                "primary",
            )

        more_like_id = st.session_state.get("more_like_movie_id")
        if more_like_id is not None:
            more_like_results = st.session_state.get("more_like_results", pd.DataFrame())
            source_title = st.session_state.get("more_like_source_title") or movie_lookup.loc[more_like_id, "title"]
            render_movie_rail(f"More Like {source_title}", more_like_results, "similarity", current_context, "more_like")

    # Keep secondary discovery rails on the browse landing page. Once the user
    # has a request, focus rendering on those picks instead of loading three
    # unrelated TMDB rails on the same rerun.
    if recommendations.empty:
        for category, title, key in (
            ("Trending", "Trending Now", "trending_now"),
            ("Popular", "Popular on MovieMind", "popular_movies"),
            ("Highest Rated", "Top Rated Movies", "top_rated_movies"),
        ):
            category_movies = get_discover_movies(category)
            if category_movies:
                category_frame = make_discover_frame(category_movies)
                if category == "Trending":
                    category_frame = category_frame.iloc[1:]
                render_movie_rail(title, category_frame, "discover", {}, key)

st.divider()
st.markdown('<div class="footer-text">MovieMind | Personalized Hybrid Movie Recommendation System</div>', unsafe_allow_html=True)
