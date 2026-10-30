"""MovieMind 3D Circular Arc Carousel Component.

Renders an interactive 3D circular/arc carousel of movie cards where:
- Cards are arranged in a curved 3D circular arc around the viewer.
- Cursor/mouse movement smoothly rotates the carousel in real time (client-side 60 FPS).
- The centered movie comes forward, scales up, gains prominent MovieMind glow, and reveals action buttons.
- Clicking any card rotates it to center.
- Left/Right arrows, touch swipe, and keyboard navigation provide full accessibility.
- Actions (Details, Watch Later, More Like This) communicate via parent URL query parameters.
"""

from __future__ import annotations

import base64
import html
import re
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from tmdb_helper import POSTER_PLACEHOLDER, get_movie_details, get_movie_poster_url, get_movie_trailer, preload_movie_media
from watch_later import watch_later_key, watch_later_movie_entry


def _card_image_uri(image_bytes: bytes | None) -> str:
    """Encode poster bytes as data URI with fallback."""
    if image_bytes:
        return f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('ascii')}"
    return f"data:image/svg+xml;base64,{base64.b64encode(POSTER_PLACEHOLDER.encode('utf-8')).decode('ascii')}"


def _clean_title_year(title_str: str) -> tuple[str, str]:
    """Extract clean title and 4-digit release year."""
    match = re.search(r"\s*\((\d{4})\)\s*$", str(title_str or ""))
    if match:
        year = match.group(1)
        cleaned = title_str[: match.start()].strip()
        return cleaned or str(title_str), year
    return str(title_str or "Untitled"), ""


def render_circular_carousel(
    movies_df: pd.DataFrame,
    mode: str,
    context: dict[str, Any],
    max_cards: int = 7,
    height: int = 540,
    carousel_id: str = "mm_circular_carousel",
) -> None:
    """Render the 3D circular movie carousel in Streamlit."""
    if movies_df is None or movies_df.empty:
        return

    cards_df = movies_df.head(max_cards)
    preload_movie_media(cards_df["tmdbId"].tolist() if "tmdbId" in cards_df.columns else [])
    cards_data = []
    saved_keys = st.session_state.get("_watch_later_keys", set())

    for _, row in cards_df.iterrows():
        movie_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        movie_id = int(movie_dict.get("movieId", 0))
        tmdb_id = movie_dict.get("tmdbId")
        raw_title = movie_dict.get("title", "")
        title, year = _clean_title_year(raw_title)

        # Get poster
        metadata = get_movie_details(tmdb_id) if tmdb_id else None
        poster_path = (metadata or {}).get("poster_path") or movie_dict.get("poster_path")
        poster_uri = get_movie_poster_url(poster_path) or _card_image_uri(None)
        trailer = get_movie_trailer(tmdb_id)
        trailer_key = trailer.get("key", "") if trailer else ""

        # Compute score
        is_intent = (
            mode == "current"
            and movie_dict.get("intent_match") is not None
            and pd.notna(movie_dict.get("intent_match"))
        )
        if is_intent:
            score_val = int(movie_dict["intent_match"])
            score_label = "Intent Match"
        elif "similarity" in movie_dict:
            score_val = round(float(movie_dict.get("similarity", 0) or 0) * 100)
            score_label = "Similarity"
        else:
            score_val = round(float(movie_dict.get("hybrid_score", 0) or 0) * 100)
            score_label = "MovieMind Match"

        # Genres
        raw_genres = str(movie_dict.get("genres", "") or "").replace("|", " · ")

        # Watch later
        entry = watch_later_movie_entry(movie_dict)
        w_key = watch_later_key(entry)
        is_saved = w_key in saved_keys if w_key else False

        is_external = bool(movie_dict.get("_external_tmdb")) or movie_id < 0

        try:
            safe_tmdb_id = int(tmdb_id)
        except (TypeError, ValueError, OverflowError):
            safe_tmdb_id = 0

        cards_data.append({
            "movieId": movie_id,
            "tmdbId": safe_tmdb_id,
            "title": title,
            "year": year,
            "genres": raw_genres,
            "posterUri": poster_uri,
            "score": score_val,
            "scoreLabel": score_label,
            "isSaved": is_saved,
            "isExternal": is_external,
            "trailerKey": trailer_key,
        })

    import json
    cards_json = json.dumps(cards_data)
    theme_css = ""
    if st.session_state.get("ui_theme", "dark") == "light":
        theme_css = """
      body { color:#2d241e; }
      .movie-arc-card { background:linear-gradient(165deg,#ffffff,#f0ebe6);border-color:rgba(75,55,42,.2);box-shadow:0 14px 35px rgba(49,35,26,.2); }
      .movie-arc-card.is-active { border-color:#bd5d43;box-shadow:0 20px 50px rgba(49,35,26,.25),0 0 25px rgba(189,93,67,.18); }
      .card-poster-wrap { background:#e9e3dd; }
      .card-title { color:#302720; }
      .card-meta,.carousel-hint { color:#74685f; }
      .action-btn { background:rgba(255,255,255,.86);border-color:rgba(75,55,42,.2);color:#392f29; }
      .action-btn:hover { background:#f5ede7;color:#302720; }
      .carousel-nav-btn { background:rgba(255,255,255,.9);border-color:#ded3ca;color:#392f29; }
      .dot { background:rgba(80,60,45,.25); }
        """

    carousel_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <style>
      * {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
        user-select: none;
      }}
      body {{
        background: transparent;
        color: #F5EBE6;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        overflow: hidden;
        width: 100%;
        height: {height}px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
      }}
      .carousel-stage {{
        position: relative;
        width: 100%;
        height: {height - 40}px;
        perspective: 1200px;
        perspective-origin: 50% 50%;
        overflow: visible;
        display: flex;
        align-items: center;
        justify-content: center;
      }}
      .carousel-track {{
        position: absolute;
        width: 100%;
        height: 100%;
        transform-style: preserve-3d;
        display: flex;
        align-items: center;
        justify-content: center;
        pointer-events: none;
      }}
      .movie-arc-card {{
        position: absolute;
        width: 205px;
        height: 330px;
        border-radius: 16px;
        background: linear-gradient(165deg, rgba(60, 42, 33, 0.97), rgba(26, 18, 11, 0.98));
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 14px 35px rgba(0, 0, 0, 0.6);
        overflow: hidden;
        cursor: pointer;
        pointer-events: auto;
        will-change: transform, opacity, filter, box-shadow;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
        display: flex;
        flex-direction: column;
      }}
      .movie-arc-card.is-active {{
        border: 2px solid #E07A5F;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8),
                    0 0 35px rgba(224, 122, 95, 0.45),
                    0 0 15px rgba(242, 204, 143, 0.28);
      }}
      .card-poster-wrap {{
        position: relative;
        width: 100%;
        height: 230px;
        overflow: hidden;
        background: #1A120B;
      }}
      .card-trailer-preview {{
        display: none;
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        border: 0;
        background: #1A120B;
        pointer-events: none;
        z-index: 3;
      }}
      .movie-arc-card.is-previewing .card-trailer-preview {{ display: block; }}
      .card-poster {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
        transition: transform 0.4s ease;
      }}
      .movie-arc-card.is-active .card-poster {{
        transform: scale(1.04);
      }}
      .card-poster-gradient {{
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 60px;
        background: linear-gradient(0deg, rgba(26, 18, 11, 1) 0%, rgba(26, 18, 11, 0) 100%);
      }}
      .card-badge {{
        position: absolute;
        top: 10px;
        left: 10px;
        padding: 4px 8px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 800;
        background: rgba(60, 42, 33, 0.92);
        color: #F2CC8F;
        border: 1px solid rgba(242, 204, 143, 0.5);
        backdrop-filter: blur(8px);
        letter-spacing: 0.2px;
      }}
      .card-saved-badge {{
        position: absolute;
        top: 10px;
        right: 10px;
        padding: 4px 8px;
        border-radius: 999px;
        font-size: 10px;
        font-weight: 700;
        background: rgba(242, 204, 143, 0.18);
        color: #F2CC8F;
        border: 1px solid rgba(242, 204, 143, 0.4);
        backdrop-filter: blur(8px);
      }}
      .card-body {{
        padding: 8px 12px 10px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        flex: 1;
      }}
      .card-title {{
        font-size: 13px;
        font-weight: 750;
        color: #F5EBE6;
        line-height: 1.25;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
      .card-meta {{
        font-size: 11px;
        color: #cbb8ad;
        margin-top: 3px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
      .card-actions {{
        display: none;
        gap: 6px;
        margin-top: 8px;
      }}
      .movie-arc-card.is-active .card-actions {{
        display: flex;
      }}
      .action-btn {{
        flex: 1;
        padding: 5px 6px;
        border-radius: 8px;
        font-size: 10px;
        font-weight: 700;
        text-align: center;
        text-decoration: none;
        cursor: pointer;
        border: 1px solid rgba(255, 255, 255, 0.15);
        background: rgba(255, 255, 255, 0.08);
        color: #f1f3f7;
        transition: all 0.16s ease;
        white-space: nowrap;
      }}
      .action-btn:hover {{
        background: rgba(255, 255, 255, 0.18);
        border-color: #E07A5F;
        color: #fff;
        transform: translateY(-1px);
      }}
      .action-btn.primary {{
        background: linear-gradient(110deg, #E07A5F, #F2CC8F);
        border: none;
        color: #1A120B;
        font-weight: 800;
      }}
      .movie-arc-card:hover .card-actions {{ display: flex; }}
      .action-btn.primary:hover {{
        filter: brightness(1.12);
      }}
      .carousel-nav-btn {{
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        width: 42px;
        height: 42px;
        border-radius: 50%;
        background: rgba(17, 23, 33, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: #f4f6fa;
        font-size: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        z-index: 120;
        transition: all 0.18s ease;
        backdrop-filter: blur(10px);
      }}
      .carousel-nav-btn:hover {{
        background: rgba(224, 166, 109, 0.25);
        border-color: #E07A5F;
        color: #fff;
        transform: translateY(-50%) scale(1.08);
      }}
      .carousel-nav-btn.prev {{ left: 14px; }}
      .carousel-nav-btn.next {{ right: 14px; }}
      .carousel-indicator {{
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 6px;
        z-index: 10;
      }}
      .dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.25);
        transition: all 0.22s ease;
        cursor: pointer;
      }}
      .dot.is-active {{
        width: 22px;
        border-radius: 999px;
        background: linear-gradient(90deg, #E07A5F, #F2CC8F);
      }}
      .carousel-hint {{
        font-size: 11px;
        color: #7b889b;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 2px;
      }}
      @media (max-width: 768px) {{
        .movie-arc-card {{
          width: 175px;
          height: 290px;
        }}
        .card-poster-wrap {{
          height: 195px;
        }}
        .carousel-nav-btn {{
          width: 34px;
          height: 34px;
          font-size: 16px;
        }}
      }}
      {theme_css}
    </style>
    </head>
    <body>
      <div class="carousel-hint">Move cursor horizontally to rotate · Click card to center</div>
      <div class="carousel-stage" id="carouselStage">
        <button class="carousel-nav-btn prev" id="btnPrev" aria-label="Previous movie">‹</button>
        <button class="carousel-nav-btn next" id="btnNext" aria-label="Next movie">›</button>
        <div class="carousel-track" id="carouselTrack"></div>
      </div>
      <div class="carousel-indicator" id="carouselIndicator"></div>

      <script>
      (function() {{
        const movies = {cards_json};
        const count = movies.length;
        if (count === 0) return;

        const stage = document.getElementById('carouselStage');
        const track = document.getElementById('carouselTrack');
        const indicator = document.getElementById('carouselIndicator');
        const btnPrev = document.getElementById('btnPrev');
        const btnNext = document.getElementById('btnNext');

        let cardElements = [];
        let dotElements = [];
        let currentIndex = 0;
        let currentRotation = 0;
        let targetRotation = 0;
        let isHovered = false;
        let isDragging = false;
        let dragStartX = 0;
        let dragStartRotation = 0;

        // Create cards DOM
        movies.forEach((m, idx) => {{
          const card = document.createElement('div');
          card.className = 'movie-arc-card';
          card.dataset.index = idx;

          const savedBadge = m.isSaved ? '<span class="card-saved-badge">✓ Saved</span>' : '';
          const moreLikeBtn = (!m.isExternal && m.movieId > 0)
            ? `<a class="action-btn" href="?action=more_like&movie_id=${{m.movieId}}" target="_top">✦ More Like</a>`
            : '';
          const watchBtnText = m.isSaved ? '✓ Remove' : '＋ Watch Later';
          const trailerHref = `?action=trailer&movie_id=${{encodeURIComponent(m.movieId)}}&tmdb_id=${{encodeURIComponent(m.tmdbId)}}&title=${{encodeURIComponent(m.title)}}`;

          card.innerHTML = `
            <div class="card-poster-wrap">
              <img class="card-poster" src="${{m.posterUri}}" alt="${{m.title}}">
              <iframe class="card-trailer-preview" title="Muted trailer preview" allow="autoplay; encrypted-media; picture-in-picture"></iframe>
              <div class="card-poster-gradient"></div>
              <span class="card-badge">${{m.score}}% ${{m.scoreLabel === 'Intent Match' ? 'Intent' : 'Match'}}</span>
              ${{savedBadge}}
            </div>
            <div class="card-body">
              <div>
                <div class="card-title" title="${{m.title}}">${{m.title}}</div>
                <div class="card-meta">${{m.year ? m.year + ' · ' : ''}}${{m.genres}}</div>
              </div>
              <div class="card-actions">
                <a class="action-btn primary" href="?action=details&movie_id=${{m.movieId}}" target="_top">▶ Details</a>
                <a class="action-btn" href="${{trailerHref}}" target="_top">▶ Trailer</a>
                <a class="action-btn" href="?action=watch_later&movie_id=${{m.movieId}}" target="_top">${{watchBtnText}}</a>
                ${{moreLikeBtn}}
              </div>
            </div>
          `;

          card.addEventListener('click', (e) => {{
            // If clicking an action link, allow standard link navigation
            if (e.target.tagName === 'A' || e.target.closest('a')) return;
            // Otherwise center this card
            rotateToIndex(idx);
          }});

          const previewFrame = card.querySelector('.card-trailer-preview');
          card.addEventListener('mouseenter', () => {{
            if (!m.trailerKey) return;
            card.classList.add('is-previewing');
            previewFrame.src = `https://www.youtube-nocookie.com/embed/${{m.trailerKey}}?autoplay=1&mute=1&controls=0&playsinline=1&rel=0&loop=1&playlist=${{m.trailerKey}}`;
          }});
          card.addEventListener('mouseleave', () => {{
            card.classList.remove('is-previewing');
            previewFrame.src = 'about:blank';
          }});

          track.appendChild(card);
          cardElements.push(card);

          // Dot indicator
          const dot = document.createElement('div');
          dot.className = 'dot' + (idx === 0 ? ' is-active' : '');
          dot.addEventListener('click', () => rotateToIndex(idx));
          indicator.appendChild(dot);
          dotElements.push(dot);
        }});

        function rotateToIndex(index) {{
          currentIndex = (index + count) % count;
          targetRotation = -currentIndex * (2 * Math.PI / count);
        }}

        btnPrev.addEventListener('click', () => rotateToIndex(currentIndex - 1));
        btnNext.addEventListener('click', () => rotateToIndex(currentIndex + 1));

        // Keyboard navigation
        window.addEventListener('keydown', (e) => {{
          if (e.key === 'ArrowLeft') rotateToIndex(currentIndex - 1);
          if (e.key === 'ArrowRight') rotateToIndex(currentIndex + 1);
        }});

        // Mousemove rotation across stage
        stage.addEventListener('mouseenter', () => {{ isHovered = true; }});
        stage.addEventListener('mouseleave', () => {{
          isHovered = false;
          // Snap back to closest integer index when mouse leaves
          const step = (2 * Math.PI) / count;
          const nearest = Math.round(-targetRotation / step);
          rotateToIndex(nearest);
        }});

        stage.addEventListener('mousemove', (e) => {{
          if (isDragging) return;
          const rect = stage.getBoundingClientRect();
          const normX = (e.clientX - rect.left) / rect.width - 0.5; // -0.5 to +0.5
          // Subtle rotation driven by mouse cursor position
          const maxInfluence = Math.PI * 0.85;
          targetRotation = -currentIndex * (2 * Math.PI / count) - normX * maxInfluence;
        }});

        // Touch support
        stage.addEventListener('touchstart', (e) => {{
          isDragging = true;
          dragStartX = e.touches[0].clientX;
          dragStartRotation = targetRotation;
        }}, {{ passive: true }});

        stage.addEventListener('touchmove', (e) => {{
          if (!isDragging) return;
          const diff = e.touches[0].clientX - dragStartX;
          targetRotation = dragStartRotation + (diff / stage.clientWidth) * Math.PI;
        }}, {{ passive: true }});

        stage.addEventListener('touchend', () => {{
          isDragging = false;
          const step = (2 * Math.PI) / count;
          const nearest = Math.round(-targetRotation / step);
          rotateToIndex(nearest);
        }});

        // 60FPS render loop
        function renderLoop() {{
          // Smooth spring damping
          currentRotation += (targetRotation - currentRotation) * 0.085;

          const stageWidth = stage.clientWidth || 900;
          const radiusX = Math.min(stageWidth * 0.40, 360);
          const radiusZ = Math.min(stageWidth * 0.22, 210);
          const step = (2 * Math.PI) / count;

          let bestDist = Infinity;
          let bestIdx = 0;

          cardElements.forEach((card, i) => {{
            let angle = (i * step) + currentRotation;
            // Normalize angle to [-PI, PI]
            let normAngle = ((angle + Math.PI) % (2 * Math.PI)) - Math.PI;
            if (normAngle < -Math.PI) normAngle += 2 * Math.PI;

            const dist = Math.abs(normAngle);
            if (dist < bestDist) {{
              bestDist = dist;
              bestIdx = i;
            }}

            const x = Math.sin(normAngle) * radiusX;
            const z = Math.cos(normAngle) * radiusZ - radiusZ;
            const y = -Math.pow(normAngle, 2) * 14;
            const rotY = normAngle * (180 / Math.PI) * 0.62;

            // Scale: front is 1.15x, back is 0.74x
            const cosVal = (Math.cos(normAngle) + 1) / 2; // 0 to 1
            const scale = 0.74 + 0.41 * Math.pow(cosVal, 1.8);
            const opacity = 0.48 + 0.52 * Math.pow(cosVal, 1.4);
            const zIndex = Math.round(cosVal * 100);

            card.style.transform = `translate3d(${{x}}px, ${{y}}px, ${{z}}px) rotateY(${{rotY}}deg) scale(${{scale}})`;
            card.style.opacity = opacity.toFixed(2);
            card.style.zIndex = zIndex;
          }});

          // Update active card class and dots
          cardElements.forEach((c, idx) => {{
            const isActive = idx === bestIdx;
            c.classList.toggle('is-active', isActive);
            dotElements[idx].classList.toggle('is-active', isActive);
          }});

          currentIndex = bestIdx;
          requestAnimationFrame(renderLoop);
        }}

        requestAnimationFrame(renderLoop);
      }})();
      </script>
    </body>
    </html>
    """

    components.html(carousel_html, height=height, scrolling=False)
