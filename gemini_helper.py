import streamlit as st
from google import genai
from pydantic import BaseModel, Field
from typing import List, Optional


class MovieIntent(BaseModel):
    genres: List[str] = Field(
        default_factory=list,
        description="Movie genres requested by the user."
    )
    mood: Optional[str] = Field(
        default=None,
        description="The user's desired viewing mood."
    )
    reference_movie: Optional[str] = Field(
        default=None,
        description="A movie the user mentioned as a reference or comparison."
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Other useful movie preferences or keywords."
    )


def get_gemini_client():
    """Create the Gemini client using the Streamlit secret."""
    api_key = st.secrets["GEMINI_API_KEY"]
    return genai.Client(
        api_key=api_key,
        http_options=genai.types.HttpOptions(
            timeout=30_000,
            retry_options=genai.types.HttpRetryOptions(attempts=1),
        ),
    )


def parse_movie_intent(user_text: str) -> MovieIntent:
    """Convert natural-language movie preferences into structured intent."""

    client = get_gemini_client()

    prompt = f"""
You are the intent parser for MovieMind, a personalized movie
recommendation system.

Analyze the user's request and extract only movie-related preferences.

Allowed common genres include:
Action, Adventure, Animation, Comedy, Crime, Documentary,
Drama, Fantasy, Horror, Mystery, Romance, Sci-Fi, Thriller,
Family, War, Western, Musical, and Film-Noir.

For mood, use a short description such as:
Funny, Romantic, Emotional, Relaxing, Exciting, Dark, Suspenseful,
Feel-Good, or null if no mood is expressed.

If the user mentions a movie as a reference, comparison, or
"something like" movie, put its title in reference_movie.

Do not invent preferences that the user did not express.

User request:
{user_text}
"""

    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt,
        timeout=30,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": MovieIntent.model_json_schema(),
        },
    )

    return MovieIntent.model_validate_json(interaction.output_text)
