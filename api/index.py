import json
import os
import re
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TMDB_KEY = os.getenv("TMDB_KEY")
OMDB_KEY = os.getenv("OMDB_KEY")

# Gemini model.
# Keep this configurable so you can change models from Vercel
# without changing the source code.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")

TMDB_BASE = "https://api.tmdb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


# ============================================================
# HTTP SESSION
# ============================================================

_session = httpx.Client(
    http2=True,
    timeout=20.0,
)

_session.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="MovieMoody API",
    version="2.0.0",
    description="Mood-based movie and TV recommendation API powered by Gemini.",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODELS
# ============================================================

class MoodRequest(BaseModel):
    mood: str
    content_type: str = "movie"


class Movie(BaseModel):
    title: str
    year: Optional[str] = None
    poster: Optional[str] = None
    genres: list[str] = []
    overview: Optional[str] = None
    imdb_rating: Optional[str] = None
    why: Optional[str] = None


# ============================================================
# VALIDATION / UTILITIES
# ============================================================

def normalize_content_type(content_type: str) -> str:
    """
    Normalize frontend values.

    Accepted:
        movie
        movies
        film
        show
        shows
        tv
        tv_show
    """

    value = (content_type or "movie").strip().lower()

    if value in {"show", "shows", "tv", "tv_show", "series"}:
        return "show"

    return "movie"


def clean_json_text(text: str) -> str:
    """
    Remove markdown code fences and extract the JSON array
    if Gemini surrounds it with additional text.
    """

    if not text:
        return ""

    text = text.strip()

    # Remove markdown fences.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    text = text.strip()

    # Find the first JSON array.
    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text


# ============================================================
# GEMINI
# ============================================================

def get_movies_from_gemini(
    mood: str,
    content_type: str = "movie",
) -> list[dict]:
    """
    Ask Gemini for exactly six recommendations.

    Gemini replaces the old Claude recommendation layer.

    The returned objects contain only:
        title
        year
        why

    TMDB and OMDb are responsible for the actual metadata.
    """

    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail=(
                "GEMINI_API_KEY is not configured. "
                "Add GEMINI_API_KEY to your Vercel Environment Variables."
            ),
        )

    content_type = normalize_content_type(content_type)

    if content_type == "show":
        kind = "TV shows or series"
        kind_single = "TV series"

        title_instruction = (
            'Use the exact TV title as it appears in databases. '
            'Examples: "Severance", "The Bear", "Dark".'
        )

    else:
        kind = "movies"
        kind_single = "movie"

        title_instruction = (
            'Use the exact movie title as it appears in databases. '
            'Examples: "The Godfather", "Inception", "Dune: Part Two".'
        )

    prompt = f"""
You are MovieMoody, an expert {kind_single} recommendation engine.

The user's current mood is:

"{mood}"

Recommend exactly 6 {kind} that genuinely match this mood.

IMPORTANT:
- Return exactly 6 items.
- Do not invent titles.
- Use real movies/shows that exist.
- Use exact database-friendly titles.
- Include a mixture of well-known and interesting choices.
- Avoid recommending the same title twice.
- Do not mention streaming platforms.
- Do not recommend something merely because it is popular.
- Match the emotional tone of the user's mood.

{title_instruction}

Each item MUST contain exactly these fields:

{{
  "title": "Exact title",
  "year": "YYYY",
  "why": "Short explanation of why it matches the mood"
}}

The "why" field must:
- Be one sentence.
- Be no more than 12 words.
- Be specific to the user's mood.

Return ONLY a valid JSON array.

Do not use Markdown.
Do not use ```json.
Do not add explanations before or after the JSON.
"""

    url = (
        f"{GEMINI_BASE}/models/"
        f"{GEMINI_MODEL}:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": prompt
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.8,
            "topP": 0.9,
            "maxOutputTokens": 1000,
            "responseMimeType": "application/json",
        },
    }

    try:
        response = _session.post(
            url,
            json=payload,
            timeout=30.0,
        )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Gemini request timed out. Please try again.",
        )

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot reach Gemini API: {exc}",
        )

    if not response.is_success:
        try:
            error_data = response.json()

            error_message = (
                error_data.get("error", {}).get("message")
                or response.text
            )

        except Exception:
            error_message = response.text

        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error: {error_message}",
        )

    try:
        data = response.json()

        candidates = data.get("candidates", [])

        if not candidates:
            raise ValueError("Gemini returned no candidates.")

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )

        text = "".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, dict)
        ).strip()

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Invalid Gemini response: {exc}",
        )

    if not text:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned an empty recommendation response.",
        )

    cleaned = clean_json_text(text)

    try:
        recommendations = json.loads(cleaned)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned invalid recommendation JSON.",
        )

    if not isinstance(recommendations, list):
        raise HTTPException(
            status_code=502,
            detail="Gemini returned an invalid recommendation format.",
        )

    # Sanitize the response.
    cleaned_recommendations = []

    seen_titles = set()

    for item in recommendations:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title", "")).strip()

        if not title:
            continue

        normalized_title = title.lower()

        if normalized_title in seen_titles:
            continue

        seen_titles.add(normalized_title)

        year = str(item.get("year", "")).strip()

        why = str(item.get("why", "")).strip()

        cleaned_recommendations.append(
            {
                "title": title,
                "year": year or None,
                "why": why or None,
            }
        )

        if len(cleaned_recommendations) == 6:
            break

    if not cleaned_recommendations:
        raise HTTPException(
            status_code=502,
            detail="Gemini did not return usable movie recommendations.",
        )

    return cleaned_recommendations


# ============================================================
# TMDB
# ============================================================

def fetch_tmdb(
    title: str,
    year: Optional[str],
    content_type: str = "movie",
) -> Optional[dict]:
    """
    Search TMDB and return normalized movie/TV information.
    """

    content_type = normalize_content_type(content_type)

    if content_type == "show":
        search_endpoint = f"{TMDB_BASE}/search/tv"

        params = {
            "api_key": TMDB_KEY,
            "query": title,
            "language": "en-US",
            "page": 1,
        }

        if year and year.isdigit():
            params["first_air_date_year"] = int(year)

    else:
        search_endpoint = f"{TMDB_BASE}/search/movie"

        params = {
            "api_key": TMDB_KEY,
            "query": title,
            "language": "en-US",
            "page": 1,
        }

        if year and year.isdigit():
            params["year"] = int(year)

    try:
        response = _session.get(
            search_endpoint,
            params=params,
            timeout=15.0,
        )

    except httpx.TimeoutException:
        return None

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot reach TMDB: {exc}",
        )

    if not response.is_success:
        return None

    try:
        data = response.json()
    except Exception:
        return None

    results = data.get("results", [])

    if not results:
        return None

    # Prefer exact title matches.
    normalized_query = title.strip().lower()

    exact = None

    for result in results:
        result_title = (
            result.get("title")
            if content_type == "movie"
            else result.get("name")
        )

        if result_title and result_title.strip().lower() == normalized_query:
            exact = result
            break

    item = exact or results[0]

    poster_path = item.get("poster_path")

    if content_type == "show":
        exact_title = item.get("name")
        release_date = item.get("first_air_date") or ""

    else:
        exact_title = item.get("title")
        release_date = item.get("release_date") or ""

    return {
        "title": exact_title,
        "year": release_date[:4] if release_date else None,
        "poster": (
            f"{TMDB_IMG}{poster_path}"
            if poster_path
            else None
        ),
        "overview": item.get("overview"),
        "genre_ids": item.get("genre_ids", []),
    }


def fetch_tmdb_genres(content_type: str = "movie") -> dict:
    """
    Return:
        {genre_id: genre_name}
    """

    content_type = normalize_content_type(content_type)

    if content_type == "show":
        endpoint = f"{TMDB_BASE}/genre/tv/list"
    else:
        endpoint = f"{TMDB_BASE}/genre/movie/list"

    try:
        response = _session.get(
            endpoint,
            params={
                "api_key": TMDB_KEY,
                "language": "en-US",
            },
            timeout=15.0,
        )

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot reach TMDB: {exc}",
        )

    if not response.is_success:
        return {}

    try:
        genres = response.json().get("genres", [])
    except Exception:
        return {}

    return {
        genre["id"]: genre["name"]
        for genre in genres
        if "id" in genre and "name" in genre
    }


# ============================================================
# OMDB
# ============================================================

def fetch_omdb_rating(
    title: str,
    content_type: str = "movie",
) -> Optional[str]:
    """
    Fetch IMDb rating from OMDb.
    """

    if not OMDB_KEY:
        return None

    params = {
        "t": title,
        "apikey": OMDB_KEY,
    }

    if normalize_content_type(content_type) == "show":
        params["type"] = "series"

    try:
        response = _session.get(
            "https://www.omdbapi.com/",
            params=params,
            timeout=15.0,
        )

    except (httpx.TimeoutException, httpx.RequestError):
        return None

    if not response.is_success:
        return None

    try:
        data = response.json()
    except Exception:
        return None

    if data.get("Response") == "False":
        return None

    rating = data.get("imdbRating")

    if not rating or rating == "N/A":
        return None

    return str(rating)


# ============================================================
# BUILD FINAL RECOMMENDATION
# ============================================================

def build_movie(
    ai_item: dict,
    genre_map: dict,
    content_type: str,
) -> Optional[dict]:
    """
    Combine:
        Gemini
        +
        TMDB
        +
        OMDb
    """

    title = ai_item.get("title", "").strip()
    year = ai_item.get("year")
    why = ai_item.get("why")

    if not title:
        return None

    tmdb = fetch_tmdb(
        title=title,
        year=year,
        content_type=content_type,
    )

    if not tmdb:
        return None

    exact_title = tmdb["title"]

    imdb_rating = fetch_omdb_rating(
        exact_title,
        content_type,
    )

    genres = [
        genre_map.get(genre_id)
        for genre_id in tmdb.get("genre_ids", [])
        if genre_id in genre_map
    ]

    genres = [
        genre
        for genre in genres
        if genre
    ]

    return {
        "title": exact_title,
        "year": tmdb.get("year"),
        "poster": tmdb.get("poster"),
        "genres": genres,
        "overview": tmdb.get("overview"),
        "imdb_rating": imdb_rating,
        "why": why,
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api")
def root():
    return {
        "status": "ok",
        "message": "MovieMoody API is running",
        "ai_provider": "Gemini",
        "model": GEMINI_MODEL,
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "gemini_configured": bool(GEMINI_API_KEY),
        "tmdb_configured": bool(TMDB_KEY),
        "omdb_configured": bool(OMDB_KEY),
        "ai_provider": "Gemini",
        "model": GEMINI_MODEL,
    }


# ============================================================
# RECOMMENDATIONS
# ============================================================

@app.post(
    "/api/recommend",
    response_model=list[Movie],
)
def recommend(body: MoodRequest):
    """
    Generate six mood-based movie/show recommendations.

    Request:

    {
        "mood": "something cozy for a rainy Sunday",
        "content_type": "movie"
    }

    Sources:

    Gemini
        -> titles + reasons

    TMDB
        -> posters + genres + overview

    OMDb
        -> IMDb rating
    """

    mood = body.mood.strip()

    if not mood:
        raise HTTPException(
            status_code=400,
            detail="mood cannot be empty",
        )

    if len(mood) > 1000:
        raise HTTPException(
            status_code=400,
            detail="mood is too long. Please keep it under 1000 characters.",
        )

    content_type = normalize_content_type(
        body.content_type
    )

    # Required services.
    missing = []

    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")

    if not TMDB_KEY:
        missing.append("TMDB_KEY")

    if not OMDB_KEY:
        missing.append("OMDB_KEY")

    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                "Missing API environment variables: "
                + ", ".join(missing)
            ),
        )

    # --------------------------------------------------------
    # STEP 1: Gemini
    # --------------------------------------------------------

    ai_recommendations = get_movies_from_gemini(
        mood=mood,
        content_type=content_type,
    )

    # --------------------------------------------------------
    # STEP 2: TMDB genres
    # --------------------------------------------------------

    genre_map = fetch_tmdb_genres(
        content_type=content_type,
    )

    # --------------------------------------------------------
    # STEP 3: Enrich every Gemini recommendation
    # --------------------------------------------------------

    movies = []

    for item in ai_recommendations:

        movie = build_movie(
            ai_item=item,
            genre_map=genre_map,
            content_type=content_type,
        )

        if movie:
            movies.append(movie)

        # Stop at six valid results.
        if len(movies) == 6:
            break

    if not movies:
        raise HTTPException(
            status_code=404,
            detail=(
                "Gemini returned recommendations, "
                "but none could be matched on TMDB."
            ),
        )

    return movies


# ============================================================
# SINGLE MOVIE / SHOW
# ============================================================

@app.get(
    "/movie/{title}",
    response_model=Movie,
)
def get_movie(title: str):
    """
    Fetch a single movie by title.

    Example:
        /movie/Inception

    This endpoint remains compatible with the existing frontend.
    """

    title = title.strip()

    if not title:
        raise HTTPException(
            status_code=400,
            detail="Movie title cannot be empty.",
        )

    if not TMDB_KEY:
        raise HTTPException(
            status_code=500,
            detail="TMDB_KEY is not configured.",
        )

    genre_map = fetch_tmdb_genres("movie")

    tmdb = fetch_tmdb(
        title=title,
        year=None,
        content_type="movie",
    )

    if not tmdb:
        raise HTTPException(
            status_code=404,
            detail=f"'{title}' not found on TMDB.",
        )

    exact_title = tmdb["title"]

    imdb_rating = fetch_omdb_rating(
        exact_title,
        "movie",
    )

    genres = [
        genre_map.get(genre_id)
        for genre_id in tmdb.get("genre_ids", [])
        if genre_id in genre_map
    ]

    genres = [
        genre
        for genre in genres
        if genre
    ]

    return {
        "title": exact_title,
        "year": tmdb.get("year"),
        "poster": tmdb.get("poster"),
        "genres": genres,
        "overview": tmdb.get("overview"),
        "imdb_rating": imdb_rating,
        "why": None,
    }
