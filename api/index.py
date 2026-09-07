import json
import os
import re
import time
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TMDB_KEY = os.getenv("TMDB_KEY")
OMDB_KEY = os.getenv("OMDB_KEY")

# Optional:
# Set GEMINI_MODEL in Vercel if you want to force a starting model.
#
# If not set, the API automatically tries the models below.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip()

# Fallback order.
# If one model is busy/unavailable, the next one is tried.
GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
]

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
TMDB_BASE = "https://api.tmdb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"


# ============================================================
# HTTP CLIENT
# ============================================================

_session = httpx.Client(
    http2=True,
    timeout=httpx.Timeout(
        connect=10.0,
        read=30.0,
        write=10.0,
        pool=10.0,
    ),
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
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="MovieMoody API",
    version="3.0.0",
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
# HELPERS
# ============================================================

def normalize_content_type(content_type: str) -> str:
    """
    Normalize movie/show input.
    """

    value = (content_type or "movie").strip().lower()

    if value in {
        "show",
        "shows",
        "tv",
        "tv_show",
        "tvshow",
        "series",
    }:
        return "show"

    return "movie"


def clean_json_text(text: str) -> str:
    """
    Remove markdown code fences and extract JSON array.
    """

    if not text:
        return ""

    text = text.strip()

    # Remove ```json
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove ```
    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    text = text.strip()

    # Find JSON array
    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text


def clean_title(title: str) -> str:
    """
    Normalize title for searching.
    """

    if not title:
        return ""

    title = str(title).strip()

    title = re.sub(
        r"\s+",
        " ",
        title,
    )

    return title


def unique_models() -> list[str]:
    """
    Put user-selected model first, followed by fallback models.
    """

    models = []

    if GEMINI_MODEL:
        models.append(GEMINI_MODEL)

    for model in GEMINI_MODELS:
        if model not in models:
            models.append(model)

    return models


# ============================================================
# GEMINI
# ============================================================

def get_movies_from_gemini(
    mood: str,
    content_type: str,
) -> list[dict]:
    """
    Ask Gemini for six recommendations.

    Automatically retries temporary Gemini errors and
    falls back to other Gemini models.
    """

    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured.",
        )

    mood = (mood or "").strip()

    if not mood:
        raise HTTPException(
            status_code=400,
            detail="Mood cannot be empty.",
        )

    content_type = normalize_content_type(content_type)

    if content_type == "show":
        media_name = "TV shows"
        media_instruction = (
            "Return television series only. "
            "Do not return movies."
        )
    else:
        media_name = "movies"
        media_instruction = (
            "Return movies only. "
            "Do not return TV shows or series."
        )

    prompt = f"""
You are MovieMoody, an expert entertainment recommendation assistant.

The user currently feels:
"{mood}"

Recommend exactly 6 {media_name} that strongly match this mood.

{media_instruction}

Choose real, released, well-known titles.

Avoid:
- duplicate titles
- fake titles
- unreleased titles
- vague recommendations
- titles that do not match the mood

Return ONLY valid JSON.

The JSON must be an array with exactly this structure:

[
  {{
    "title": "Movie or Show Title",
    "year": "2024",
    "why": "Short explanation of why it matches the user's mood."
  }}
]

Rules:
- Exactly 6 recommendations
- "year" must be the release year
- "why" must be concise
- Do not include markdown
- Do not include commentary outside the JSON
"""

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
            "maxOutputTokens": 1200,
            "responseMimeType": "application/json",
        },
    }

    models = unique_models()

    temporary_statuses = {
        429,
        500,
        502,
        503,
        504,
    }

    last_error = None

    # ========================================================
    # TRY EVERY MODEL
    # ========================================================

    for model in models:

        url = (
            f"{GEMINI_BASE}/models/"
            f"{model}:generateContent"
        )

        # ----------------------------------------------------
        # RETRY CURRENT MODEL
        # ----------------------------------------------------

        for attempt in range(3):

            try:

                response = _session.post(
                    url,
                    params={
                        "key": GEMINI_API_KEY,
                    },
                    json=payload,
                )

                # ------------------------------------------------
                # SUCCESS
                # ------------------------------------------------

                if response.status_code == 200:

                    data = response.json()

                    candidates = data.get(
                        "candidates",
                        [],
                    )

                    if not candidates:
                        last_error = (
                            f"{model}: Gemini returned no candidates."
                        )
                        break

                    parts = (
                        candidates[0]
                        .get("content", {})
                        .get("parts", [])
                    )

                    text_parts = []

                    for part in parts:
                        text = part.get("text")

                        if text:
                            text_parts.append(text)

                    text = "".join(text_parts).strip()

                    if not text:
                        last_error = (
                            f"{model}: Gemini returned empty text."
                        )
                        break

                    cleaned = clean_json_text(text)

                    try:
                        recommendations = json.loads(cleaned)

                    except json.JSONDecodeError as exc:
                        last_error = (
                            f"{model}: Invalid JSON returned by Gemini: "
                            f"{exc}"
                        )

                        # Try the model again.
                        if attempt < 2:
                            time.sleep(1.5 * (attempt + 1))
                            continue

                        break

                    if not isinstance(
                        recommendations,
                        list,
                    ):
                        last_error = (
                            f"{model}: Gemini response was not a list."
                        )
                        break

                    # ------------------------------------------------
                    # CLEAN RECOMMENDATIONS
                    # ------------------------------------------------

                    cleaned_recommendations = []
                    seen_titles = set()

                    for item in recommendations:

                        if not isinstance(item, dict):
                            continue

                        title = clean_title(
                            item.get("title", "")
                        )

                        if not title:
                            continue

                        normalized_title = title.lower()

                        if normalized_title in seen_titles:
                            continue

                        seen_titles.add(normalized_title)

                        year = item.get("year")

                        if year is not None:
                            year = str(year).strip()

                        why = item.get("why")

                        if why is not None:
                            why = str(why).strip()

                        cleaned_recommendations.append(
                            {
                                "title": title,
                                "year": year,
                                "why": why,
                            }
                        )

                        if len(cleaned_recommendations) >= 6:
                            break

                    if not cleaned_recommendations:
                        last_error = (
                            f"{model}: Gemini returned no usable titles."
                        )
                        break

                    print(
                        f"Gemini recommendation success using {model}"
                    )

                    return cleaned_recommendations

                # ------------------------------------------------
                # TEMPORARY GEMINI ERROR
                # ------------------------------------------------

                if response.status_code in temporary_statuses:

                    try:
                        error_data = response.json()

                    except Exception:
                        error_data = response.text

                    last_error = (
                        f"{model}: HTTP "
                        f"{response.status_code}: "
                        f"{error_data}"
                    )

                    print(
                        f"Gemini temporary error "
                        f"{model} "
                        f"attempt {attempt + 1}/3: "
                        f"{response.status_code}"
                    )

                    # Retry with exponential delay.
                    if attempt < 2:

                        delay = 1.5 * (
                            2 ** attempt
                        )

                        time.sleep(delay)

                        continue

                    # All retries failed.
                    break

                # ------------------------------------------------
                # INVALID API KEY
                # ------------------------------------------------

                if response.status_code in {
                    400,
                    401,
                    403,
                }:

                    try:
                        error_data = response.json()

                    except Exception:
                        error_data = response.text

                    raise HTTPException(
                        status_code=502,
                        detail=(
                            "Gemini API authentication or "
                            "request error: "
                            f"{error_data}"
                        ),
                    )

                # ------------------------------------------------
                # OTHER ERROR
                # ------------------------------------------------

                try:
                    error_data = response.json()

                except Exception:
                    error_data = response.text

                last_error = (
                    f"{model}: HTTP "
                    f"{response.status_code}: "
                    f"{error_data}"
                )

                break

            except httpx.TimeoutException as exc:

                last_error = (
                    f"{model}: timeout: {exc}"
                )

                print(
                    f"Gemini timeout "
                    f"{model} "
                    f"attempt {attempt + 1}/3"
                )

                if attempt < 2:

                    time.sleep(
                        1.5 * (attempt + 1)
                    )

                    continue

                break

            except httpx.RequestError as exc:

                last_error = (
                    f"{model}: network error: {exc}"
                )

                print(
                    f"Gemini network error "
                    f"{model}: {exc}"
                )

                if attempt < 2:

                    time.sleep(
                        1.5 * (attempt + 1)
                    )

                    continue

                break

            except HTTPException:
                raise

            except Exception as exc:

                last_error = (
                    f"{model}: unexpected error: {exc}"
                )

                print(
                    f"Unexpected Gemini error "
                    f"{model}: {exc}"
                )

                break

    # ========================================================
    # ALL MODELS FAILED
    # ========================================================

    print(
        "All Gemini models failed.",
        last_error,
    )

    raise HTTPException(
        status_code=503,
        detail=(
            "Gemini is temporarily unavailable. "
            "All configured Gemini models are currently "
            "busy or unavailable. Please try again shortly."
        ),
    )


# ============================================================
# TMDB REQUEST
# ============================================================

def tmdb_get(
    endpoint: str,
    params: Optional[dict] = None,
) -> dict:

    if not TMDB_KEY:
        raise HTTPException(
            status_code=500,
            detail="TMDB_KEY is not configured.",
        )

    request_params = dict(params or {})

    request_params["api_key"] = TMDB_KEY

    try:

        response = _session.get(
            f"{TMDB_BASE}{endpoint}",
            params=request_params,
        )

        response.raise_for_status()

        return response.json()

    except httpx.HTTPStatusError as exc:

        print(
            "TMDB HTTP error:",
            exc,
        )

        return {}

    except httpx.RequestError as exc:

        print(
            "TMDB network error:",
            exc,
        )

        return {}

    except Exception as exc:

        print(
            "TMDB error:",
            exc,
        )

        return {}


# ============================================================
# TMDB SEARCH
# ============================================================

def search_tmdb(
    title: str,
    content_type: str,
) -> Optional[dict]:

    content_type = normalize_content_type(
        content_type
    )

    # ========================================================
    # TV SHOW
    # ========================================================

    if content_type == "show":

        data = tmdb_get(
            "/search/tv",
            {
                "query": title,
                "include_adult": "false",
                "language": "en-US",
                "page": 1,
            },
        )

    # ========================================================
    # MOVIE
    # ========================================================

    else:

        data = tmdb_get(
            "/search/movie",
            {
                "query": title,
                "include_adult": "false",
                "language": "en-US",
                "page": 1,
            },
        )

    results = data.get(
        "results",
        [],
    )

    if not results:
        return None

    # ========================================================
    # TRY EXACT / CLOSE TITLE MATCH
    # ========================================================

    title_lower = title.strip().lower()

    for result in results:

        result_title = (
            result.get("name")
            if content_type == "show"
            else result.get("title")
        )

        if not result_title:
            continue

        if result_title.strip().lower() == title_lower:
            return result

    # Otherwise use best TMDB result.
    return results[0]


# ============================================================
# TMDB GENRES
# ============================================================

def get_tmdb_genres(
    content_type: str,
) -> dict[int, str]:

    content_type = normalize_content_type(
        content_type
    )

    if content_type == "show":

        data = tmdb_get(
            "/genre/tv/list",
            {
                "language": "en-US",
            },
        )

    else:

        data = tmdb_get(
            "/genre/movie/list",
            {
                "language": "en-US",
            },
        )

    genres = data.get(
        "genres",
        [],
    )

    return {
        int(item["id"]): item["name"]
        for item in genres
        if item.get("id") is not None
        and item.get("name")
    }


# ============================================================
# OMDB
# ============================================================

def get_imdb_rating(
    title: str,
    content_type: str,
) -> Optional[str]:

    if not OMDB_KEY:
        return None

    content_type = normalize_content_type(
        content_type
    )

    omdb_type = (
        "series"
        if content_type == "show"
        else "movie"
    )

    try:

        response = _session.get(
            "https://www.omdbapi.com/",
            params={
                "apikey": OMDB_KEY,
                "t": title,
                "type": omdb_type,
            },
        )

        response.raise_for_status()

        data = response.json()

        if data.get("Response") != "True":
            return None

        rating = data.get(
            "imdbRating"
        )

        if not rating:
            return None

        if rating == "N/A":
            return None

        return str(rating)

    except httpx.RequestError as exc:

        print(
            "OMDb network error:",
            exc,
        )

        return None

    except Exception as exc:

        print(
            "OMDb error:",
            exc,
        )

        return None


# ============================================================
# BUILD MOVIE / SHOW
# ============================================================

def build_movie(
    recommendation: dict,
    content_type: str,
    genre_map: dict[int, str],
) -> Movie:

    title = clean_title(
        recommendation.get("title", "")
    )

    year = recommendation.get(
        "year"
    )

    if year is not None:
        year = str(year).strip()

    why = recommendation.get(
        "why"
    )

    if why:
        why = str(why).strip()

    tmdb_result = search_tmdb(
        title,
        content_type,
    )

    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    poster = None
    genres = []
    overview = None
    actual_year = year

    # ========================================================
    # TMDB DATA
    # ========================================================

    if tmdb_result:

        poster_path = tmdb_result.get(
            "poster_path"
        )

        if poster_path:
            poster = (
                f"{TMDB_IMG}"
                f"{poster_path}"
            )

        overview = tmdb_result.get(
            "overview"
        )

        # ----------------------------------------------------
        # Genres
        # ----------------------------------------------------

        genre_ids = tmdb_result.get(
            "genre_ids",
            [],
        )

        genres = [
            genre_map[genre_id]
            for genre_id in genre_ids
            if genre_id in genre_map
        ]

        # ----------------------------------------------------
        # Release year
        # ----------------------------------------------------

        if content_type == "show":

            first_air_date = tmdb_result.get(
                "first_air_date"
            )

            if first_air_date:
                actual_year = (
                    first_air_date[:4]
                )

        else:

            release_date = tmdb_result.get(
                "release_date"
            )

            if release_date:
                actual_year = (
                    release_date[:4]
                )

    # ========================================================
    # IMDB RATING
    # ========================================================

    imdb_rating = get_imdb_rating(
        title,
        content_type,
    )

    return Movie(
        title=title,
        year=actual_year,
        poster=poster,
        genres=genres,
        overview=overview,
        imdb_rating=imdb_rating,
        why=why,
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "name": "MovieMoody API",
        "version": "3.0.0",
        "status": "online",
        "ai": "Google Gemini",
        "endpoints": {
            "health": "/api/health",
            "recommend": "/api/recommend",
            "movie": "/movie/{title}",
        },
    }


# ============================================================
# API STATUS
# ============================================================

@app.get("/api")
def api_status():

    return {
        "name": "MovieMoody API",
        "status": "online",
        "ai": "Gemini",
        "gemini_configured": bool(
            GEMINI_API_KEY
        ),
        "tmdb_configured": bool(
            TMDB_KEY
        ),
        "omdb_configured": bool(
            OMDB_KEY
        ),
        "models": unique_models(),
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "healthy",
        "gemini": bool(
            GEMINI_API_KEY
        ),
        "tmdb": bool(
            TMDB_KEY
        ),
        "omdb": bool(
            OMDB_KEY
        ),
        "gemini_models": unique_models(),
    }


# ============================================================
# RECOMMENDATIONS
# ============================================================

@app.post(
    "/api/recommend",
    response_model=list[Movie],
)
def recommend(
    request: MoodRequest,
):

    mood = (
        request.mood or ""
    ).strip()

    if not mood:

        raise HTTPException(
            status_code=400,
            detail="Please provide a mood.",
        )

    content_type = normalize_content_type(
        request.content_type
    )

    # ========================================================
    # GEMINI
    # ========================================================

    recommendations = get_movies_from_gemini(
        mood,
        content_type,
    )

    # ========================================================
    # GENRES
    # ========================================================

    genre_map = get_tmdb_genres(
        content_type
    )

    # ========================================================
    # BUILD RESULTS
    # ========================================================

    results = []

    for recommendation in recommendations:

        try:

            movie = build_movie(
                recommendation,
                content_type,
                genre_map,
            )

            results.append(movie)

        except Exception as exc:

            print(
                "Error building recommendation:",
                exc,
            )

            continue

    # ========================================================
    # FALLBACK
    # ========================================================

    if not results:

        raise HTTPException(
            status_code=502,
            detail=(
                "Recommendations were generated, "
                "but movie/TV metadata could not be loaded."
            ),
        )

    return results


# ============================================================
# MOVIE / SHOW DETAILS
# ============================================================

@app.get(
    "/movie/{title}",
)
def movie_details(
    title: str,
):

    title = clean_title(title)

    if not title:

        raise HTTPException(
            status_code=400,
            detail="Title cannot be empty.",
        )

    # Search movie first.
    movie = search_tmdb(
        title,
        "movie",
    )

    if movie:

        return {
            "type": "movie",
            "title": movie.get(
                "title"
            ),
            "overview": movie.get(
                "overview"
            ),
            "poster": (
                f"{TMDB_IMG}"
                f"{movie['poster_path']}"
                if movie.get("poster_path")
                else None
            ),
            "release_date": movie.get(
                "release_date"
            ),
            "rating": movie.get(
                "vote_average"
            ),
        }

    # If no movie was found, search TV.
    show = search_tmdb(
        title,
        "show",
    )

    if show:

        return {
            "type": "show",
            "title": show.get(
                "name"
            ),
            "overview": show.get(
                "overview"
            ),
            "poster": (
                f"{TMDB_IMG}"
                f"{show['poster_path']}"
                if show.get("poster_path")
                else None
            ),
            "first_air_date": show.get(
                "first_air_date"
            ),
            "rating": show.get(
                "vote_average"
            ),
        }

    raise HTTPException(
        status_code=404,
        detail="Movie or TV show not found.",
    )
