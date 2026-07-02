"""
Phi-4 Mini Travel Client — powered by phi4-mini via Ollama.

Responsible for one thing only: generating high-quality, structured
travel itineraries. Called exclusively from travel_planner.generate_itinerary().

Fallback chain:  phi4-mini → existing generate_response() (Llama)
"""

import logging
import requests

logger = logging.getLogger(__name__)

PHI4_MINI_MODEL = "phi4-mini"
OLLAMA_URL      = "http://localhost:11434/api/generate"

# Conservative options — phi4-mini is compact, keep output focused
_PHI4_OPTIONS = {
    "temperature":   0.6,   # slightly creative but grounded
    "num_predict":   2560,  # enough for a multi-day itinerary
    "num_gpu":       0,     # set to 1 if you have a working CUDA setup
    "top_p":         0.9,
    "repeat_penalty": 1.1,
}

SYSTEM_PROMPT = """You are an expert travel planner with deep knowledge of local culture,
transport, food, and logistics. You create detailed, realistic, day-by-day itineraries.

Rules:
- ONLY recommend places that are actually IN or within 30km of the destination city
- NEVER suggest places from other cities or regions — this is critical
- Organise each day with morning / afternoon / evening slots
- Include meal suggestions with local cuisine where possible
- Group nearby attractions to minimise travel time
- Add practical tips (best time to visit, ticket info, transport)
- Be specific — avoid vague phrases like "explore the city"
- Keep tone friendly and informative
- If it is a road trip, include key stops ALONG the route between origin and destination
- Also suggest some hotels to stay at night in the destination city, with a brief description of each"""

def _build_phi4_prompt(
    city: str,
    attractions: list,
    num_days: int,
    check_in: str,
    num_adults: int,
    flight_summary: str | None,
    hotel_summary: str | None,
) -> str:
    """Build the full prompt sent to phi4-mini."""
    if attractions:
        attractions_text = "\n".join(
            f"  - {a['name']} ({a['category']})"
            for a in attractions[:15]
        )
    else:
        attractions_text = (
            f"No pre-fetched list available — use your own knowledge of {city} "
            f"to recommend the best real attractions, restaurants, and experiences."
        )

    context_lines = [
        f"Destination   : {city}",
        f"Duration      : {num_days} day(s)",
        f"Check-in date : {check_in}",
        f"Travelers     : {num_adults}",
    ]
    if flight_summary:
        context_lines.append(f"Flight booked : {flight_summary}")
    if hotel_summary:
        context_lines.append(f"Hotel booked  : {hotel_summary}")

    context_block = "\n".join(context_lines)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"TRIP DETAILS:\n{context_block}\n\n"
        f"TOP ATTRACTIONS:\n{attractions_text}\n\n"
        f"Task: Write a detailed {num_days}-day itinerary strictly for {city} and its immediate surroundings only. "
        f"Do NOT include places from other cities. "
        f"Structure each day clearly (Day 1, Day 2, …). "
        f"Include timings, meals, transport between spots, and one practical tip per day.\n\n"
        f"ITINERARY:\n"
    )


def generate_itinerary_phi4(
    city: str,
    attractions: list,
    num_days: int,
    check_in: str,
    num_adults: int = 1,
    flight_summary: str | None = None,
    hotel_summary: str | None = None,
) -> str | None:
    """
    Generate a travel itinerary using phi4-mini via Ollama.

    Returns the itinerary string on success, or None on any failure
    so the caller can fall back to the Llama-based generator.
    """
    prompt = _build_phi4_prompt(
        city=city,
        attractions=attractions,
        num_days=num_days,
        check_in=check_in,
        num_adults=num_adults,
        flight_summary=flight_summary,
        hotel_summary=hotel_summary,
    )

    payload = {
        "model":   PHI4_MINI_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": _PHI4_OPTIONS,
    }

    try:
        logger.info("[Phi4Mini] Sending itinerary request for %s (%d days)", city, num_days)
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()

        text = resp.json().get("response", "").strip()
        if not text:
            logger.warning("[Phi4Mini] Empty response received")
            return None

        logger.info("[Phi4Mini] Itinerary generated successfully (%d chars)", len(text))
        return text

    except requests.exceptions.ConnectionError:
        logger.warning("[Phi4Mini] Ollama not reachable — is `ollama serve` running?")
        return None
    except requests.exceptions.Timeout:
        logger.warning("[Phi4Mini] Request timed out after 120s")
        return None
    except Exception as e:
        logger.error("[Phi4Mini] Unexpected error: %s", e)
        return None


def is_phi4_available() -> bool:
    """Quick health-check: is phi4-mini loaded in Ollama?"""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        available = any(PHI4_MINI_MODEL in m for m in models)
        if not available:
            logger.warning(
                "[Phi4Mini] Model '%s' not found in Ollama. "
                "Run: ollama pull %s",
                PHI4_MINI_MODEL, PHI4_MINI_MODEL,
            )
        return available
    except Exception:
        return False
