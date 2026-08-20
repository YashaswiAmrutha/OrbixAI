
import os
import sys
import logging
import requests
import traceback
import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Optional, Dict, List

from llm.ollama_options import ollama_options

logger = logging.getLogger(__name__)


# ── Amadeus credentials (override via .env) ─────────────────────────────────
SERPAPI_KEY = os.getenv(
    "SERPAPI_KEY",
    "5ab2608cd458efa6c7db985d21da7e56159f6e42084828fa5eb63deba0c4a64a"
)

# ── Amadeus credentials (override via .env) ─────────────────────────────────
AMADEUS_CLIENT_ID     = os.environ.get("AMADEUS_CLIENT_ID",     "GL4lMSLONHWXs0kroqnYabMGjaqzXAHR")
AMADEUS_CLIENT_SECRET = os.environ.get("AMADEUS_CLIENT_SECRET", "CA25nHIoPpmb1ks6")


OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OLLAMA_URL   = "http://localhost:11434/api/generate"

IATA_CODES = {
    "Bengaluru": "BLR",
    "Bangalore": "BLR",
    "Delhi": "DEL",
    "Mumbai": "BOM",
    "Chennai": "MAA",
    "Hyderabad": "HYD",
    "Kolkata": "CCU",
    "Srinagar": "SXR",
    "Kashmir": "SXR",
    "Srinagar, Kashmir": "SXR"
}

_IATA_CACHE: dict[str, str | None] = {}


def _serpapi_json(params: dict) -> dict:
    """Call SerpAPI with an explicit network deadline."""
    timeout = float(os.environ.get("ORBIX_SERPAPI_TIMEOUT_S", "12"))
    response = requests.get(
        "https://serpapi.com/search.json",
        params=params,
        timeout=(5, timeout),
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload


def resolve_iata_code(place: str) -> Optional[str]:
    """Resolve an arbitrary city/airport name to an IATA code.

    The Google Flights endpoint requires an IATA code or Google location ID; a
    raw city string often produces an empty result with no error. Known codes
    stay instant, while unknown cities are resolved through the already
    configured SerpAPI Google search and cached for the life of the backend.
    """
    place = (place or "").split(",")[0].strip()
    if not place:
        return None
    if re.fullmatch(r"[A-Za-z]{3}", place):
        return place.upper()

    key = place.casefold()
    known = {name.casefold(): code for name, code in IATA_CODES.items()}
    if key in known:
        return known[key]
    if key in _IATA_CACHE:
        return _IATA_CACHE[key]

    params = {
        "engine": "google",
        "q": f"{place} main commercial airport IATA code",
        "hl": "en",
        "api_key": SERPAPI_KEY,
    }
    try:
        payload = _serpapi_json(params)
        candidates: list[str] = []
        for row in (payload.get("organic_results") or [])[:5]:
            text = f"{row.get('title', '')} {row.get('snippet', '')}"
            for pattern in (
                r"\bIATA(?:\s+(?:airport\s+)?code)?\s*[:\-]?\s*([A-Z]{3})\b",
                r"\(([A-Z]{3})\)\s*(?:is|,|\-|airport)",
                r"\bairport\s+code\s+(?:is\s+|for\s+[^—-]+[—-]\s*)?([A-Z]{3})\b",
            ):
                match = re.search(pattern, text, re.I)
                if match:
                    candidates.append(match.group(1).upper())
                    break
        code = Counter(candidates).most_common(1)[0][0] if candidates else None
        _IATA_CACHE[key] = code
        logger.info("Resolved airport location %r -> %s", place, code or "not found")
        return code
    except Exception as exc:
        logger.warning("Airport-code resolution failed for %r: %s", place, exc)
        return None

# ═══════════════════════════════════════════════════════════════════════════
# 3. AMADEUS  (flights + hotels)
# ═══════════════════════════════════════════════════════════════════════════

def _amadeus_client():
    print("CLIENT ID:", AMADEUS_CLIENT_ID)
    print("CLIENT SECRET:", AMADEUS_CLIENT_SECRET[:5] + "*****")

    from amadeus import Client

    return Client(
        client_id=AMADEUS_CLIENT_ID,
        client_secret=AMADEUS_CLIENT_SECRET
    )


def _city_to_iata(amadeus, city):
    try:
        response = amadeus.reference_data.locations.get(
            keyword=city,
            subType=["CITY"]
        )

        print("\n========================")
        print("Searching:", city)
        print(response.data)
        print("========================")

        if response.data:
            return response.data[0]["iataCode"]

        return None

    except Exception as e:
        print("CITY LOOKUP ERROR")
        print(type(e))
        print(e)

        if hasattr(e, "response"):
            print(e.response.body)

        return None



def search_flights(from_city: str, to_city: str,
    departure_date: str, adults: int = 1, max_results: int = 5,):

    print("ENTERED search_flights()")

    origin = resolve_iata_code(from_city)
    destination = resolve_iata_code(to_city)
    if not origin or not destination:
        logger.warning("Cannot search flights: unresolved route %r -> %r", from_city, to_city)
        return []

    params = {
        "engine": "google_flights",
        "departure_id": origin,
        "arrival_id": destination,
        "outbound_date": departure_date,
        "type": "2",
        "travel_class": "1",
        "adults": adults,
        "currency": "INR",
        "hl": "en",
        "api_key": SERPAPI_KEY,
    }

    try:
        results = _serpapi_json(params)

        flights = []

        for item in results.get("best_flights", [])[:max_results]:

            first = item["flights"][0]
            last = item["flights"][-1]

            flights.append({

                "airline": first["airline"],

                "price": item["price"],

                "currency": "INR",

                "departure": first["departure_airport"]["time"],

                "arrival": last["arrival_airport"]["time"],

                "duration": item["total_duration"],

                "stops": len(item["flights"]) - 1

            })

        return flights

    except Exception as e:

        logger.error(e)

        return []


def search_hotels(city: str, check_in: str, check_out: str,
                  num_adults: int = 1, max_results: int = 5) -> List[Dict]:
    print("ENTERED search_hotels()")

    amadeus = _amadeus_client()
    if not amadeus:
        return []
    try:
        print("Looking up city...")
        city_code = _city_to_iata(amadeus, city)
        print("CITY CODE:", city_code)
        if not city_code:
            return []
        print("Fetching hotel list...")
        hotels_resp = amadeus.reference_data.locations.hotels.by_city.get(cityCode=city_code)
        print("Hotel list fetched")
        hotel_ids   = [h["hotelId"] for h in hotels_resp.data[:20]]
        if not hotel_ids:
            return []
        offers_resp = amadeus.shopping.hotel_offers_search.get(
            hotelIds=hotel_ids, checkInDate=check_in, checkOutDate=check_out,
            adults=num_adults, roomQuantity=1
        )
        print("Offers received")
        results = []
        for item in offers_resp.data[:max_results]:
            hotel = item.get("hotel", {})
            offer = item.get("offers", [{}])[0]
            results.append({
                "name":      hotel.get("name"),
                "price":     offer.get("price", {}).get("total"),
                "currency":  offer.get("price", {}).get("currency"),
                "room_type": offer.get("room", {}).get("typeEstimated", {}).get("category"),
            })
        return results
    except Exception as e:
        logger.error("[Travel] Hotel search error: %s", e)
        return []


# ═══════════════════════════════════════════════════════════════════════════
# 4. OPENSTREETMAP ATTRACTIONS  (free, no API key)
# ═══════════════════════════════════════════════════════════════════════════

def _score(tags: Dict) -> int:
    s = 0
    if "wikidata"  in tags: s += 15
    if "wikipedia" in tags: s += 10
    t = tags.get("tourism",""); h = tags.get("historic",""); a = tags.get("amenity","")
    if t == "museum":             s += 12
    if t == "attraction":         s += 10
    if h in ("fort","palace","monument"): s += 11
    if a in ("aquarium","zoo"):   s += 9
    if tags.get("man_made") == "tower": s += 10
    if tags.get("leisure") == "park":   s += 7
    if t == "viewpoint":          s += 6
    if "website"       in tags:   s += 3
    if "opening_hours" in tags:   s += 2
    return s


def _category(tags: Dict) -> str:
    if tags.get("tourism") == "museum":  return "Museum"
    if tags.get("historic"):             return "Historical Site"
    if tags.get("leisure") == "park":    return "Park"
    if tags.get("amenity") in ("aquarium","zoo"): return "Family Attraction"
    if tags.get("tourism") == "viewpoint": return "Viewpoint"
    if tags.get("man_made") == "tower":  return "Landmark"
    return "Tourist Attraction"


def get_attractions(city: str, max_attractions: int = 20) -> List[Dict]:
    """Fetch attractions from OpenStreetMap (Overpass API)."""

    def _parse(elements):
        seen, results = set(), []

        for el in elements:
            tags = el.get("tags", {})
            name = tags.get("name")

            if not name or len(name) < 3:
                continue

            lat = el.get("lat") or el.get("center", {}).get("lat")
            lon = el.get("lon") or el.get("center", {}).get("lon")

            if lat is None or lon is None:
                continue

            key = (round(lat, 3), round(lon, 3))
            if key in seen:
                continue

            seen.add(key)

            results.append({
                "name": name,
                "category": _category(tags),
                "score": _score(tags),
                "lat": lat,
                "lon": lon
            })

        return results


    def _query_bbox(city_name):
        # Get city's bounding box from Nominatim
        nr = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": city_name,
                "format": "json",
                "limit": 1
            },
            headers={"User-Agent": "OrbixAI/1.0"},
            timeout=10
        )

        nr.raise_for_status()

        data = nr.json()

        if not data:
            return []

        bb = data[0].get("boundingbox")

        if not bb:
            return []

        south, north, west, east = bb[0], bb[1], bb[2], bb[3]

        q = f"""
[out:json][timeout:30];
(
  nwr["tourism"~"museum|attraction|viewpoint"]({south},{west},{north},{east});
  nwr["amenity"~"aquarium|zoo|arts_centre|theatre"]({south},{west},{north},{east});
  nwr["leisure"="park"]({south},{west},{north},{east});
  nwr["man_made"="tower"]({south},{west},{north},{east});
  nwr["historic"~"monument|memorial|fort|palace"]({south},{west},{north},{east});
  nwr["building"="cathedral"]({south},{west},{north},{east});
);
out center tags;
"""

        print("===== OVERPASS QUERY =====")
        print(q)

        r = requests.post(
            OVERPASS_URL,
            data={"data": q},
            headers={"User-Agent": "OrbixAI/1.0"},
            timeout=60
        )

        r.raise_for_status()

        return r.json().get("elements", [])


    try:
        elements = _query_bbox(city)
    except Exception as e:
        logger.error("[Travel] OSM query failed: %s", e)
        elements = []

    attractions = _parse(elements)
    attractions.sort(key=lambda x: x["score"], reverse=True)

    return attractions[:max_attractions] 


# ═══════════════════════════════════════════════════════════════════════════
# 5. ITINERARY GENERATION  (LLM)
# ═══════════════════════════════════════════════════════════════════════════

def _fallback_itinerary(city: str, attractions: List[Dict], num_days: int) -> str:
    """Deterministic day-by-day plan from the attractions list (never empty).
    Uses the 'Day N:' format so the calendar parser can turn it into events."""
    num_days = max(1, int(num_days or 1))
    names = [a.get("name") for a in attractions if a.get("name")] or [f"Explore {city}"]
    per_day = max(1, -(-len(names) // num_days))  # ceil
    lines = [f"# {num_days}-Day Trip to {city}", ""]
    idx = 0
    for d in range(1, num_days + 1):
        chunk = names[idx:idx + per_day] or [f"Free time in {city}"]
        idx += per_day
        lines.append(f"Day {d}: " + ", ".join(chunk[:3]))
        for spot in chunk:
            lines.append(f"  - Visit {spot}")
        lines.append("")
    return "\n".join(lines).strip()


def _itinerary_day_numbers(text: str) -> set[int]:
    """Return every explicit Day N heading present in an itinerary."""
    return {
        int(value)
        for value in re.findall(
            r"(?im)^\s*(?:#{1,6}\s*)?(?:\*{1,2})?day\s+(\d+)\b",
            text or "",
        )
    }


def _fallback_day_sections(city: str, attractions: List[Dict], num_days: int,
                           wanted: set[int]) -> str:
    """Extract only missing day sections from the deterministic fallback."""
    fallback = _fallback_itinerary(city, attractions, num_days)
    matches = list(re.finditer(r"(?im)^day\s+(\d+)\b", fallback))
    sections = []
    for index, match in enumerate(matches):
        day = int(match.group(1))
        if day not in wanted:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(fallback)
        sections.append(fallback[match.start():end].strip())
    return "\n\n".join(sections)


def generate_itinerary(city: str, attractions: List[Dict], num_days: int,
                       check_in: str, num_adults: int = 1,
                       flight_summary: str = None,
                       hotel_summary: str = None) -> str:
    num_days = max(1, min(int(num_days or 1), 30))
    attractions_text = "\n".join(
        f"- {a['name']} ({a['category']})"
        for a in attractions[:15]
    )

    ctx_parts = [
        f"Destination: {city}",
        f"Trip duration: {num_days} day(s)",
        f"Check-in: {check_in}",
        f"Travelers: {num_adults}",
    ]
    if flight_summary: ctx_parts.append(f"Flight: {flight_summary}")
    if hotel_summary:  ctx_parts.append(f"Hotel: {hotel_summary}")

    prompt = (
        f"You are an expert travel planner. Create a detailed {num_days}-day itinerary for {city}.\n\n"
        f"TRIP DETAILS:\n" + "\n".join(ctx_parts) + "\n\n"
        f"AVAILABLE ATTRACTIONS:\n{attractions_text}\n\n"
        "INSTRUCTIONS: Include every heading from Day 1 through Day "
        f"{num_days}; never stop before the final day. Under each day use 4-6 concise "
        "bullets covering morning, lunch, afternoon, and evening. Keep each day under "
        "90 words. Include useful timings and meal suggestions, group nearby places, "
        "and avoid a long introduction or conclusion.\n\n"
        "ITINERARY:"
    )

    # Use an instruction-following model for prose (NOT the fine-tuned orchestrator,
    # which emits tool-calls). Always return a non-empty string — an empty return
    # would break the MCP itinerary round-trip.
    try:
        import ollama
        from llm.model_registry import get_instruction_model
        model = get_instruction_model()
        # This is an output-token ceiling, not a requested length. Scale it with
        # the requested duration so a four-day plan cannot be cut off by the same
        # fixed budget used for a one-day plan.
        output_budget = max(900, min(3800, 200 + (120 * num_days)))
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options=ollama_options(
                temperature=0.5, num_predict=output_budget, num_ctx=8192
            ),
        )
        text = (resp.get("message", {}).get("content") or "").strip()
        if text:
            expected = set(range(1, num_days + 1))
            missing = expected - _itinerary_day_numbers(text)
            if missing:
                logger.warning(
                    "[Travel] itinerary omitted days %s; requesting continuation",
                    sorted(missing),
                )
                continuation_prompt = (
                    f"Continue this {num_days}-day {city} itinerary. Produce ONLY the "
                    f"missing days: {', '.join(f'Day {d}' for d in sorted(missing))}. "
                    "Use one heading per day and 4-6 concise bullets covering morning, "
                    "lunch, afternoon, and evening. Do not repeat completed days.\n\n"
                    f"EXISTING ITINERARY:\n{text}"
                )
                try:
                    continuation = ollama.chat(
                        model=model,
                        messages=[{"role": "user", "content": continuation_prompt}],
                        options=ollama_options(
                            temperature=0.4,
                            num_predict=max(600, min(2400, 180 * len(missing))),
                            num_ctx=8192,
                        ),
                    )
                    extra = (
                        continuation.get("message", {}).get("content") or ""
                    ).strip()
                    if extra:
                        text = text.rstrip() + "\n\n" + extra
                except Exception as continuation_error:
                    logger.warning(
                        "[Travel] itinerary continuation failed: %s",
                        continuation_error,
                    )

                # Never return a structurally incomplete plan. If the model still
                # omitted a requested day, append deterministic day sections.
                still_missing = expected - _itinerary_day_numbers(text)
                if still_missing:
                    text += "\n\n" + _fallback_day_sections(
                        city, attractions, num_days, still_missing
                    )
            return text
        logger.warning("[Travel] LLM returned empty itinerary; using deterministic fallback")
    except Exception as e:
        logger.error("[Travel] Itinerary generation failed: %s; using fallback", e)
    return _fallback_itinerary(city, attractions, num_days)
