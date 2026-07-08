
import os
import sys
import logging
import requests
import traceback
from datetime import datetime, timedelta
from typing import Optional, Dict, List

from serpapi import GoogleSearch

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
    "Srinagar, Kashmir": "SXR"
}

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

    from_city = from_city.split(",")[0].strip()
    to_city = to_city.split(",")[0].strip()

    origin = IATA_CODES.get(from_city, from_city)
    destination = IATA_CODES.get(to_city, to_city)

    params = {
        "engine": "google_flights",
        "departure_id": origin,
        "arrival_id": destination,
        "outbound_date": departure_date,
        "type": "2",
        "currency": "INR",
        "hl": "en",
        "api_key": SERPAPI_KEY,
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()

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


def generate_itinerary(city: str, attractions: List[Dict], num_days: int,
                       check_in: str, num_adults: int = 1,
                       flight_summary: str = None,
                       hotel_summary: str = None) -> str:
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
        "INSTRUCTIONS: Create Day-by-Day plan with timings, meals, and travel tips. "
        "Group nearby attractions. Mix activity types. Keep it realistic and enjoyable.\n\n"
        "ITINERARY:"
    )

    # Use an instruction-following model for prose (NOT the fine-tuned orchestrator,
    # which emits tool-calls). Always return a non-empty string — an empty return
    # would break the MCP itinerary round-trip.
    try:
        import ollama
        model = os.environ.get("ORBIX_TRAVEL_MODEL", "llama3.2:3b")
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.6, "num_predict": 900,
                     "num_gpu": int(os.environ.get("OLLAMA_NUM_GPU", "0"))},
        )
        text = (resp.get("message", {}).get("content") or "").strip()
        if text:
            return text
        logger.warning("[Travel] LLM returned empty itinerary; using deterministic fallback")
    except Exception as e:
        logger.error("[Travel] Itinerary generation failed: %s; using fallback", e)
    return _fallback_itinerary(city, attractions, num_days)
