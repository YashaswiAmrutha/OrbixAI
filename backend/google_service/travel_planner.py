"""
Travel Planner pipeline — mirrors the notebook demo.

Steps:
  1. Extract entities  (gpraneeth555/llama-3-13k or Ollama fallback)
  2. Route → Amadeus flight search
  3. Route → Amadeus hotel search
  4. Fetch attractions via OpenStreetMap (free, no API key)
  5. Generate day-by-day itinerary with LLM
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable

logger = logging.getLogger(__name__)

# ── Amadeus credentials (override via .env) ─────────────────────────────────
AMADEUS_CLIENT_ID     = os.environ.get("AMADEUS_CLIENT_ID",     "GL4lMSLONHWXs0kroqnYabMGjaqzXAHR")
AMADEUS_CLIENT_SECRET = os.environ.get("AMADEUS_CLIENT_SECRET", "CA25nHIoPpmb1ks6")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OLLAMA_URL   = "http://localhost:11434/api/generate"

# ═══════════════════════════════════════════════════════════════════════════
# 1. ENTITY EXTRACTION  (llama-3-13k → Ollama fallback)
# ═══════════════════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """You are a travel information extraction assistant.
Extract ONLY the following fields from the query and return strict JSON.
Use null for missing values.

Required JSON shape:
{
  "from_city": "CityName or null",
  "to_city": "CityName or null",
  "check_in": "YYYY-MM-DD or null",
  "check_out": "YYYY-MM-DD or null",
  "num_nights": integer_or_null,
  "num_adults": integer_or_null
}

Rules:
- Convert natural dates ("27 jan 2026") to YYYY-MM-DD
- "X days" → num_nights = X
- Default num_adults to 1 if not mentioned
- Return ONLY the JSON object, no prose

Query: """


def _validate_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return s
    except ValueError:
        return None


def extract_travel_entities(query: str, emit=None) -> Dict:
    """Extract travel details using gpraneeth555/llama-3-13k (HF) or Ollama."""
    import re

    def _emit(msg):
        if emit:
            emit(msg)

    # ── Try HuggingFace model first ──────────────────────────────────────────
    raw = None
    try:
        from llm.hf_client import call_hf_model
        _emit("Extracting travel details with llama-3-13k…")
        prompt_text = EXTRACTION_PROMPT + query + "\nJSON:"
        raw = call_hf_model(prompt_text, max_new_tokens=256)
        logger.info("[Travel] HF extraction raw: %s", raw[:200] if raw else "empty")
    except Exception as e:
        logger.warning("[Travel] HF extraction failed (%s), falling back to Ollama", e)

    # ── Ollama fallback ──────────────────────────────────────────────────────
    if not raw:
        try:
            _emit("Extracting travel details with Ollama…")
            payload = {
                "model": "llama3.2:latest",
                "prompt": EXTRACTION_PROMPT + query + "\nJSON:",
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 256}
            }
            resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
            resp.raise_for_status()
            raw = resp.json().get("response", "")
        except Exception as e:
            logger.error("[Travel] Ollama extraction also failed: %s", e)
            return _empty_entities()

    # ── Parse JSON ────────────────────────────────────────────────────────────
    try:
        raw_clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        match = re.search(r"\{[\s\S]*\}", raw_clean)
        if not match:
            raise ValueError("No JSON found in extraction output")
        data = json.loads(match.group())
        return {
            "from_city":  data.get("from_city", "").strip().title() if data.get("from_city") else None,
            "to_city":    data.get("to_city", "").strip().title() if data.get("to_city") else None,
            "check_in":   _validate_date(data.get("check_in")),
            "check_out":  _validate_date(data.get("check_out")),
            "num_nights": data.get("num_nights") if isinstance(data.get("num_nights"), int) else None,
            "num_adults": data.get("num_adults") if isinstance(data.get("num_adults"), int) else 1,
        }
    except Exception as e:
        logger.error("[Travel] Entity parse error: %s | raw: %s", e, raw[:300])
        return _empty_entities()


def _empty_entities():
    return {"from_city": None, "to_city": None, "check_in": None,
            "check_out": None, "num_nights": None, "num_adults": 1}


# ═══════════════════════════════════════════════════════════════════════════
# 2. ROUTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def route_flight_api(entities: Dict):
    """Returns (decision, src, dest)."""
    to_city   = entities.get("to_city")
    from_city = entities.get("from_city")
    if not to_city:
        return "SKIP_FLIGHT", None, None
    if not from_city:
        return "SKIP_FLIGHT", None, to_city
    if from_city.lower() == to_city.lower():
        return "SKIP_FLIGHT", from_city, to_city
    return "CALL_FLIGHT_API", from_city, to_city


def route_hotel_api(entities: Dict, has_flights: bool):
    """Returns (decision, params | None)."""
    to_city   = entities.get("to_city")
    check_in  = entities.get("check_in")
    check_out = entities.get("check_out")
    num_nights = entities.get("num_nights")
    num_adults = entities.get("num_adults", 1)
    if not to_city:
        return "SKIP_HOTEL", None
    if check_in and check_out:
        return "CALL_HOTEL_API", {"city": to_city, "check_in": check_in, "check_out": check_out, "num_adults": num_adults}
    if check_in and num_nights:
        co = (datetime.strptime(check_in, "%Y-%m-%d") + timedelta(days=num_nights)).strftime("%Y-%m-%d")
        return "CALL_HOTEL_API", {"city": to_city, "check_in": check_in, "check_out": co, "num_adults": num_adults}
    return "SKIP_HOTEL", None


# ═══════════════════════════════════════════════════════════════════════════
# 3. AMADEUS  (flights + hotels)
# ═══════════════════════════════════════════════════════════════════════════

def _amadeus_client():
    try:
        from amadeus import Client
        return Client(client_id=AMADEUS_CLIENT_ID, client_secret=AMADEUS_CLIENT_SECRET)
    except ImportError:
        logger.warning("[Travel] amadeus package not installed — skipping Amadeus calls")
        return None
    except Exception as e:
        logger.error("[Travel] Amadeus init error: %s", e)
        return None


def _city_to_iata(amadeus, city: str) -> Optional[str]:
    try:
        locs = amadeus.reference_data.locations.get(keyword=city, subType="CITY")
        return locs.data[0]["iataCode"] if locs.data else None
    except Exception as e:
        logger.error("[Travel] IATA lookup failed for %s: %s", city, e)
        return None


def search_flights(from_city: str, to_city: str, departure_date: str,
                   adults: int = 1, max_results: int = 5) -> List[Dict]:
    amadeus = _amadeus_client()
    if not amadeus:
        return []
    try:
        origin = _city_to_iata(amadeus, from_city)
        dest   = _city_to_iata(amadeus, to_city)
        if not origin or not dest:
            logger.warning("[Travel] Could not resolve IATA codes: %s→%s", from_city, to_city)
            return []
        offers = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=dest,
            departureDate=departure_date,
            adults=adults,
            max=max_results
        ).data
        results = []
        for o in offers:
            price     = o["price"]["total"]
            currency  = o["price"]["currency"]
            segments  = o["itineraries"][0]["segments"]
            dep_time  = segments[0]["departure"]["at"].split("T")[1][:5]
            arr_time  = segments[-1]["arrival"]["at"].split("T")[1][:5]
            duration  = o["itineraries"][0]["duration"].replace("PT","").replace("H","h ").replace("M","m")
            results.append({
                "price": price, "currency": currency,
                "departure": dep_time, "arrival": arr_time,
                "duration": duration, "stops": len(segments) - 1
            })
        return results
    except Exception as e:
        logger.error("[Travel] Flight search error: %s", e)
        return []


def search_hotels(city: str, check_in: str, check_out: str,
                  num_adults: int = 1, max_results: int = 5) -> List[Dict]:
    amadeus = _amadeus_client()
    if not amadeus:
        return []
    try:
        city_code = _city_to_iata(amadeus, city)
        if not city_code:
            return []
        hotels_resp = amadeus.reference_data.locations.hotels.by_city.get(cityCode=city_code)
        hotel_ids   = [h["hotelId"] for h in hotels_resp.data[:20]]
        if not hotel_ids:
            return []
        offers_resp = amadeus.shopping.hotel_offers_search.get(
            hotelIds=hotel_ids, checkInDate=check_in, checkOutDate=check_out,
            adults=num_adults, roomQuantity=1
        )
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
            if not lat or not lon:
                continue
            key = (round(lat, 3), round(lon, 3))
            if key in seen:
                continue
            seen.add(key)
            results.append({"name": name, "category": _category(tags),
                             "score": _score(tags), "lat": lat, "lon": lon})
        return results

    def _query_geocode(city_name):
        q = f"""[out:json][timeout:30];
{{geocodeArea:{city_name}}}->.a;
(nwr["tourism"~"museum|attraction|viewpoint"](area.a);
 nwr["amenity"~"aquarium|zoo|arts_centre|theatre"](area.a);
 nwr["leisure"="park"]["name"](area.a);
 nwr["man_made"="tower"](area.a);
 nwr["historic"~"monument|memorial|fort|palace"](area.a);
 nwr["building"="cathedral"](area.a););
out center tags 100;"""
        r = requests.post(OVERPASS_URL, data={"data": q},
                          headers={"User-Agent": "OrbixAI/1.0"}, timeout=60)
        r.raise_for_status()
        return r.json().get("elements", [])

    def _query_bbox(city_name):
        nr = requests.get("https://nominatim.openstreetmap.org/search",
                          params={"q": city_name, "format": "json", "limit": 1},
                          headers={"User-Agent": "OrbixAI/1.0"}, timeout=10)
        data = nr.json()
        if not data:
            return []
        bb = data[0].get("boundingbox")  # [min_lat, max_lat, min_lon, max_lon]
        if not bb:
            return []
        s, n, w, e = bb[0], bb[1], bb[2], bb[3]
        q = f"""[out:json][timeout:30];
(nwr["tourism"~"museum|attraction|viewpoint"]({s},{w},{n},{e});
 nwr["amenity"~"aquarium|zoo|arts_centre|theatre"]({s},{w},{n},{e});
 nwr["leisure"="park"]["name"]({s},{w},{n},{e});
 nwr["man_made"="tower"]({s},{w},{n},{e});
 nwr["historic"~"monument|memorial|fort|palace"]({s},{w},{n},{e});
 nwr["building"="cathedral"]({s},{w},{n},{e}););
out center tags 100;"""
        r = requests.post(OVERPASS_URL, data={"data": q},
                          headers={"User-Agent": "OrbixAI/1.0"}, timeout=60)
        r.raise_for_status()
        return r.json().get("elements", [])

    try:
        elements = _query_geocode(city)
        if not elements:
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

def generate_itinerary(city: str, attractions: List[Dict], num_days: int,
                       check_in: str, num_adults: int = 1,
                       flight_summary: str = None,
                       hotel_summary: str = None) -> str:
    from llm.ollama_client import generate_response

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

    try:
        return generate_response(prompt)
    except Exception as e:
        logger.error("[Travel] Itinerary generation failed: %s", e)
        return f"Could not generate itinerary: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# 6. MAIN PIPELINE  (with streaming emit callbacks)
# ═══════════════════════════════════════════════════════════════════════════

def plan_trip(query: str, emit: Callable[[str], None] = None) -> Dict:
    """
    Full pipeline. `emit(step_text)` is called at each major step so the
    caller can stream progress to the frontend.
    Returns a dict with all results.
    """
    def _emit(msg):
        if emit:
            emit(msg)
        logger.info("[Travel] %s", msg)

    # Step 1: Entity extraction
    _emit("Extracting travel details…")
    entities = extract_travel_entities(query, emit=_emit)
    logger.info("[Travel] Entities: %s", entities)

    to_city    = entities.get("to_city")
    from_city  = entities.get("from_city")
    check_in   = entities.get("check_in")
    num_nights = entities.get("num_nights") or 3
    num_adults = entities.get("num_adults") or 1

    if not to_city:
        return {"error": "Could not determine destination from your query. Please specify a destination city."}

    result = {"entities": entities, "flights": [], "hotels": [], "attractions": [], "itinerary": ""}

    # Step 2: Flights
    flight_decision, src, dest = route_flight_api(entities)
    flight_summary = None
    if flight_decision == "CALL_FLIGHT_API":
        dep_date = check_in or datetime.now().strftime("%Y-%m-%d")
        _emit(f"Searching flights from {src} to {dest}…")
        flights = search_flights(src, dest, dep_date, adults=num_adults)
        result["flights"] = flights
        if flights:
            f = flights[0]
            flight_summary = f"{f['currency']} {f['price']} | {f['departure']}→{f['arrival']} | {f['duration']}"
            _emit(f"Found {len(flights)} flight option(s)")
        else:
            _emit("No flight data available")
    else:
        _emit("Skipping flight search (same city or missing origin)")

    # Step 3: Hotels
    hotel_decision, hotel_params = route_hotel_api(entities, bool(result["flights"]))
    hotel_summary = None
    if hotel_decision == "CALL_HOTEL_API":
        _emit(f"Searching hotels in {to_city}…")
        hotels = search_hotels(
            hotel_params["city"], hotel_params["check_in"],
            hotel_params["check_out"], hotel_params["num_adults"]
        )
        result["hotels"] = hotels
        if hotels:
            h = hotels[0]
            hotel_summary = f"{h['name']} — {h['currency']} {h['price']}/night"
            _emit(f"Found {len(hotels)} hotel option(s)")
        else:
            _emit("No hotel data available")
    else:
        _emit("Skipping hotel search (no dates provided)")

    # Step 4: Attractions
    _emit(f"Finding top attractions in {to_city}…")
    attractions = get_attractions(to_city, max_attractions=20)
    result["attractions"] = attractions
    _emit(f"Found {len(attractions)} attraction(s)")

    # Step 5: Itinerary
    _emit(f"Generating {num_nights}-day itinerary…")
    itinerary = generate_itinerary(
        city=to_city,
        attractions=attractions,
        num_days=num_nights,
        check_in=check_in or datetime.now().strftime("%Y-%m-%d"),
        num_adults=num_adults,
        flight_summary=flight_summary,
        hotel_summary=hotel_summary
    )
    result["itinerary"] = itinerary
    _emit("Itinerary ready!")

    return result
