#travel_planner.py:
"""
Travel Planner pipeline — mirrors the notebook demo.

Steps:
  1. Extract entities  (gpraneeth555/llama-3-13k or Ollama fallback)
  2. Route → Amadeus flight search
  3. Route → Amadeus hotel search
  4. Fetch attractions via OpenStreetMap (free, no API key)
  5. Generate day-by-day itinerary with LLM
"""

import re
import os
import json
import logging
import requests
import sys
import asyncio
import calendar
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable
from llm.ollama_client import generate_response

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from mcp_travel_client import (
    search_flights_mcp,
    search_hotels_mcp,
    get_attractions_mcp,
    generate_itinerary_mcp
)


logger = logging.getLogger(__name__)

TRAVEL_PLACE_ALIASES = {
    "kashmir": "Srinagar",
    "jammu and kashmir": "Srinagar",
}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
#OLLAMA_URL   = "http://localhost:11434/api/generate"

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
- The query may contain previous conversation turns. Inherit destination,
  origin, dates, and trip length from those turns when the current message is
  only answering a clarification question.
- Currency and budget values such as "₹2000", "2000rs", or "budget 2000"
  are money, NEVER a year.
- If a date omits its year, use the current year when still upcoming,
  otherwise use the next year.
- Return ONLY the JSON object, no prose

Query: """


def _validate_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        parsed = datetime.strptime(s, "%Y-%m-%d")
        if parsed.year < datetime.now().year or parsed.year > datetime.now().year + 2:
            return None
        return s
    except ValueError:
        return None


def _infer_month_dates(query: str, num_nights: int = 3):
    """Turn a month-only request such as 'in July' into usable future dates."""
    month_names = {
        name.lower(): number
        for number, name in enumerate(calendar.month_name)
        if name
    }
    month_names.update({
        name.lower(): number
        for number, name in enumerate(calendar.month_abbr)
        if name
    })
    match = re.search(
        r"\b(" + "|".join(sorted(month_names, key=len, reverse=True)) + r")\b",
        query.lower(),
    )
    if not match:
        return None, None

    now = datetime.now()
    month = month_names[match.group(1)]
    year_match = re.search(r"\b(20\d{2})\b", query)
    year = int(year_match.group(1)) if year_match else now.year
    if not year_match and month < now.month:
        year += 1

    if year == now.year and month == now.month:
        departure = (now + timedelta(days=7)).date()
        if departure.month != month:
            departure = now.date() + timedelta(days=1)
    else:
        departure = datetime(year, month, 7).date()

    return (
        departure.strftime("%Y-%m-%d"),
        (departure + timedelta(days=num_nights)).strftime("%Y-%m-%d"),
    )


def _clean_city(value: str) -> str:
    value = re.sub(r"\b(?:airport|flight|flights|hotel|hotels|nearby|near)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" ,.-")
    return value.title() if value else value


def _month_number(month_name: str) -> int:
    names = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
    names.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
    return names[month_name.lower()]


def _resolve_year(month: int, day: int, explicit_year: Optional[str]) -> int:
    if explicit_year:
        return int(explicit_year)
    now = datetime.now()
    year = now.year
    if datetime(year, month, day).date() < now.date():
        year += 1
    return year


def _apply_text_hints(query: str, entities: Dict) -> Dict:
    """Deterministic guardrails for current prompts and same-session follow-ups."""
    month_pattern = "|".join(calendar.month_name[1:] + calendar.month_abbr[1:])

    route_matches = list(re.finditer(
        r"\bfrom\s+([a-z][a-z .'-]{1,45}?)\s+to\s+([a-z][a-z .'-]{1,45}?)"
        r"(?=\s+(?:on|in|for|with|and|generate|create|search|find)|[,.\n]|$)",
        query,
        re.IGNORECASE,
    ))
    if route_matches:
        route = route_matches[-1]
        entities["from_city"] = _clean_city(route.group(1))
        entities["to_city"] = _clean_city(route.group(2))

    reverse_route_matches = list(re.finditer(
        r"\b(?:to\s+go\s+to|go\s+to|to|for)\s+([a-z][a-z .'-]{1,45}?)\s+from\s+([a-z][a-z .'-]{1,45}?)"
        r"(?=\s+(?:on|in|for|with|and|generate|create|search|find)|[,.\n]|$)",
        query,
        re.IGNORECASE,
    ))
    if reverse_route_matches:
        route = reverse_route_matches[-1]
        entities["to_city"] = _clean_city(route.group(1))
        entities["from_city"] = _clean_city(route.group(2))

    origin = re.search(
        r"\b(?:depart(?:ing)?|leav(?:e|ing)|from)\s+(?:from\s+)?([a-z][a-z ]{1,35}?)(?=\s*(?:,|\.|budget|on|$))",
        query, re.IGNORECASE,
    )
    if origin and not entities.get("from_city"):
        entities["from_city"] = origin.group(1).strip().title()

    destination_patterns = [
        r"\btravel\s+plan\s+for\s+([a-z][a-z ]{1,40}?)(?=\s*[\n,.:()-]|$)",
        r"\bflight\s+details\s+for\s+([a-z][a-z ]{1,40}?)(?=\s*[\n,.:()-]|$)",
        r"\bhotel\s+options\s+for\s+([a-z][a-z ]{1,40}?)(?=\s*[\n,.:()-]|$)",
        r"\bitinerary\s+for\s+([a-z][a-z ]{1,40}?)(?=\s*[\n,.:()-]|$)",
        r"\btravel\s+details\s+for\s+([a-z][a-z ]{1,40}?)(?=\s*[\n,.:()-]|$)",
        r"\b(?:go|going|travel|fly|flying)\s+to\s+([a-z][a-z ]{1,40}?)(?=\s+(?:near|and|from|for|on|with)|[,.\n]|$)",
        r"\bhotel(?:s)?(?:\s+to\s+stay)?\s+in\s+([a-z][a-z ]{1,40}?)(?=\s+(?:near|and|from|for|on|with)|[,.\n]|$)",
        r"\b(?:trip|travel|itinerary)\s+(?:for|to|in)\s+([a-z][a-z ]{1,40}?)(?=\s+(?:near|and|from|for|on|with)|[,.\n]|$)",
    ]
    if not entities.get("to_city"):
        for pattern in destination_patterns:
            destination = re.search(pattern, query, re.IGNORECASE)
            if destination:
                entities["to_city"] = _clean_city(destination.group(1))
                break

    nights_match = re.search(r"\b(\d{1,2})\s*[- ]?\s*(?:day|days|night|nights)\b", query, re.IGNORECASE)
    if nights_match:
        entities["num_nights"] = int(nights_match.group(1))

    explicit_single_dates = list(re.finditer(
        rf"\bon\s+({month_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*,?\s*(20\d{{2}}))?",
        query,
        re.IGNORECASE,
    ))
    explicit_single_date = explicit_single_dates[-1] if explicit_single_dates else None
    if explicit_single_date:
        month = _month_number(explicit_single_date.group(1))
        day = int(explicit_single_date.group(2))
        year = _resolve_year(month, day, explicit_single_date.group(3))
        start = datetime(year, month, day)
        entities["check_in"] = start.strftime("%Y-%m-%d")

    date_range = re.search(
        rf"\b({month_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?"
        rf"\s*(?:to|through|-)\s*(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*,?\s*(20\d{{2}}))?",
        query, re.IGNORECASE,
    )
    if date_range and not entities.get("check_in"):
        month = _month_number(date_range.group(1))
        year = _resolve_year(month, int(date_range.group(2)), date_range.group(4))
        start = datetime(year, month, int(date_range.group(2)))
        end = datetime(start.year, month, int(date_range.group(3)))
        entities["check_in"] = start.strftime("%Y-%m-%d")
        entities["check_out"] = end.strftime("%Y-%m-%d")
        entities["num_nights"] = max(1, (end - start).days)

    if not entities.get("check_in"):
        single_dates = list(re.finditer(
            rf"\b(?:on\s+)?({month_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*,?\s*(20\d{{2}}))?",
            query,
            re.IGNORECASE,
        ))
        single_date = single_dates[-1] if single_dates else None
        if single_date:
            month = _month_number(single_date.group(1))
            day = int(single_date.group(2))
            year = _resolve_year(month, day, single_date.group(3))
            start = datetime(year, month, day)
            entities["check_in"] = start.strftime("%Y-%m-%d")

    if entities.get("check_in") and not entities.get("check_out"):
        start = datetime.strptime(entities["check_in"], "%Y-%m-%d")
        entities["check_out"] = (
            start + timedelta(days=entities.get("num_nights") or 3)
        ).strftime("%Y-%m-%d")

    current_message_match = re.search(r"Current user message:\s*(.*)", query, re.IGNORECASE | re.DOTALL)
    current_message = current_message_match.group(1) if current_message_match else query
    if re.search(r"\breturn\s+flight\b", current_message, re.IGNORECASE):
        from_city, to_city = entities.get("from_city"), entities.get("to_city")
        if from_city and to_city:
            entities["from_city"], entities["to_city"] = to_city, from_city
        # A return-flight lookup on a specific date should search that date as
        # the departure date, not preserve the outbound hotel checkout range.
        entities["check_out"] = None

    return entities


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
            
            

            _emit("Extracting travel details with Ollama...")

            raw = generate_response(
                EXTRACTION_PROMPT + query + "\nJSON:"
            )
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
        entities = {
            "from_city":  data.get("from_city", "").strip().title() if data.get("from_city") else None,
            "to_city":    data.get("to_city", "").strip().title() if data.get("to_city") else None,
            "check_in":   _validate_date(data.get("check_in")),
            "check_out":  _validate_date(data.get("check_out")),
            "num_nights": data.get("num_nights") if isinstance(data.get("num_nights"), int) else None,
            "num_adults": data.get("num_adults") if isinstance(data.get("num_adults"), int) else 1,
        }
        entities = _apply_text_hints(query, entities)
        if not entities["check_in"]:
            inferred_in, inferred_out = _infer_month_dates(
                query, entities["num_nights"] or 3
            )
            entities["check_in"] = inferred_in
            entities["check_out"] = inferred_out
        return entities
    except Exception as e:
        logger.error("[Travel] Entity parse error: %s | raw: %s", e, raw[:300])
        return _empty_entities()


def _empty_entities():
    return {"from_city": None, "to_city": None, "check_in": None,
            "check_out": None, "num_nights": None, "num_adults": 1}


def detect_request_scope(text: str) -> Dict[str, bool]:
    """Decide which travel sections the user asked for in the current turn."""
    q = (text or "").lower()
    wants_flights = bool(re.search(
        r"\b(flight|flights|airfare|fare|fares|ticket|tickets|return flight)\b",
        q,
    ))
    wants_hotels = bool(re.search(
        r"\b(hotel|hotels|stay|stays|accommodation|accommodations|room|rooms|resort|lodging)\b",
        q,
    ))
    wants_itinerary = bool(re.search(
        r"\b(itinerary|plan|travel plan|trip plan|schedule|day[- ]?by[- ]?day|"
        r"\d+\s*[- ]?\s*(?:day|days|night|nights)\s+(?:itinerary|plan|trip))\b",
        q,
    ))
    wants_attractions = bool(re.search(
        r"\b(attraction|attractions|places to visit|sightseeing|things to do|visit)\b",
        q,
    ))

    # Broad travel requests like "plan a trip" should produce a full plan.
    wants_full_plan = bool(re.search(
        r"\b(full trip|complete trip|plan (?:a |my )?trip|travel planner|vacation plan)\b",
        q,
    ))

    if wants_full_plan or not any((wants_flights, wants_hotels, wants_itinerary, wants_attractions)):
        return {
            "flights": True,
            "hotels": True,
            "attractions": True,
            "itinerary": True,
            "full_plan": True,
        }

    if wants_itinerary:
        wants_attractions = True

    return {
        "flights": wants_flights,
        "hotels": wants_hotels,
        "attractions": wants_attractions,
        "itinerary": wants_itinerary,
        "full_plan": False,
    }


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

    to_city = entities.get("to_city")
    check_in = entities.get("check_in")
    check_out = entities.get("check_out")
    num_nights = entities.get("num_nights")
    num_adults = entities.get("num_adults", 1)

    if not to_city:
        return "SKIP_HOTEL", None

    # Default dates if not provided
    if not check_in:
        check_in = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

    if not check_out:
        check_out = (
            datetime.strptime(check_in, "%Y-%m-%d")
            + timedelta(days=num_nights or 3)
        ).strftime("%Y-%m-%d")

    return "CALL_HOTEL_API", {
        "city": to_city,
        "check_in": check_in,
        "check_out": check_out,
        "num_adults": num_adults
    }


"Could not generate itinerary: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# 6. MAIN PIPELINE  (with streaming emit callbacks)
# ═══════════════════════════════════════════════════════════════════════════

async def _plan_trip_impl(query: str, emit: Callable[[str], None] = None,
                          request_text: str | None = None) -> Dict:

    print("===== PLAN_TRIP CALLED =====")
    print(query)

    def _emit(msg):
        if emit:
            emit(msg)
        logger.info("[Travel] %s", msg)

    server_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "mcp_travel_server.py",
    )
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
        cwd=os.path.dirname(server_script),
    )

    async with stdio_client(server_params) as (read_stream, write_stream):

        print("STEP 1 - MCP process started")

        async with ClientSession(read_stream, write_stream) as session:

            print("STEP 2 - ClientSession created")

            try:
                await asyncio.wait_for(
                    session.initialize(),
                    timeout=10
                )
            except asyncio.TimeoutError:
                print("MCP initialization timed out")
                raise

            print("STEP 3 - MCP session initialized")

            # Step 1: Entity extraction
            _emit("Extracting travel details...")

            entities = extract_travel_entities(
                query,
                emit=_emit
            )
            scope = detect_request_scope(request_text or query)

            logger.info("[Travel] Entities: %s", entities)
            logger.info("[Travel] Scope: %s", scope)

            to_city = entities.get("to_city")
            from_city = entities.get("from_city")
            check_in = entities.get("check_in")
            num_nights = entities.get("num_nights") or 3
            num_adults = entities.get("num_adults") or 1

            if not to_city:
                return {
                    "error": "Could not determine destination from your query."
                }

            result = {
                "entities": entities,
                "flights": [],
                "hotels": [],
                "attractions": [],
                "itinerary": "",
                "scope": scope,
            }

            # Step 2: Flights
            flight_decision, src, dest = route_flight_api(entities)

            flight_summary = None

            if scope["flights"] and flight_decision == "CALL_FLIGHT_API":

                dep_date = check_in or (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
                search_src = TRAVEL_PLACE_ALIASES.get(src.lower(), src)
                search_dest = TRAVEL_PLACE_ALIASES.get(dest.lower(), dest)
                has_explicit_departure_date = bool(re.search(
                    r"\bon\s+(?:"
                    + "|".join(calendar.month_name[1:] + calendar.month_abbr[1:])
                    + r")\s+\d{1,2}(?:st|nd|rd|th)?",
                    query,
                    re.IGNORECASE,
                ))
                # Flight searches should be one-way by default.  Trip length
                # words like "3-day itinerary" or "4 days" describe the hotel
                # / itinerary window, not a return flight request.  Only pass a
                # return date when the user explicitly asks for a return/round
                # trip flight.
                has_explicit_return_or_duration = bool(
                    re.search(
                        r"\b(?:return(?:ing)?|return\s+flight|round\s*trip|roundtrip|two[-\s]?way)\b",
                        query,
                        re.IGNORECASE,
                    )
                )
                flight_return_date = entities.get("check_out") if has_explicit_return_or_duration else None
                print("=== BEFORE MCP FLIGHT CALL ===")
                if flight_return_date:
                    _emit(
                        f"Searching round-trip flights from {search_src} to {search_dest} "
                        f"on {dep_date}, returning {flight_return_date}..."
                    )
                else:
                    _emit(f"Searching one-way flights from {search_src} to {search_dest} on {dep_date}...")

                try:
                    response = await search_flights_mcp(
                        session,
                        search_src,
                        search_dest,
                        dep_date,
                        flight_return_date
                    )
                except Exception as e:
                    logger.error("[Travel] Flight search failed: %s", e, exc_info=True)
                    response = []
                # Google may expose no inventory for a near-term, month-only
                # request. Retry a nearby future window and label it clearly
                # instead of silently returning an empty plan.
                if not response and dep_date and not has_explicit_departure_date:
                    requested_departure = datetime.strptime(dep_date, "%Y-%m-%d")
                    if requested_departure <= datetime.now() + timedelta(days=31):
                        fallback_departure = datetime.now() + timedelta(days=39)
                        trip_days = num_nights or 3
                        fallback_dep_text = fallback_departure.strftime("%Y-%m-%d")
                        fallback_return_text = (
                            (fallback_departure + timedelta(days=trip_days)).strftime("%Y-%m-%d")
                            if flight_return_date
                            else None
                        )
                        _emit(
                            "No flights found for the requested near-term dates; "
                            f"checking {fallback_dep_text} instead..."
                        )
                        response = await search_flights_mcp(
                            session,
                            search_src,
                            search_dest,
                            fallback_dep_text,
                            fallback_return_text,
                        )
                        for flight in response if isinstance(response, list) else []:
                            flight["departure_date"] = fallback_dep_text
                            if fallback_return_text:
                                flight["return_date"] = fallback_return_text
                            flight["availability_note"] = (
                                f"No inventory was returned for {dep_date}; "
                                f"showing available options for {fallback_dep_text}."
                            )
                print("=== AFTER MCP FLIGHT CALL ===")
                print("=" * 80)
                print(type(response))
                print(repr(response))
                print("=" * 80)
                
                
                print("FLIGHTS DATA:", response)

                result["flights"] = response

                if isinstance(response, list) and len(response) > 0:

                    f = response[0]

                    flight_summary = (
                        f"{f.get('currency','')} "
                        f"{f.get('price','')} | "
                        f"{f.get('departure','')}→{f.get('arrival','')} | "
                        f"{f.get('duration','')}"
                    )

                    _emit(f"Found {len(response)} flight option(s)")

                else:
                    _emit("No flight data available")

            else:
                _emit("Skipping flight search")

            # Step 3: Hotels
            hotel_decision, hotel_params = route_hotel_api(
                entities,
                bool(result["flights"])
            )

            hotel_summary = None

            if scope["hotels"] and hotel_decision == "CALL_HOTEL_API":

                hotel_city = TRAVEL_PLACE_ALIASES.get(to_city.lower(), to_city)
                _emit(f"Searching hotels in {hotel_city}...")

                try:
                    hotels = await asyncio.wait_for(
                        search_hotels_mcp(
                            session,
                            hotel_city,
                            hotel_params["check_in"],
                            hotel_params["check_out"],
                            hotel_params["num_adults"]
                        ),
                        timeout=120
                    )
                except asyncio.TimeoutError:
                    print("HOTEL SEARCH TIMED OUT")
                    hotels = []

                print("HOTELS TYPE:", type(hotels))
                print("HOTELS DATA:", hotels)

                result["hotels"] = hotels

                if isinstance(hotels, list) and len(hotels) > 0:

                    h = hotels[0]

                    hotel_summary = (
                        f"{h.get('name','')} — "
                        f"{h.get('currency','')} "
                        f"{h.get('price','')}/night"
                    )

                    _emit(f"Found {len(hotels)} hotel option(s)")

                else:
                    _emit("No hotel data available")
            else:
                _emit("Skipping hotel search")

            # Step 4: Attractions
            attractions = []

            if scope["attractions"] or scope["itinerary"]:
                _emit(f"Finding attractions in {to_city}...")

                attractions = await get_attractions_mcp(
                    session,
                    to_city
                )

                print("ATTRACTIONS TYPE:", type(attractions))
                print("ATTRACTIONS DATA:", attractions)

                result["attractions"] = attractions

                if isinstance(attractions, list):
                    _emit(f"Found {len(attractions)} attraction(s)")
                else:
                    _emit("Attractions retrieved")
            else:
                _emit("Skipping attractions search")

            # Step 5: Itinerary
            if scope["itinerary"]:
                _emit(f"Generating {num_nights}-day itinerary...")

                try:
                    itinerary = await asyncio.wait_for(
                        generate_itinerary_mcp(
                            session=session,
                            city=to_city,
                            attractions=attractions,
                            num_days=num_nights,
                            check_in=check_in or datetime.now().strftime("%Y-%m-%d"),
                            num_adults=num_adults,
                            flight_summary=flight_summary,
                            hotel_summary=hotel_summary
                        ),
                        timeout=60
                    )
                except asyncio.TimeoutError:
                    print("ITINERARY TIMED OUT")
                    itinerary = {"error": "Timeout"}

                print("ITINERARY TYPE:", type(itinerary))
                print("ITINERARY DATA:", itinerary)

                result["itinerary"] = itinerary

                _emit("Itinerary ready!")
            else:
                _emit("Skipping itinerary generation")

            return result


# anyio's stdio subprocess transport is not reliable when several Playwright
# MCP process trees are launched from the same FastAPI worker simultaneously.
# Queue plans within a worker instead of allowing one request to crash all of
# OrbixAI.
_travel_plan_lock = asyncio.Lock()


async def plan_trip(query: str, emit: Callable[[str], None] = None,
                    request_text: str | None = None) -> Dict:
    async with _travel_plan_lock:
        return await _plan_trip_impl(query, emit=emit, request_text=request_text)
