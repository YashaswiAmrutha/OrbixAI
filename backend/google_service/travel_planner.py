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

# ── Amadeus credentials (override via .env) ─────────────────────────────────
AMADEUS_CLIENT_ID     = os.environ.get("AMADEUS_CLIENT_ID",     "GL4lMSLONHWXs0kroqnYabMGjaqzXAHR")
AMADEUS_CLIENT_SECRET = os.environ.get("AMADEUS_CLIENT_SECRET", "CA25nHIoPpmb1ks6")

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


# JSON schema constraining the extractor's output (Ollama structured output).
_TRAVEL_SCHEMA = {
    "type": "object",
    "properties": {
        "from_city":  {"type": "string"},
        "to_city":    {"type": "string"},
        "check_in":   {"type": "string"},
        "check_out":  {"type": "string"},
        "num_nights": {"type": "integer"},
        "num_adults": {"type": "integer"},
    },
    "required": ["to_city"],
}


def extract_travel_entities(query: str, emit=None) -> Dict:
    """
    Extract travel details as strict JSON via Ollama structured output.

    Uses an instruction-following model (llama3.1:8b by default) with a JSON schema —
    the same reliable pattern the memory extractor uses. The fine-tuned orchestrator
    model is NOT used here: it emits tool-call plans, not extraction JSON.
    """
    import ollama

    def _emit(msg):
        if emit:
            emit(msg)

    _emit("Extracting travel details…")
    model = os.environ.get("ORBIX_TRAVEL_MODEL", "llama3.2:3b")
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT.rstrip().rstrip("Query:").rstrip()},
                {"role": "user", "content": query},
            ],
            format=_TRAVEL_SCHEMA,
            options={"temperature": 0, "num_gpu": int(os.environ.get("OLLAMA_NUM_GPU", "0"))},
        )
        data = json.loads(resp["message"]["content"])
        logger.info("[Travel] extracted entities: %s", data)
    except Exception as e:
        logger.error("[Travel] entity extraction failed: %s", e)
        return _empty_entities()

    return {
        "from_city":  (data.get("from_city") or "").strip().title() or None,
        "to_city":    (data.get("to_city") or "").strip().title() or None,
        "check_in":   _validate_date(data.get("check_in")),
        "check_out":  _validate_date(data.get("check_out")),
        "num_nights": data.get("num_nights") if isinstance(data.get("num_nights"), int) else None,
        "num_adults": data.get("num_adults") if isinstance(data.get("num_adults"), int) else 1,
    }


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

async def plan_trip(query: str, emit: Callable[[str], None] = None) -> Dict:

    print("===== PLAN_TRIP CALLED =====")
    print(query)

    def _emit(msg):
        if emit:
            emit(msg)
        logger.info("[Travel] %s", msg)

    # Resolve the MCP travel server path relative to this file so it works on any
    # machine/OS (was previously hardcoded to a teammate's macOS Downloads folder).
    _backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _server_script = os.path.join(_backend_dir, "mcp_travel_server.py")
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[_server_script],
        cwd=_backend_dir,  # so the subprocess can import google_service.* / mcp_travel_client
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

            logger.info("[Travel] Entities: %s", entities)

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
                "itinerary": ""
            }

            # Step 2: Flights
            flight_decision, src, dest = route_flight_api(entities)

            flight_summary = None

            if flight_decision == "CALL_FLIGHT_API":

                dep_date = check_in or datetime.now().strftime("%Y-%m-%d")
                print("=== BEFORE MCP FLIGHT CALL ===")
                _emit(f"Searching flights from {src} to {dest}...")

                response = await asyncio.wait_for(
                    search_flights_mcp(
                        session,
                        src,
                        dest,
                        dep_date
                    ),
                    timeout=20
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

            if hotel_decision == "CALL_HOTEL_API":

                _emit(f"Searching hotels in {to_city}...")

                try:
                    hotels = await asyncio.wait_for(
                        search_hotels_mcp(
                            session,
                            hotel_params["city"],
                            hotel_params["check_in"],
                            hotel_params["check_out"],
                            hotel_params["num_adults"]
                        ),
                        timeout=20
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

            # Step 4: Attractions
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

            # Step 5: Itinerary (via MCP; falls back to an in-process build if MCP
            # returns nothing, so a demo never ends up with an empty itinerary).
            _emit(f"Generating {num_nights}-day itinerary...")

            itinerary = ""
            try:
                itinerary = await generate_itinerary_mcp(
                    session=session,
                    city=to_city,
                    attractions=attractions,
                    num_days=num_nights,
                    check_in=check_in or datetime.now().strftime("%Y-%m-%d"),
                    num_adults=num_adults,
                    flight_summary=flight_summary,
                    hotel_summary=hotel_summary
                )
            except Exception as e:
                logger.error("[Travel] MCP itinerary failed: %s", e)

            _it_str = str(itinerary or "").strip()
            if (not _it_str) or _it_str.lower().startswith("error executing tool") \
                    or "validation error" in _it_str.lower():
                logger.warning("[Travel] Bad/empty MCP itinerary — building in-process fallback")
                from google_service.travel_services import generate_itinerary as _gi
                itinerary = _gi(
                    city=to_city, attractions=attractions, num_days=num_nights,
                    check_in=check_in or datetime.now().strftime("%Y-%m-%d"),
                    num_adults=num_adults,
                    flight_summary=flight_summary, hotel_summary=hotel_summary,
                )

            result["itinerary"] = itinerary

            _emit("Itinerary ready!")

            return result


