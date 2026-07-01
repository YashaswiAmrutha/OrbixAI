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

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["/Users/nryashaswiamrutha/Downloads/OrbixAI-main copy/backend/mcp_travel_server.py"]
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

            # Step 5: Itinerary
            _emit(f"Generating {num_nights}-day itinerary...")

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

            print("ITINERARY TYPE:", type(itinerary))
            print("ITINERARY DATA:", itinerary)

            result["itinerary"] = itinerary

            _emit("Itinerary ready!")

            return result


