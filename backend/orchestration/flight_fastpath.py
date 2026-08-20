"""Deterministic live flight-price lookup for clearly specified requests.

Flight fares are current external data, so a small model must not narrate a
pretend web_search call.  This parser normalizes the common request shape and
calls the project's existing Google Flights/SerpAPI service directly.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from google_service.travel_services import resolve_iata_code, search_flights


LOCAL_TZ = ZoneInfo("Asia/Kolkata")
_MONTHS = {
    name: number
    for number, names in enumerate(
        [(), ("jan", "january"), ("feb", "february"), ("mar", "march"),
         ("apr", "april"), ("may",), ("jun", "june"), ("jul", "july"),
         ("aug", "august"), ("sep", "sept", "september"),
         ("oct", "october"), ("nov", "november"), ("dec", "december")]
    )
    for name in names
}
_CITY_ALIASES = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "kashmir": "Srinagar",
    "srinagar": "Srinagar",
}
_CITY_STATE_DESTINATIONS = {"singapore", "hong kong", "macau", "macao"}


def _is_country_name(value: str) -> bool:
    """Use Babel's CLDR territory names when available; fail open if absent."""
    try:
        from babel import Locale
        countries = {str(name).casefold() for name in Locale("en").territories.values()}
        normalized = value.strip().casefold()
        return normalized in countries and normalized not in _CITY_STATE_DESTINATIONS
    except Exception:
        return False


def _clean_city(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" ,.?\t\n")
    return _CITY_ALIASES.get(value.lower(), value.title())


def _parse_date(text: str, now: datetime):
    lower = text.lower()
    if "day after tomorrow" in lower:
        return (now + timedelta(days=2)).date()
    if "tomorrow" in lower:
        return (now + timedelta(days=1)).date()
    if "today" in lower:
        return now.date()

    iso = re.search(r"\b20\d{2}-\d{2}-\d{2}\b", text)
    if iso:
        try:
            return datetime.strptime(iso.group(0), "%Y-%m-%d").date()
        except ValueError:
            return None

    month_first = re.search(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?(?:[ ,]+(20\d{2}))?\b",
        text, re.I,
    )
    day_first = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|"
        r"apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[ ,]+(20\d{2}))?\b",
        text, re.I,
    )
    match = month_first or day_first
    if match:
        if month_first:
            month_text, day, year_text = match.group(1), match.group(2), match.group(3)
        else:
            day, month_text, year_text = match.group(1), match.group(2), match.group(3)
        try:
            candidate = datetime(
                int(year_text or now.year), _MONTHS[month_text.lower()], int(day)
            ).date()
            if not year_text and candidate < now.date():
                candidate = candidate.replace(year=candidate.year + 1)
            return candidate
        except (KeyError, ValueError):
            return None
    return None


def parse_flight_request(text: str, now: datetime | None = None) -> dict | None:
    if not re.search(r"\b(flights?|airfare|fare)\b", text, re.I):
        return None
    if not re.search(r"\b(price|prices|cost|fare|fares|cheap|cheapest|search|find|available)\b", text, re.I):
        return None

    route = re.search(
        r"\bfrom\s+(.+?)\s+to\s+(.+?)(?=\s+(?:on|for|departing|leaving)\b|[?.!,]|$)",
        text, re.I,
    )
    if not route:
        return {"missing": "origin and destination"}

    now = now or datetime.now(LOCAL_TZ)
    now = now.replace(tzinfo=LOCAL_TZ) if now.tzinfo is None else now.astimezone(LOCAL_TZ)
    departure_date = _parse_date(text, now)
    if departure_date is None:
        return {"missing": "departure date"}

    adults_match = re.search(r"\b(\d+)\s+(?:adults?|passengers?|travellers?|travelers?)\b", text, re.I)
    adults = int(adults_match.group(1)) if adults_match else 1
    return {
        "from_city": _clean_city(route.group(1)),
        "to_city": _clean_city(route.group(2)),
        "departure_date": departure_date.isoformat(),
        "adults": max(1, min(adults, 9)),
    }


def execute_flight_search(text: str) -> dict | None:
    details = parse_flight_request(text)
    if details is None:
        return None
    if details.get("missing"):
        return {
            "handled": True,
            "response": f"What {details['missing']} should I use for the flight search?",
            "flights": [],
        }

    if _is_country_name(details["from_city"]):
        return {
            "handled": True,
            "response": (
                f"Which departure city or airport in {details['from_city']} should I use? "
                "A country can contain several airports with very different fares."
            ),
            "flights": [],
        }
    if _is_country_name(details["to_city"]):
        return {
            "handled": True,
            "response": (
                f"Which destination city or airport in {details['to_city']} should I use? "
                "For example, choose the city you actually want to visit."
            ),
            "flights": [],
        }

    origin_code = resolve_iata_code(details["from_city"])
    destination_code = resolve_iata_code(details["to_city"])
    if not origin_code or not destination_code:
        unresolved = details["from_city"] if not origin_code else details["to_city"]
        return {
            "handled": True,
            "response": (
                f"I couldn't identify an airport for {unresolved}. Please provide a "
                "city, airport name, or three-letter IATA code."
            ),
            "flights": [],
        }

    flights = search_flights(
        origin_code, destination_code, details["departure_date"],
        adults=details["adults"], max_results=5,
    )
    if not flights:
        return {
            "handled": True,
            "response": (
                f"I couldn't find live economy fares from {details['from_city']} to "
                f"{details['to_city']} for {details['departure_date']}. The flight data "
                "provider returned no current options; no price was invented."
            ),
            "flights": [],
        }

    lines = [
        f"Live one-way economy fares from {details['from_city']} to "
        f"{details['to_city']} on {details['departure_date']}:"
    ]
    for index, flight in enumerate(flights, 1):
        stops = int(flight.get("stops", 0) or 0)
        stop_text = "nonstop" if stops == 0 else f"{stops} stop" + ("s" if stops != 1 else "")
        lines.append(
            f"{index}. {flight.get('airline', 'Airline')} — "
            f"{flight.get('currency', 'INR')} {flight.get('price', '—')} — "
            f"{flight.get('departure', '—')} to {flight.get('arrival', '—')} — "
            f"{flight.get('duration', '—')} min, {stop_text}"
        )
    lines.append("Fares are live search results and can change before booking.")
    return {"handled": True, "response": "\n".join(lines), "flights": flights}
