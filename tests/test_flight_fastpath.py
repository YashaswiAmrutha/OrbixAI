from datetime import datetime
from zoneinfo import ZoneInfo

from backend.orchestration.flight_fastpath import _is_country_name, parse_flight_request


def test_parses_live_fare_request_and_maps_kashmir_to_srinagar():
    now = datetime(2026, 8, 15, 20, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    result = parse_flight_request(
        "what are the economy seat flight prices from bengaluru to kashmir on Aug 18th",
        now=now,
    )

    assert result == {
        "from_city": "Bengaluru",
        "to_city": "Srinagar",
        "departure_date": "2026-08-18",
        "adults": 1,
    }


def test_asks_for_missing_departure_date_instead_of_guessing():
    result = parse_flight_request("find cheap flights from Bengaluru to Kashmir")
    assert result == {"missing": "departure date"}


def test_country_destinations_require_a_city_but_city_states_do_not():
    assert _is_country_name("Morocco") is True
    assert _is_country_name("Singapore") is False
