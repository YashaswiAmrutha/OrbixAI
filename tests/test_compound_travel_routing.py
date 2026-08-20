from backend.google_service.travel_planner import _fast_travel_entities
from backend.orchestration.routing import _regex_route


def test_flight_plus_itinerary_routes_to_full_travel_pipeline():
    intent, confidence = _regex_route(
        "Find flights from Delhi to Chicago on Aug 29 and generate a "
        "3-day itinerary with lunch spots"
    )
    assert intent == "travel_planner"
    assert confidence >= 0.9


def test_pure_flight_request_still_uses_fast_fare_path():
    intent, _ = _regex_route(
        "What are flight prices from Bengaluru to Kashmir on Aug 23?"
    )
    assert intent == "flight_search"


def test_generic_compound_travel_fields_are_extracted_without_model_call():
    result = _fast_travel_entities(
        "Find economy flights from Mumbai to Tokyo on 2026-09-03 and "
        "prepare a 5-day itinerary with food and museums"
    )
    assert result["from_city"] == "Mumbai"
    assert result["to_city"] == "Tokyo"
    assert result["check_in"] == "2026-09-03"
    assert result["num_nights"] == 5


def test_trip_wording_with_hotel_request_is_not_prompt_specific():
    result = _fast_travel_entities(
        "Plan a trip to Kyoto for 6 days with hotels near the city centre"
    )
    assert result["to_city"] == "Kyoto"
    assert result["num_nights"] == 6
