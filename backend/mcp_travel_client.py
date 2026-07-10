# mcp_travel_client.py
from mcp.client.session import ClientSession
import json


def _decode_result(result, default):
    if getattr(result, "isError", False):
        message = result.content[0].text if result.content else "MCP tool failed"
        raise RuntimeError(message)
    if not result.content:
        return default
    return json.loads(result.content[0].text)

async def search_flights_mcp(
    session,
    from_city,
    to_city,
    departure_date,
    return_date=None
):
    print(">>> search_flights_tool called <<<")
    print(from_city, to_city, departure_date)

    result = await session.call_tool(
        "search_flights_tool",
        arguments={
            "from_city": from_city,
            "to_city": to_city,
            "departure_date": departure_date,
            "return_date": return_date,
        }
    )

    print("MCP RESULT:", result)

    return _decode_result(result, [])


async def search_hotels_mcp(
    session,
    city,
    check_in,
    check_out,
    num_adults=1
):
    result = await session.call_tool(
        "search_hotels_tool",
        arguments={
            "city": city,
            "check_in": check_in,
            "check_out": check_out,
            "num_adults": num_adults
        }
    )
    return _decode_result(result, [])


async def get_attractions_mcp(
    session,
    city
):
    result =await session.call_tool(
        "attractions_tool",
        arguments={
            "city": city
        }
    )
    return _decode_result(result, [])


async def generate_itinerary_mcp(
    session,
    city,
    attractions,
    num_days,
    check_in,
    num_adults=1,
    flight_summary=None,
    hotel_summary=None
):
    result = await session.call_tool(
        "itinerary_tool",
        arguments={
            "city": city,
            "attractions": attractions,
            "num_days": num_days,
            "check_in": check_in,
            "num_adults": num_adults,
            "flight_summary": flight_summary,
            "hotel_summary": hotel_summary
        }
    )

    return _decode_result(result, "")
    
