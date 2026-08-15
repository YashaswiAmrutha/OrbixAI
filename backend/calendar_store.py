import json
import uuid
from pathlib import Path
from datetime import datetime

STORE_FILE = Path(__file__).parent / "events.json"

_COLOR_MAP = {
    "meeting": "#3b82f6",
    "travel":  "#22c55e",
    "task":    "#f59e0b",
    "general": "#8b5cf6",
}


def _load() -> dict:
    if STORE_FILE.exists():
        return json.loads(STORE_FILE.read_text(encoding="utf-8"))
    return {"events": []}


def _save(data: dict) -> None:
    STORE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_events(year: int = None, month: int = None) -> list:
    events = _load()["events"]
    if year and month:
        prefix = f"{int(year)}-{int(month):02d}"
        events = [e for e in events if e.get("date", "").startswith(prefix)]
    return events


def get_upcoming_events(from_date: str = None) -> list:
    """Events on or after `from_date` (default: today), sorted by date. There
    was previously no date-based filtering anywhere in this store — a request
    for "upcoming" events actually returned every event ever created, past and
    future, and relied on the model to mentally filter by date, which it did
    not do reliably."""
    from_date = from_date or datetime.utcnow().strftime("%Y-%m-%d")
    events = [e for e in _load()["events"] if e.get("date", "") >= from_date]
    return sorted(events, key=lambda e: (e.get("date", ""), e.get("time", "")))


def get_trips() -> list:
    """Group travel-type calendar events into trips instead of listing each day
    separately. _travel_calendar_events() (orchestration/workflow.py) creates
    one calendar event PER DAY of an itinerary ("Day 1: ...", "Day 2: ...") —
    correct for the calendar view, but with no shared trip identifier. Asked
    "how many trips do I have", the model had no way to know 3 day-events were
    one trip and listed each as its own trip. All days of one trip share the
    same `description` ("Trip to {destination}") — group on that instead.
    Returns [{destination, start_date, end_date, num_days}, ...].
    """
    travel_events = [e for e in _load()["events"] if e.get("type") == "travel"]
    groups: dict[str, list] = {}
    for e in travel_events:
        groups.setdefault(e.get("description", "Trip"), []).append(e)
    trips = []
    for description, evs in groups.items():
        dates = sorted(e.get("date", "") for e in evs if e.get("date"))
        if not dates:
            continue
        destination = description.replace("Trip to ", "").strip() or description
        trips.append({
            "destination": destination,
            "start_date": dates[0],
            "end_date": dates[-1],
            "num_days": len(evs),
        })
    return sorted(trips, key=lambda t: t["start_date"])


def create_event(title: str, date: str, time: str = "", description: str = "",
                 type: str = "general", source: str = "manual", color: str = None,
                 **_) -> dict:
    data = _load()
    event = {
        "id":          str(uuid.uuid4()),
        "title":       title,
        "date":        date,
        "time":        time,
        "description": description,
        "type":        type,
        "source":      source,
        "color":       color or _COLOR_MAP.get(type, "#8b5cf6"),
        "created_at":  datetime.utcnow().isoformat(),
    }
    data["events"].append(event)
    _save(data)
    return event


def bulk_create_events(events_list: list) -> list:
    return [create_event(**ev) for ev in events_list]


def update_event(event_id: str, **fields) -> dict | None:
    data = _load()
    fields.pop("id", None)
    fields.pop("created_at", None)
    for e in data["events"]:
        if e["id"] == event_id:
            e.update(fields)
            _save(data)
            return e
    return None


def delete_event(event_id: str) -> bool:
    data = _load()
    before = len(data["events"])
    data["events"] = [e for e in data["events"] if e["id"] != event_id]
    if len(data["events"]) < before:
        _save(data)
        return True
    return False
