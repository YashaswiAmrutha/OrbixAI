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
