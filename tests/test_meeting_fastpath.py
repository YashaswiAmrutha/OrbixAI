from datetime import datetime
from zoneinfo import ZoneInfo

from backend.orchestration.meeting_fastpath import parse_explicit_meeting_request


def test_parses_tomorrow_time_duration_and_email_invitation():
    now = datetime(2026, 8, 15, 20, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    result = parse_explicit_meeting_request(
        "Schedule a meeting tomorrow at 3 PM for 30 minutes with "
        "nryashaswiamrutha@gmail.com and email the invitation.",
        now=now,
    )

    assert result is not None
    assert result["attendee_email"] == "nryashaswiamrutha@gmail.com"
    assert result["start_time"].isoformat() == "2026-08-16T15:00:00+05:30"
    assert result["duration_minutes"] == 30
    assert result["email_invitation"] is True


def test_refuses_to_execute_without_an_explicit_time():
    assert parse_explicit_meeting_request(
        "Schedule a meeting tomorrow with person@example.com"
    ) is None


def test_parses_weekday_written_date_and_natural_duration():
    now = datetime(2026, 8, 15, 20, 0, tzinfo=ZoneInfo("Asia/Kolkata"))

    weekday = parse_explicit_meeting_request(
        "Arrange a call next Monday at noon for half an hour with person@example.com",
        now=now,
    )
    written = parse_explicit_meeting_request(
        "Book a meeting on 20 August at 15:30 with person@example.com",
        now=now,
    )

    assert weekday["start_time"].isoformat() == "2026-08-24T12:00:00+05:30"
    assert weekday["duration_minutes"] == 30
    assert written["start_time"].isoformat() == "2026-08-20T15:30:00+05:30"
