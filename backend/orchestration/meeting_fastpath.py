"""Deterministic execution for explicit scheduled-meeting requests.

Small local models can hallucinate tool-result placeholders when many MCP tools
are present.  Scheduling is also too important to let a model invent the date.
This module only activates when an email address, a supported date expression,
and an explicit clock time are all present in a meeting request.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from google_service.gmail_client import GmailClient


LOCAL_TZ = ZoneInfo("Asia/Kolkata")
_EMAIL_RE = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.I)
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _parse_meeting_date(text: str, now: datetime):
    """Parse safe, unambiguous date forms commonly used in meeting requests."""
    lower = text.lower()
    if "day after tomorrow" in lower:
        return (now + timedelta(days=2)).date()
    # Accept the common shorthand/typo "tom" as well as "tomorrow". This is
    # word-bounded so names such as "Tommy" are not interpreted as dates.
    if re.search(r"\btom(?:orrow)?\b", lower):
        return (now + timedelta(days=1)).date()
    if "today" in lower:
        return now.date()

    iso = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text)
    if iso:
        try:
            return datetime.strptime(iso.group(0), "%Y-%m-%d").date()
        except ValueError:
            return None

    # India-first numeric form; require a four-digit year to avoid silently
    # interpreting an ambiguous value such as 8/9.
    numeric = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b", text)
    if numeric:
        try:
            return datetime.strptime(numeric.group(0).replace("-", "/"), "%d/%m/%Y").date()
        except ValueError:
            return None

    month_names = (
        "january|february|march|april|may|june|july|august|september|"
        "october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
    )
    written = re.search(
        rf"\b(?:(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_names})|"
        rf"({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?)(?:[ ,]+(20\d{{2}}))?\b",
        text,
        re.I,
    )
    if written:
        day = int(written.group(1) or written.group(4))
        month_text = written.group(2) or written.group(3)
        year = int(written.group(5) or now.year)
        try:
            month = datetime.strptime(month_text[:3].title(), "%b").month
            candidate = datetime(year, month, day).date()
            if not written.group(5) and candidate < now.date():
                candidate = datetime(year + 1, month, day).date()
            return candidate
        except ValueError:
            return None

    weekday = re.search(
        r"\b(?:(next|this)\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        lower,
    )
    if weekday:
        target = _WEEKDAYS[weekday.group(2)]
        days = (target - now.weekday()) % 7
        if days == 0 or weekday.group(1) == "next":
            days += 7
        return (now + timedelta(days=days)).date()
    return None


def parse_explicit_meeting_request(text: str, now: datetime | None = None) -> dict | None:
    """Return normalized meeting details, or None when clarification is safer."""
    if not re.search(
        r"\b(schedule|create|book|set\s+up|arrange|organize)\b.*\b(meet|meeting|call)\b",
        text,
        re.I,
    ):
        return None

    email_match = _EMAIL_RE.search(text)
    if not email_match:
        return None

    now = now or datetime.now(LOCAL_TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=LOCAL_TZ)
    else:
        now = now.astimezone(LOCAL_TZ)

    meeting_date = _parse_meeting_date(text, now)
    if meeting_date is None:
        return None

    # Mobile typing commonly joins the conjunction to the meridiem
    # ("3pmand mail it"). Normalize that harmless typo before parsing.
    normalized_text = re.sub(
        r"(?<=\d)(a\.?m\.?|p\.?m\.?)(?=and\b)",
        r"\1 ",
        text,
        flags=re.I,
    )
    time_match = re.search(
        r"\bat\s+(?:(noon|midnight)|(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?)\b",
        normalized_text,
        re.I,
    )
    if not time_match:
        return None
    named_time = (time_match.group(1) or "").lower()
    hour = 12 if named_time == "noon" else (0 if named_time == "midnight" else int(time_match.group(2)))
    minute = int(time_match.group(3) or 0)
    meridiem = (time_match.group(4) or "").lower().replace(".", "")
    if minute > 59 or hour > (12 if meridiem else 23) or hour == 0 and meridiem:
        return None
    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0

    duration_minutes = 60
    duration_match = re.search(
        r"\bfor\s+(\d+)\s*(minutes?|mins?|hours?|hrs?)\b", text, re.I
    )
    if duration_match:
        amount = int(duration_match.group(1))
        unit = duration_match.group(2).lower()
        duration_minutes = amount * 60 if unit.startswith(("hour", "hr")) else amount
    elif re.search(r"\bfor\s+(?:half\s+an?|a\s+half)\s+hours?\b", text, re.I):
        duration_minutes = 30
    elif re.search(r"\bfor\s+(?:an?|one)\s+hours?\b", text, re.I):
        duration_minutes = 60
    if duration_minutes < 5 or duration_minutes > 24 * 60:
        return None

    topic_match = re.search(r"\b(?:about|regarding|discussing)\s+([^,.]{3,80})", text, re.I)
    attendee = email_match.group(0)
    title = topic_match.group(1).strip() if topic_match else f"Meeting with {attendee}"
    email_invitation = bool(
        re.search(r"\b(email|mail|send)\b.*\b(invite|invitation|link)\b", text, re.I)
        or re.search(r"\b(invite|invitation|link)\b.*\b(email|mail|send)\b", text, re.I)
        # Natural shorthand such as "mail that to alice@example.com" clearly
        # refers to the meeting/invitation even when the word "link" is omitted.
        or re.search(r"\b(?:email|mail|send)\s+(?:that|it|this)?\s*(?:to\s+)?"
                     + _EMAIL_RE.pattern, text, re.I)
    )

    start = datetime(
        meeting_date.year, meeting_date.month, meeting_date.day,
        hour, minute, tzinfo=LOCAL_TZ,
    )
    return {
        "attendee_email": attendee,
        "event_title": title,
        "start_time": start,
        "duration_minutes": duration_minutes,
        "email_invitation": email_invitation,
    }


def execute_explicit_meeting(text: str, fallback_texts: list[str] | None = None) -> dict | None:
    """Execute a fully specified request and return only verified outcomes."""
    details = parse_explicit_meeting_request(text)
    # Follow-up requests often omit slots already supplied in the previous turn,
    # e.g. "schedule a meeting and send me the link here" immediately after a
    # fully specified request. Reuse the newest complete user request rather than
    # asking the model to invent a date or address.
    if details is None:
        for previous in reversed(fallback_texts or []):
            details = parse_explicit_meeting_request(previous)
            if details is not None:
                break
    if details is None:
        return None

    # Delivery instructions belong to the CURRENT request, not the request whose
    # date/attendee slots were reused. "here/in chat" means show the verified URL
    # in the response and do not send another email.
    if re.search(r"\b(link|invite|invitation)\b.*\b(here|chat)\b", text, re.I):
        details["email_invitation"] = False
    elif re.search(r"\b(email|mail)\b", text, re.I):
        details["email_invitation"] = True
    else:
        details["email_invitation"] = False

    client = GmailClient()
    if not client.is_authenticated():
        return {"handled": True, "error": "Google is not connected. Sign in at /auth/login."}

    result = client.create_google_meet(
        details["event_title"],
        "Scheduled by OrbixAI",
        details["attendee_email"],
        start_time=details["start_time"],
        duration_minutes=details["duration_minutes"],
    )
    if not result.get("success"):
        return {"handled": True, "error": result.get("error", "Meeting creation failed")}

    when = details["start_time"].strftime("%A, %d %B %Y at %I:%M %p %Z")
    response = (
        f"Created the meeting with {details['attendee_email']} for {when} "
        f"({details['duration_minutes']} minutes).\n\n"
        f"Google Meet: {result['meet_link']}"
    )
    if details["email_invitation"]:
        # create_google_meet uses sendUpdates='all', so Calendar has already
        # emailed its official invitation. Do not make a duplicate Gmail call.
        response += f"\n\nGoogle Calendar emailed the invitation to {details['attendee_email']}."

    return {
        "handled": True,
        "response": response,
        "meeting": {
            "title": details["event_title"],
            "date": details["start_time"].strftime("%Y-%m-%d"),
            "time": details["start_time"].strftime("%H:%M"),
            "description": (
                f"Attendee: {details['attendee_email']}\nMeet link: {result['meet_link']}"
            ),
            "type": "meeting",
            "source": "ai_meeting",
            "external_event_id": result.get("event_id"),
        },
    }
