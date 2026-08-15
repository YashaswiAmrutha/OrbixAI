"""
OrbixAI Google-suite MCP server.

Exposes the user's Gmail, Google Meet and local calendar as typed MCP tools so the
agent brain can *act* — send mail, spin up a Meet, add/list/remove calendar events —
through the same tool-calling loop it already uses for memory (architecture §7).
This replaces the old hardcoded intent → fixed-script action code: the model now
decides which tool to call per task and can chain them (create_meet → send_email).

All the real work lives in existing modules — this file only maps them to tools and
advertises read/write hints:
  * Gmail / Meet : google_service/gmail_client.py  (GmailClient)
  * mail content : google_service/mail_generator.py (MailGenerator)
  * calendar     : calendar_store.py                (JSON-backed local calendar)

Auth: a single GmailClient is built lazily against ABSOLUTE credentials.json /
token.json paths (this server runs as its own stdio subprocess with an unknown cwd).
If the user hasn't completed OAuth, action tools return {"error":"needs_auth", ...}
so the agent can ask them to sign in at /auth/login instead of crashing.

Tools
  read  (readOnlyHint):  list_emails · list_calendar_events
  write (action)      :  send_email · create_meet · create_calendar_event
  destructive         :  delete_calendar_event

Run (stdio transport — how the host spawns it):
    python backend/mcp_servers/gsuite_server.py
"""

import re
import sys
from pathlib import Path

# GmailClient / MailGenerator / calendar_store all live under backend/ and import
# each other by package/bare name, so put backend/ on the path and import directly.
_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

# Allow OAuth token refresh over the local (http) redirect during dev, matching main.py.
import os  # noqa: E402
os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

import calendar_store  # type: ignore  # noqa: E402
from google_service.gmail_client import GmailClient  # type: ignore  # noqa: E402
from google_service.mail_generator import MailGenerator  # type: ignore  # noqa: E402

# graph/memory.py lives in a sibling dir and imports its own siblings by bare
# name, same pattern as neo4j_cypher_server.py.
sys.path.insert(0, str(_BACKEND / "graph"))
import memory as mem  # type: ignore  # noqa: E402

from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

# Absolute paths so the subprocess finds them regardless of cwd.
_CREDENTIALS = str(_BACKEND / "credentials.json")
_TOKEN = str(_BACKEND / "token.json")

mcp = FastMCP(
    "orbix-gsuite",
    instructions=(
        "OrbixAI actions over the user's Google account and local calendar. Use these "
        "tools to actually DO things: send_email, create_meet (Google Meet link), "
        "create_calendar_event, list_emails, list_calendar_events, delete_calendar_event. "
        "Chain them when needed — e.g. create_meet first, then pass its meet_link into "
        "send_email. If a tool returns {'error':'needs_auth'}, tell the user to sign in "
        "at /auth/login; do not retry."
    ),
)

_READONLY = ToolAnnotations(readOnlyHint=True)
_ACTION = ToolAnnotations(readOnlyHint=False, idempotentHint=False)
_DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True)

_NEEDS_AUTH = {"error": "needs_auth", "auth_url": "/auth/login",
               "message": "Gmail isn't connected. Ask the user to sign in at /auth/login."}

# Lazily-built singleton Gmail client (shared across tool calls in this subprocess).
_gmail: GmailClient | None = None


def _client() -> GmailClient | None:
    """Return an authenticated GmailClient, or None if OAuth hasn't been completed."""
    global _gmail
    if _gmail is None:
        try:
            _gmail = GmailClient(credentials_file=_CREDENTIALS, token_file=_TOKEN)
        except Exception:
            return None
    if not _gmail.is_authenticated():
        # token may have been written/refreshed by the web OAuth flow since last check
        _gmail._try_load_token()
    return _gmail if _gmail.is_authenticated() else None


# --------------------------------------------------------------------------- #
# READ tools (safe, no side effects)
# --------------------------------------------------------------------------- #
@mcp.tool(annotations=_READONLY)
def list_emails(max_results: int = 10, query: str = "") -> dict:
    """
    List the user's emails (inbox + sent), most recent first. Returns
    {emails: [{from, subject, date, snippet, type}], count}. Returns needs_auth
    if the user isn't signed in.

    Pass `query` (optional) to actually SEARCH instead of just listing whatever's
    most recent — Gmail's native search syntax: from:name@x.com, subject:word,
    has:attachment, after:2024/01/01, etc. Use this whenever the user asks about
    a SPECIFIC email ("the zoomin mail", "the email from Sarah") rather than
    calling this with no query and hoping it's in the most-recent few — without a
    query this only ever returns whatever's newest, which usually is NOT the
    email being asked about.

    There is NO delete_email tool — email deletion is not supported by this
    integration. If asked to delete an email, say so plainly; do not describe a
    plan to call a tool that doesn't exist.
    """
    client = _client()
    if not client:
        return _NEEDS_AUTH
    emails = client.get_all_recent_emails(max_results, query=query)
    return {"emails": emails, "count": len(emails)}


@mcp.tool(annotations=_READONLY)
def list_calendar_events(year: int | None = None, month: int | None = None,
                         upcoming_only: bool = False) -> dict:
    """
    List saved calendar events. Returns {events: [{id, title, date, time,
    description, type}], count}.

    - Pass year AND month together for a specific month.
    - Pass upcoming_only=true (and no year/month) when the user asks about
      "upcoming"/"future" events — this filters to today onward server-side.
      Do NOT try to filter past events out yourself by reasoning about dates;
      this flag does it exactly.
    - With neither set, returns every event ever created (past and future).

    For "how many trips do I have" style questions, use `list_trips` instead —
    a multi-day trip is stored as one calendar event PER DAY, so this tool will
    list each day of the same trip separately.
    """
    if upcoming_only and not (year and month):
        events = calendar_store.get_upcoming_events()
    else:
        events = calendar_store.get_events(year, month)
    return {"events": events, "count": len(events)}


@mcp.tool(annotations=_READONLY)
def list_trips() -> dict:
    """
    List planned trips, one entry per trip (not per day). A multi-day trip is
    stored internally as one calendar event per day ("Day 1: ...", "Day 2:
    ..."); this tool groups those back into trips. Returns {trips:
    [{destination, start_date, end_date, num_days}], count}. Use this — not
    list_calendar_events — for any "how many trips" / "what trips are planned"
    style question.
    """
    trips = calendar_store.get_trips()
    return {"trips": trips, "count": len(trips)}


# --------------------------------------------------------------------------- #
# ACTION tools (execute autonomously — the model decides when to call them)
# --------------------------------------------------------------------------- #
@mcp.tool(annotations=_ACTION)
def send_email(recipient_email: str = "", subject: str = "", body: str = "",
               user_prompt: str = "", meet_link: str = "", recipient: str = "",
               to: str = "") -> dict:
    """
    Send an email. `recipient_email` MUST be a real address containing '@' — if the
    user gave only a name, resolve it with recall/search first, or ask them; do NOT
    call this with a bare name. Provide subject+body directly, OR leave them blank and
    pass `user_prompt` describing what to say and a subject/body is drafted for you.

    To include a Google Meet link: pass the FULL url in `meet_link` (e.g. the meet_link
    you just got back from create_meet) — it is guaranteed to be added to the email. Do
    NOT rely on `user_prompt` prose to carry a URL; auto-drafting cannot invent the link.
    Returns {sent: true, recipient, subject} on success, or {error: ...} on failure —
    only report success to the user if you get sent: true.
    """
    # `recipient`/`to` are accepted as tolerated aliases: this model was observed
    # calling this tool with each of these instead of `recipient_email` (repeatedly,
    # across independent turns) — since recipient_email had no default, that was a
    # hard schema-validation failure (missing required field). Worse than just
    # failing: with `to` unrecognized, the call silently fell through to the
    # "email myself" fallback below and reported success — the model HAD correctly
    # extracted the right recipient from the user's message, it just used a
    # plausible alternate key name for it, and the tool sent to the wrong person
    # while claiming success. Recognizing the alias is what actually fixes that
    # (the self-fallback below is only for genuinely no recipient given at all).
    recipient_email = (recipient_email or recipient or to or "").strip()

    # Fallback for genuinely "email this to myself" requests only — gated on an
    # explicit self-reference in user_prompt. This USED to fire unconditionally
    # on any empty recipient_email (added because the model correctly recalls its
    # own email early in the turn but was observed losing it a few steps later),
    # but that was too blunt: confirmed live, "mail this to my assistant" (a third
    # party with no email on file) also left recipient_email empty, and the
    # unconditional fallback silently sent it to the OWNER instead — a real
    # wrong-recipient bug, not just a missing one. Gating on explicit self-language
    # keeps the original fix for real "email me" cases while making the assistant
    # case fail loudly (below) instead of silently succeeding at the wrong person.
    _SELF_REF = re.compile(r"\b(my ?self|my own email|to me\b|send it to me|"
                           r"email me\b)\b", re.I)
    if not recipient_email and _SELF_REF.search(user_prompt or ""):
        try:
            self_info = mem.recall("user:self")
            self_email = (self_info or {}).get("properties", {}).get("email", "")
            if self_email and "@" in self_email:
                recipient_email = self_email
        except Exception:  # noqa: BLE001 — Neo4j down etc.; fall through to the error below
            pass

    if not recipient_email:
        return {"error": "recipient_email is required and was empty. If this is meant "
                         "for the user themselves, use the email shown as 'You "
                         "(user:self)' in your context directly. If it's meant for "
                         "someone else, resolve their email via recall/search, or ask "
                         "the user for it — do not guess or leave this blank."}

    if "@" not in recipient_email or "." not in recipient_email.split("@")[-1]:
        return {"error": f"'{recipient_email}' is not a valid email address. Resolve the "
                         "contact's email from memory (recall/search) or ask the user for it."}
    client = _client()
    if not client:
        return _NEEDS_AUTH
    if not subject or not body:
        content = MailGenerator.generate_mail_content(
            user_prompt or subject or f"Message to {recipient_email}",
            recipient_name=recipient_email.split("@")[0],
        )
        subject = subject or content.get("subject", "")
        body = body or content.get("body", "")
    # Guarantee a supplied Meet link lands in the email, even if the body was drafted
    # without it. Coerce defensively — a small model may pass a non-string here.
    meet_link = meet_link.strip() if isinstance(meet_link, str) else ""
    if meet_link and meet_link not in body:
        body = (body.rstrip() + f"\n\nGoogle Meet link: {meet_link}").strip()
    if not subject or not body:
        return {"error": "subject and body (or a user_prompt to draft them) are required"}
    result = client.send_email(recipient_email, subject, body)
    if not result.get("success"):
        return {"error": result.get("error", "send failed")}
    return {"sent": True, "recipient": recipient_email, "subject": subject,
            "message_id": result.get("message_id")}


@mcp.tool(annotations=_ACTION)
def create_meet(attendee_email: str, event_title: str = "Meeting",
                event_description: str = "", email_link_to: str = "") -> dict:
    """
    Create a Google Meet (with calendar invite) for attendee_email and return
    {meet_link, event_id, event_title, attendee_email}.

    If the user asked you to "email/send/mail the link" to someone, set `email_link_to`
    to that recipient's real email address (e.g. the same address, or resolved from
    memory). This tool will then create the meet AND send them a plain email containing
    the link in ONE step — the reliable way to satisfy "set up a meet and email the
    link". The result includes {emailed: true, email_recipient} when that succeeds.
    (You may instead chain a separate send_email call yourself, passing meet_link — but
    email_link_to is simpler and guarantees the link is in the email.)
    """
    client = _client()
    if not client:
        return _NEEDS_AUTH
    result = client.create_google_meet(event_title, event_description, attendee_email)
    if not result.get("success"):
        return {"error": result.get("error", "meet creation failed")}
    meet_link = result.get("meet_link")
    out = {"meet_link": meet_link, "event_id": result.get("event_id"),
           "event_title": event_title, "attendee_email": attendee_email}

    # One-step "create meet AND email the link": the tool owns the body, so the actual
    # url is guaranteed to be in the email (a small model can't drop it here).
    email_link_to = email_link_to.strip() if isinstance(email_link_to, str) else ""
    if email_link_to and "@" in email_link_to:
        body = (f"Hi,\n\nHere is the Google Meet link for {event_title}:\n"
                f"{meet_link}\n\nSee you there.")
        send = client.send_email(email_link_to, f"{event_title} — Google Meet link", body)
        out["emailed"] = bool(send.get("success"))
        out["email_recipient"] = email_link_to
        if not send.get("success"):
            out["email_error"] = send.get("error", "send failed")
    return out


@mcp.tool(annotations=_ACTION)
def create_calendar_event(title: str, date: str, time: str = "",
                          description: str = "", type: str = "event") -> dict:
    """
    Add an event to the user's calendar. `date` is YYYY-MM-DD, `time` optional HH:MM,
    `type` one of general/event/meeting/travel/task. Returns the stored event
    {id, title, date, time, description, type, ...}.
    """
    ev = calendar_store.create_event(
        title=title, date=date, time=time, description=description,
        type=type, source="ai_agent",
    )
    return ev


@mcp.tool(annotations=_DESTRUCTIVE)
def delete_calendar_event(event_id: str) -> dict:
    """Delete a calendar event by its id. Returns {deleted: bool}."""
    return {"deleted": calendar_store.delete_event(event_id)}


if __name__ == "__main__":
    mcp.run()  # stdio transport
