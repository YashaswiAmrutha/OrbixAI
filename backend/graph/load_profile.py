"""
Load the OrbixAI profile (Docs/profile.md) into the Neo4j memory graph.

Parses the ```yaml block in profile.md and writes it through the shared memory
engine (backend/graph/memory.py), so every node is MERGEd on its type-prefixed
`key` and the write protocol lives in one place. Re-running updates the same
nodes instead of duplicating them.

Mutable attributes are stored as :Fact nodes, not properties: the user's `city`
becomes a `home_city` fact (so "I moved to X" later overwrites it in place — the
single source of truth for that value). Stable identity (name, email, birthday,
languages, ...) stays on the :User node.

Usage:
    python backend/graph/load_profile.py            # load Docs/profile.md
    python backend/graph/load_profile.py --dry-run  # parse + print, no writes
    python backend/graph/load_profile.py <path.md>  # load a different profile

Run init_schema.py first so the :Entity.key uniqueness constraint exists.
"""

import re
import sys
import logging
from pathlib import Path

import yaml

import memory  # type: ignore  (run from this dir)
from connection import verify  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("load_profile")

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = _ROOT / "Docs" / "profile.md"

# relation (from profile) -> edge from :User
FAMILY_EDGE = {
    "mother": "CHILD_OF", "father": "CHILD_OF", "parent": "CHILD_OF",
    "son": "PARENT_OF", "daughter": "PARENT_OF", "child": "PARENT_OF",
    "brother": "SIBLING_OF", "sister": "SIBLING_OF", "sibling": "SIBLING_OF",
    "spouse": "MARRIED_TO", "husband": "MARRIED_TO", "wife": "MARRIED_TO",
}


# --------------------------------------------------------------------------- #
# parsing helpers
# --------------------------------------------------------------------------- #
def extract_yaml(md_text: str) -> dict:
    m = re.search(r"```ya?ml\s*\n(.*?)```", md_text, re.DOTALL)
    if not m:
        raise ValueError("No ```yaml block found in the profile markdown.")
    return yaml.safe_load(m.group(1)) or {}


def norm_date(val) -> str | None:
    """Normalize to ISO YYYY-MM-DD. Accepts YYYY-MM-DD or DD-MM-YYYY."""
    if not val:
        return None
    s = str(val).strip()
    parts = s.split("-")
    if len(parts) != 3:
        return s
    if len(parts[0]) == 4:
        y, mo, d = parts
    else:
        d, mo, y = parts
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"


def clean(v):
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip() or None
    if isinstance(v, (list, tuple)):
        items = [clean(x) for x in v if clean(x) is not None]
        return items or None
    return v


def _phone(v):
    return clean(str(v)) if v else None


def add_birthday(owner_key: str, name: str, date_iso: str) -> None:
    """Create an :Event:Birthday and link the owner via HAS_EVENT."""
    ekey = f"event:{owner_key}:birthday"
    memory.upsert_entity(
        "event", f"{name}'s birthday", key=ekey, sublabels=["Birthday"],
        props={"title": f"{name}'s birthday", "date": date_iso, "recurrence": "yearly"},
    )
    memory.link(owner_key, "HAS_EVENT", ekey)


# --------------------------------------------------------------------------- #
# write the whole profile via the memory engine
# --------------------------------------------------------------------------- #
def load(data: dict) -> list[str]:
    summary: list[str] = []

    # ---- USER (stable identity on the node; city -> home_city :Fact) ------- #
    you = data.get("you") or {}
    bday = norm_date(you.get("birthday"))
    memory.upsert_entity("user", clean(you.get("name")), props={
        "email": clean(you.get("email")),
        "other_emails": clean(you.get("other_emails")),
        "phone": _phone(you.get("phone")),
        "home_address": clean(you.get("home_address")),
        "work_address": clean(you.get("work_address")),
        "birthday": bday,
        "timezone": clean(you.get("timezone")),
        "blood_group": clean(you.get("blood_group")),
        "languages": clean(you.get("languages")),
    })
    summary.append(f"User: {you.get('name')}")
    if bday:
        add_birthday("user:self", clean(you.get("name")) or "You", bday)
    city = clean(you.get("city"))
    if city:
        res = memory.remember_fact("user:self", "home_city", city, source="manual")
        summary.append(f"Fact: home_city = {city}"
                       + (f"  (was {res['previous']})" if res["changed"] and res["previous"] else ""))

    # ---- FAMILY ----------------------------------------------------------- #
    for fam in data.get("family") or []:
        name = clean(fam.get("name"))
        if not name:
            continue
        rel = (clean(fam.get("relation")) or "").lower()
        edge = FAMILY_EDGE.get(rel, "KNOWS")
        fbday = norm_date(fam.get("birthday"))
        key = memory.upsert_entity("person", name, sublabels=["Family"], props={
            "relationship": rel or None,
            "city": clean(fam.get("city")),
            "phone": _phone(fam.get("phone")),
            "birthday": fbday,
        })
        memory.link("user:self", edge, key)
        if fbday:
            add_birthday(key, name, fbday)
        summary.append(f"Family: {name} ({rel}) -[:{edge}]-")

    # ---- CONTACTS --------------------------------------------------------- #
    for c in data.get("contacts") or []:
        name = clean(c.get("name"))
        if not name:
            continue
        how = (clean(c.get("how_you_know")) or "").lower()
        sub = {"colleague": "Colleague", "friend": "Friend",
               "client": "Client", "neighbor": "Neighbor"}.get(how, "Contact")
        key = memory.upsert_entity("person", name, sublabels=[sub], props={
            "email": clean(c.get("email")),
            "phone": _phone(c.get("phone")),
            "role": clean(c.get("role")),
            "relationship": how or None,
        })
        memory.link("user:self", "KNOWS", key)
        company = clean(c.get("company"))
        if company:
            okey = memory.upsert_entity("organization", company)
            role = clean(c.get("role"))
            memory.link(key, "WORKS_AT", okey, {"title": role} if role else None)
        summary.append(f"Contact: {name} ({how}) -[:KNOWS]-")

    # ---- ORGANIZATIONS ---------------------------------------------------- #
    for o in data.get("organizations") or []:
        name = clean(o.get("name"))
        if not name:
            continue
        low = name.lower()
        sub = "School" if any(w in low for w in ("univ", "school", "college")) else "Company"
        okey = memory.upsert_entity("organization", name, sublabels=[sub])
        role = clean(o.get("your_role"))
        memory.link("user:self", "WORKS_AT", okey, {"title": role} if role else None)
        summary.append(f"Org: {name} (you: {role}) -[:WORKS_AT]-")

    # ---- PROJECTS --------------------------------------------------------- #
    for p in data.get("projects") or []:
        name = clean(p.get("name"))
        if not name:
            continue
        key = memory.upsert_entity("project", name, props={
            "status": clean(p.get("status")),
            "deadline": norm_date(p.get("deadline")),
            "repo_url": clean(p.get("repo_url")),
            "description": clean(p.get("description")),
        })
        memory.link("user:self", "WORKS_ON", key)
        summary.append(f"Project: {name} ({p.get('status')}) -[:WORKS_ON]-")

    # ---- PREFERENCES ------------------------------------------------------ #
    for pk, pv in (data.get("preferences") or {}).items():
        pv = clean(pv)
        if not pv:
            continue
        cat, _, leaf = str(pk).partition("_")
        leaf = leaf or cat
        key = memory.upsert_entity(
            "preference", str(pk),
            props={"category": cat, "pref_key": leaf, "value": pv},
        )
        memory.link("user:self", "PREFERS", key)
        summary.append(f"Preference: {pk} = {pv}")

    # ---- INTERESTS -------------------------------------------------------- #
    for it in data.get("interests") or []:
        it = clean(it)
        if not it:
            continue
        key = memory.upsert_entity("interest", it)
        memory.link("user:self", "HAS_INTEREST", key)
        summary.append(f"Interest: {it}")

    # ---- EVENTS ----------------------------------------------------------- #
    for ev in data.get("events") or []:
        title = clean(ev.get("title"))
        if not title:
            continue
        key = memory.upsert_entity("event", title, props={
            "title": title, "date": norm_date(ev.get("date")),
            "recurrence": clean(ev.get("recurrence")),
        })
        memory.link("user:self", "HAS_EVENT", key)
        summary.append(f"Event: {title}")

    # ---- MEDICATIONS ------------------------------------------------------ #
    for med in data.get("medications") or []:
        name = clean(med.get("name"))
        if not name:
            continue
        key = memory.upsert_entity("medication", name, props={
            "dosage": clean(med.get("dosage")), "schedule": clean(med.get("schedule")),
        })
        memory.link("user:self", "TAKES", key)
        summary.append(f"Medication: {name}")

    return summary


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry = "--dry-run" in sys.argv
    path = Path(args[0]) if args else DEFAULT_PROFILE

    if not path.exists():
        log.error("Profile not found: %s", path)
        return 1

    data = extract_yaml(path.read_text(encoding="utf-8"))

    if dry:
        log.info("Parsed profile (dry run, no writes):\n%s",
                 yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        return 0

    if not verify():
        log.error("Cannot reach Neo4j — start the DB and re-run.")
        return 1

    summary = load(data)
    log.info("Loaded %s into the memory graph:", path.name)
    for line in summary:
        log.info("  + %s", line)
    log.info("\nDone: %d items written (re-runnable — MERGE on key, no dupes).",
             len(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
