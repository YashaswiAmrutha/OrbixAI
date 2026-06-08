"""
Background memory extractor — the single write path SQLite -> Neo4j.

The agent brain is READ-ONLY. Every durable memory write happens here, off the
user-facing path: drain the working-memory queue, cheaply gate out chit-chat, run a
small structured-output model on the survivors, and persist via memory.py's
resolve -> reconcile -> upsert (so writes can never duplicate or contradict).

Because the queue lives in SQLite (durable) and rows are claimed atomically, this is
safe to run from several places against one queue:
  - drain-on-startup        (catches turns left pending when the app was closed)
  - after each agent reply   (keeps Neo4j seconds-fresh)
  - flush-barrier            (drain a session before its next query, §diagram)

If Neo4j or Ollama is down, rows simply stay pending and drain later — nothing is
ever lost (architecture §5.1, §11.2: "draining late is harmless").

CLI:
    python backend/mcp_host/extractor.py          # one drain pass over all pending
    python backend/mcp_host/extractor.py --prune  # drain, then prune old rows + vacuum
"""

import os
import re
import sys
import json
import logging
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph"))
import working_memory as wm   # type: ignore  # noqa: E402
import memory as mem          # type: ignore  # noqa: E402

logger = logging.getLogger(__name__)

# Reuse the agent model by default (already pulled); swap for a smaller one later.
EXTRACT_MODEL = os.environ.get("ORBIX_EXTRACT_MODEL", "llama3.1:8b")
_OPTIONS = {"temperature": 0.0, "num_gpu": int(os.environ.get("OLLAMA_NUM_GPU", "0"))}

# JSON schema constraining the extractor's output (Ollama structured output).
_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string",
                             "enum": ["person", "organization", "project",
                                      "location", "topic"]},
                },
                "required": ["name", "type"],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "relationship": {"type": "string"},
                    "object": {"type": "string"},
                },
                "required": ["subject", "relationship", "object"],
            },
        },
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["subject", "predicate", "value"],
            },
        },
        "episodes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["entities", "relationships", "facts", "episodes"],
}

_EXTRACT_SYSTEM = (
    "Extract durable long-term memory from one chat message as JSON. The user "
    "refers to themselves as 'self'.\n"
    "- entities: people/organizations/projects/locations/topics mentioned, each "
    "{name, type}. EVERY name you use in relationships or facts must appear here.\n"
    "- relationships: links as {subject, relationship, object}. Use NATURAL words "
    "for relationship (mother, father, sister, brother, spouse, friend, colleague, "
    "manager, 'works at', 'works on', 'member of', knows). Always write from the "
    "USER's view: a parent is the user's 'mother'/'father'; a child is 'son'/"
    "'daughter'.\n"
    "- facts: single-valued CHANGEABLE attributes as {subject, predicate, value}, "
    "subject = 'self' or an entity name. snake_case predicates like home_city, "
    "employer, job_title, marital_status. A person's city of residence is a FACT "
    "(home_city), NOT a relationship.\n"
    "- episodes: short standalone sentences for things that happened/were decided.\n"
    "Extract ONLY what is explicitly stated; never invent. Questions, commands, and "
    "chit-chat yield all-empty arrays.\n"
    "Example — 'My sister Riya lives in Goa and works at Infosys':\n"
    '{"entities":[{"name":"Riya","type":"person"},'
    '{"name":"Infosys","type":"organization"}],'
    '"relationships":[{"subject":"self","relationship":"sister","object":"Riya"},'
    '{"subject":"Riya","relationship":"works at","object":"Infosys"}],'
    '"facts":[{"subject":"Riya","predicate":"home_city","value":"Goa"}],'
    '"episodes":[]}'
)

# natural relationship word -> (graph edge from subject, sub-label to add to object)
_REL_MAP = {
    "mother": ("CHILD_OF", "Family"), "father": ("CHILD_OF", "Family"),
    "parent": ("CHILD_OF", "Family"), "mom": ("CHILD_OF", "Family"),
    "dad": ("CHILD_OF", "Family"),
    "son": ("PARENT_OF", "Family"), "daughter": ("PARENT_OF", "Family"),
    "child": ("PARENT_OF", "Family"),
    "brother": ("SIBLING_OF", "Family"), "sister": ("SIBLING_OF", "Family"),
    "sibling": ("SIBLING_OF", "Family"),
    "husband": ("MARRIED_TO", "Family"), "wife": ("MARRIED_TO", "Family"),
    "spouse": ("MARRIED_TO", "Family"),
    "friend": ("FRIENDS_WITH", "Friend"),
    "colleague": ("KNOWS", "Colleague"), "coworker": ("KNOWS", "Colleague"),
    "manager": ("REPORTS_TO", None), "boss": ("REPORTS_TO", None),
    "works at": ("WORKS_AT", None), "work at": ("WORKS_AT", None),
    "works for": ("WORKS_AT", None), "employer": ("WORKS_AT", None),
    "works on": ("WORKS_ON", None), "working on": ("WORKS_ON", None),
    "member of": ("MEMBER_OF", None), "knows": ("KNOWS", None),
    "contact": ("KNOWS", "Contact"),
}

# ---- the gate: cheap, no LLM ------------------------------------------------ #
_CHITCHAT = {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "k", "lol",
             "cool", "nice", "great", "bye", "yes", "no", "yeah", "yep", "nope", "sure"}
_FIRST_PERSON = re.compile(
    r"\b(i|i'm|i am|i've|my|mine|me|we|our)\b", re.IGNORECASE)
_DECLARATIVE_HINT = re.compile(
    r"\b(moved|live|living|work|working|job|employer|married|single|named|name is|"
    r"prefer|like|love|hate|bought|got|started|joined|birthday|anniversary|"
    r"my new|now at|remember)\b", re.IGNORECASE)


_PROPER_NOUN = re.compile(r"\b[A-Z][a-z]{2,}\b")           # mid-sentence capital
_NUM_OR_EMAIL = re.compile(r"\d|@")


def is_memory_worthy(text: str) -> bool:
    """
    Cheap pre-filter to avoid LLM calls on obvious non-memorable turns. It is
    high-recall on purpose — the extraction model is the real precision filter
    (it returns empty for nothing-durable). We only drop: questions (they ask, not
    assert), greetings/acknowledgements, and turns with no memorable signal at all.
    """
    t = (text or "").strip()
    if len(t) < 6:
        return False
    if t.lower().strip("!. ") in _CHITCHAT:
        return False
    is_question = t.endswith("?") or bool(re.match(
        r"^(what|where|when|who|which|how|why|is|are|do|does|did|can|could|should|"
        r"would|will|tell me|show me|list|remind)\b", t, re.IGNORECASE))
    if is_question:
        return False
    # a statement worth a look if it has any memorable signal
    return bool(_FIRST_PERSON.search(t) or _DECLARATIVE_HINT.search(t)
                or _PROPER_NOUN.search(t[1:]) or _NUM_OR_EMAIL.search(t))


# ---- the extractor: small structured-output model -------------------------- #
def extract(text: str) -> dict:
    """Run the constrained model on one turn. Returns the four extraction arrays."""
    try:
        resp = ollama.chat(
            model=EXTRACT_MODEL,
            messages=[{"role": "system", "content": _EXTRACT_SYSTEM},
                      {"role": "user", "content": text}],
            format=_EXTRACT_SCHEMA,
            options=_OPTIONS,
        )
        data = json.loads(resp["message"]["content"])
        return {"entities": data.get("entities") or [],
                "relationships": data.get("relationships") or [],
                "facts": data.get("facts") or [],
                "episodes": data.get("episodes") or []}
    except Exception as e:  # noqa: BLE001
        logger.warning("extract() failed: %s", e)
        raise


_SELF_NAMES = {"self", "user", "i", "me", "myself", "my", "owner"}


def _snake(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").strip().lower()).strip("_")


def _resolve_key(name: str, entity_map: dict[str, str]) -> str | None:
    """Map a name to an entity key. Owns identity so the model never invents keys."""
    n = (name or "").strip()
    if not n:
        return None
    if n.lower() in _SELF_NAMES:
        return "user:self"
    if n in entity_map:
        return entity_map[n]
    for k, v in entity_map.items():          # case-insensitive
        if k.lower() == n.lower():
            return v
    hits = mem.search(n, limit=1)            # already in the graph?
    return hits[0]["key"] if hits else None


def _persist(result: dict, source: str = "extractor") -> int:
    """Write entities, relationships, facts, and episodes via memory.py. Returns #writes."""
    n = 0
    entity_map: dict[str, str] = {}

    # 1) entities -> nodes (name -> key)
    for e in result.get("entities", []):
        name = (e.get("name") or "").strip()
        etype = (e.get("type") or "").strip().lower()
        if not name or etype not in mem.PRIMARY_LABEL or name.lower() in _SELF_NAMES:
            continue
        key = mem.upsert_entity(etype, name, source=source)
        entity_map[name] = key
        n += 1

    # 2) relationships -> edges (natural word -> graph edge, + sub-label on object)
    for r in result.get("relationships", []):
        rel_word = (r.get("relationship") or "").strip().lower()
        edge, sublabel = _REL_MAP.get(rel_word, (None, None))
        if edge is None:                     # fall back: keep the link, don't lose it
            edge = re.sub(r"[^A-Z_]", "", rel_word.upper().replace(" ", "_")) or "RELATED_TO"
        s_key = _resolve_key(r.get("subject", "self"), entity_map)
        o_name = (r.get("object") or "").strip()
        o_key = _resolve_key(o_name, entity_map)
        if o_key is None and o_name and o_name.lower() not in _SELF_NAMES:
            o_key = mem.upsert_entity("person", o_name, source=source)  # assume a person
            entity_map[o_name] = o_key
        if sublabel and o_key:               # e.g. mark the relative as :Family
            mem.upsert_entity("person", o_name, sublabels=[sublabel], source=source)
        if s_key and o_key and s_key != o_key:
            mem.link(s_key, edge, o_key)
            n += 1

    # 3) facts -> mutable :Fact nodes (subject can be self or any entity)
    for f in result.get("facts", []):
        s_key = _resolve_key(f.get("subject", "self"), entity_map)
        pred = _snake(f.get("predicate", ""))
        val = (f.get("value") or "").strip()
        if s_key and pred and val:
            mem.remember_fact(s_key, pred, val, source=source)
            n += 1

    # 4) episodes -> appended :Memory, linked to any entities mentioned
    links = list(entity_map.values()) or None
    for text in result.get("episodes", []):
        text = (text or "").strip()
        if len(text) >= 6:
            mem.remember_episode(text, links=links, source=source)
            n += 1
    return n


# ---- the drain loop -------------------------------------------------------- #
def drain(limit: int = 50, session_id: str | None = None) -> dict:
    """
    Process pending turns: claim -> gate -> extract -> persist -> mark_done.
    Gated-out turns are marked done with no LLM call. Returns a stats dict.
    Requires Neo4j reachable; if not, claimed rows are released back to pending.
    """
    wm.init_db()
    if not mem_reachable():
        logger.info("drain skipped: Neo4j not reachable (rows stay pending)")
        return {"skipped": "neo4j_unreachable"}

    claimed = wm.claim_pending(limit=limit, session_id=session_id)
    gated = extracted = writes = failed = 0
    for row in claimed:
        msg_id, text = row["msg_id"], row["text"]
        try:
            if not is_memory_worthy(text):
                wm.mark_done(msg_id)
                gated += 1
                continue
            result = extract(text)
            writes += _persist(result)
            wm.mark_done(msg_id)
            extracted += 1
        except Exception as e:  # noqa: BLE001
            wm.mark_failed(msg_id)
            failed += 1
            logger.warning("drain: msg %s failed: %s", msg_id, e)
    stats = {"claimed": len(claimed), "gated_out": gated,
             "extracted": extracted, "writes": writes, "failed": failed}
    if claimed:
        logger.info("drain: %s", stats)
    return stats


def mem_reachable() -> bool:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph"))
        from connection import verify  # type: ignore
        return verify()
    except Exception:  # noqa: BLE001
        return False


def drain_all(prune: bool = False) -> dict:
    """Drain everything pending; optionally prune + vacuum SQLite afterward."""
    total = {"claimed": 0, "gated_out": 0, "extracted": 0, "writes": 0, "failed": 0}
    wm.reset_stale_processing()
    while True:
        s = drain(limit=50)
        if "skipped" in s or not s.get("claimed"):
            break
        for k in total:
            total[k] += s.get(k, 0)
        if s["claimed"] < 50:
            break
    if prune:
        total["prune"] = wm.prune()
        wm.vacuum()
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    out = drain_all(prune="--prune" in sys.argv)
    print("drain result:", out)
