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
import threading
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph"))
import working_memory as wm   # type: ignore  # noqa: E402
import memory as mem          # type: ignore  # noqa: E402

logger = logging.getLogger(__name__)

# Reuse the agent model by default (already pulled); swap for a smaller one later.
# NOTE: this was "llama3.2:3b", a tag not actually pulled on this machine
# (`ollama list` only has llama3.2:latest) — every call silently failed and
# fell through to the empty-result path, meaning background memory extraction
# (durable facts written to Neo4j after each turn) was never actually running.
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
    "- relationships: links as {subject, relationship, object}. Use whatever "
    "NATURAL word describes the connection — this is NOT a fixed list, extract "
    "ANY role/relation the message states (family: mother, father, sister, "
    "brother, spouse; social/work: friend, colleague, manager, assistant, "
    "roommate, doctor, landlord, mentor, client, 'works at', 'works on', "
    "'member of'; or just knows). If the message names ANY person and how they "
    "relate to the user, that is ALWAYS a relationship to extract, even if the "
    "exact word isn't in this example list. Always write from the USER's view: "
    "a parent is the user's 'mother'/'father'; a child is 'son'/'daughter'.\n"
    "- facts: single-valued CHANGEABLE attributes as {subject, predicate, value}, "
    "subject = 'self' or an entity name. snake_case predicates like home_city, "
    "employer, job_title, marital_status. A person's city of residence is a FACT "
    "(home_city), NOT a relationship. For an email address ALWAYS use the exact "
    "predicate 'email' (never mail_id/email_address/contact_email/e_mail); for a "
    "phone number ALWAYS use exactly 'phone' (never mobile/contact_number/cell) — "
    "these two get read back by an exact predicate match, so any other spelling "
    "makes the value permanently unreadable even though it was written.\n"
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
    # Edge-type strings here must match memory.py's _FAMILY_RELS/_PERSONAL_RELS
    # EXACTLY (recall()'s priority-tier ranking does a plain set-membership check,
    # not a substring/fuzzy match) — confirmed live: "FRIENDS_WITH" here vs.
    # "IS_FRIENDS_WITH" there meant every newly-extracted friend silently landed
    # in the generic tier and got crowded out by older clutter, no matter how
    # recent. The "IS_" prefix is the pre-existing convention (already used by
    # seed data / _FAMILY_RELS's IS_MARRIED_TO) — matched here, not changed there.
    "husband": ("IS_MARRIED_TO", "Family"), "wife": ("IS_MARRIED_TO", "Family"),
    "spouse": ("IS_MARRIED_TO", "Family"),
    "friend": ("IS_FRIENDS_WITH", "Friend"),
    "colleague": ("KNOWS", "Colleague"), "coworker": ("KNOWS", "Colleague"),
    "manager": ("REPORTS_TO", None), "boss": ("REPORTS_TO", None),
    "works at": ("WORKS_AT", None), "work at": ("WORKS_AT", None),
    "works for": ("WORKS_AT", None), "employer": ("WORKS_AT", None),
    "works on": ("WORKS_ON", None), "working on": ("WORKS_ON", None),
    "member of": ("MEMBER_OF", None), "knows": ("KNOWS", None),
    "contact": ("KNOWS", "Contact"),
    # Extended kinship. Confirmed live: "my cousin Neha" produced a bare COUSIN edge
    # with NO sublabel, so recall() ranked a real relative in the generic tier beneath
    # stale project/meeting nodes. recall() now also ranks any directly-linked :Person
    # as personal (the durable, vocabulary-independent safety net), but mapping the
    # common kinship words here is what puts actual family in the top tier rather than
    # merely above clutter.
    "cousin": ("RELATED_TO", "Family"),
    "aunt": ("RELATED_TO", "Family"), "uncle": ("RELATED_TO", "Family"),
    "niece": ("RELATED_TO", "Family"), "nephew": ("RELATED_TO", "Family"),
    "grandmother": ("RELATED_TO", "Family"), "grandfather": ("RELATED_TO", "Family"),
    "grandparent": ("RELATED_TO", "Family"), "grandson": ("RELATED_TO", "Family"),
    "granddaughter": ("RELATED_TO", "Family"),
    "in-law": ("RELATED_TO", "Family"), "relative": ("RELATED_TO", "Family"),
    "partner": ("IS_MARRIED_TO", "Family"), "fiance": ("IS_MARRIED_TO", "Family"),
    # Role contacts: actionable people, so tag them so they survive truncation.
    "assistant": ("KNOWS", "Contact"), "doctor": ("KNOWS", "Contact"),
    "landlord": ("KNOWS", "Contact"), "neighbor": ("NEIGHBOR_OF", "Contact"),
    "classmate": ("KNOWS", "Friend"), "mentor": ("KNOWS", "Colleague"),
    "client": ("KNOWS", "Client"), "teammate": ("KNOWS", "Colleague"),
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
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.IGNORECASE)


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
    # An email address is a strong, rarely-false-positive signal that this turn is
    # naming a durable contact — worth one LLM look even inside a question/command
    # sentence. Checked BEFORE the is_question short-circuit on purpose: confirmed
    # live that "Do i hv any upcoming events? if yes, do mail the list of events to
    # my assistant (x@y.com)" was being dropped whole (leading "Do" + trailing "?"
    # both read as a pure question), silently losing the one durable fact — the
    # assistant's email — buried in what is really a conditional command, not a
    # question the gate should be filtering as unanswerable-yet-assertable chitchat.
    if _EMAIL.search(t):
        return True
    is_question = t.endswith("?") or bool(re.match(
        r"^(what|where|when|who|which|how|why|is|are|do|does|did|can|could|should|"
        r"would|will|tell me|show me|list|remind)\b", t, re.IGNORECASE))
    if is_question:
        return False
    # a statement worth a look if it has any memorable signal
    return bool(_FIRST_PERSON.search(t) or _DECLARATIVE_HINT.search(t)
                or _PROPER_NOUN.search(t[1:]) or _NUM_OR_EMAIL.search(t))


# ---- the extractor: small structured-output model -------------------------- #
# The bare `ollama.chat()` module function uses a shared default client with NO
# request timeout — confirmed live via py-spy: 8 threads were found permanently
# blocked inside this exact call, reading an HTTP response that never returned
# (likely Ollama busy/thrashing swapping models while a different model was
# active for the main agent). A blocked synchronous call in a worker thread
# can't be cancelled from outside (asyncio.wait_for only stops AWAITING it, not
# the thread itself) — so the only real fix is bounding the call itself. 90s is
# generous for CPU inference on this small a schema; a genuine timeout here just
# means the row is marked failed and retried later (already-safe per drain()'s
# existing exception handling), never an unbounded hang.
_EXTRACT_CLIENT = ollama.Client(timeout=90.0)


def extract(text: str) -> dict:
    """Run the constrained model on one turn. Returns the four extraction arrays."""
    try:
        resp = _EXTRACT_CLIENT.chat(
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


# Defense-in-depth for the prompt instruction above: predicates are matched
# EXACTLY elsewhere (memory.NEIGHBOR_PROPS decides what gets mirrored onto an
# entity property and is therefore readable back — confirmed live: a fact
# written as "mail_id" was invisible forever because nothing reads that exact
# string). A prompt instruction alone isn't reliable for a small model, so
# normalize the common synonyms it's actually produced at write time too.
_PREDICATE_CANON = {
    syn: canon
    for canon, syns in {
        "email": ("email", "email_address", "mail_id", "mail", "e_mail",
                  "contact_email", "emailid", "email_id"),
        "phone": ("phone", "phone_number", "mobile", "mobile_number",
                  "contact_number", "cell", "cell_number"),
    }.items()
    for syn in syns
}


def _canon_predicate(pred: str) -> str:
    return _PREDICATE_CANON.get(pred, pred)


# A small structured-output model occasionally emits its OWN schema/instruction
# artifacts as if they were extracted content, instead of failing loudly. These
# shapes are generic — not tied to any one predicate or phrasing — so this is a
# write-time plausibility gate, not a per-incident rule: it rejects the FORM a
# hallucinated value takes (an unfilled placeholder, a value that's just the
# predicate echoed back), whatever the underlying message happened to be.
_PLACEHOLDER_VALUE = re.compile(r"^[\[<{].*[\]>}]$")

# Some predicates have a checkable FORMAT, and those are exactly the ones that do
# the most damage when wrong: memory.NEIGHBOR_PROPS mirrors them onto the entity's
# own node, so recall() inlines them into the system prompt on every single turn.
# Confirmed live: user:self carried `email = "mail about message to pes2ug"` — a
# fragment of an unrelated sentence stored as the user's own address, and fed to the
# model as ground truth forever after. The shape checks above can't catch that: it's
# non-empty, unbracketed, and not an echo of the predicate. So validate the value
# against the predicate's known format wherever a format exists.
_EMAIL_VALUE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


def _valid_email(v: str) -> bool:
    return bool(_EMAIL_VALUE.match(v))


def _valid_phone(v: str) -> bool:
    # Accept the usual separators/country prefixes; reject prose that merely
    # happens to contain a number.
    digits = re.sub(r"\D", "", v)
    return 7 <= len(digits) <= 15 and bool(re.match(r"^[+()\-\s\d.]+$", v))


_PREDICATE_VALIDATORS = {
    "email": _valid_email,
    "phone": _valid_phone,
}


def _is_plausible_fact(predicate: str, value: str) -> bool:
    """Reject a fact value that looks like an extraction artifact rather than
    real content. Confirmed live: `email_address: "[user's email address]"` and
    `spouse: "spouse"` were both persisted as ground-truth facts with 0.9
    confidence — the model couldn't find a real value and echoed the schema's
    own placeholder / the predicate name back instead of returning nothing."""
    v = value.strip()
    if not v:
        return False
    if _PLACEHOLDER_VALUE.match(v):
        return False
    pred_words = predicate.strip().lower().replace("_", " ")
    if v.lower() in (predicate.strip().lower(), pred_words):
        return False
    validator = _PREDICATE_VALIDATORS.get(_canon_predicate(predicate.strip().lower()))
    if validator and not validator(v):
        return False
    return True


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
    _SELF_KEY = "user:self"
    self_linked: set[str] = set()   # entities reachable via an explicit self -> X edge
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
            if s_key == _SELF_KEY:
                self_linked.add(o_key)
                # Register the ROLE word itself as a searchable alias (the
                # entitySearch fulltext index already covers e.aliases —
                # schema.cypher — nothing populated it until now). Without
                # this, an entity is only findable by its literal name: a later
                # "email my assistant" can't resolve to "burgirr" via search()
                # at all, only "email burgirr" can — but the user refers to
                # people by role far more often than by re-stating their name.
                if rel_word:
                    mem.add_alias(o_key, rel_word)

    # Safety net: recall('user:self') can only ever reach an entity via a direct
    # self -> X edge (one-hop traversal). If the extraction model names an entity
    # (and even attaches real facts to it) but skips the relationship triple —
    # confirmed live for "assistant", not in this prompt's example list, and the
    # same gap would hit ANY relation word the model doesn't happen to reach for —
    # that entity is otherwise permanently orphaned: correct data, unreachable.
    # A generic self -[:MENTIONS]-> edge guarantees every entity born from a
    # message about the user's life stays reachable regardless of whether the
    # model captured the specific relationship correctly.
    for key in entity_map.values():
        if key != _SELF_KEY and key not in self_linked:
            mem.link(_SELF_KEY, "MENTIONS", key)
            n += 1

    # 3) facts -> mutable :Fact nodes (subject can be self or any entity)
    for f in result.get("facts", []):
        s_key = _resolve_key(f.get("subject", "self"), entity_map)
        pred = _canon_predicate(_snake(f.get("predicate", "")))
        val = (f.get("value") or "").strip()
        if not (s_key and pred and val):
            continue
        if not _is_plausible_fact(pred, val):
            logger.warning("Rejected implausible fact %s.%s = %r (source=%s)",
                           s_key, pred, val, source)
            continue
        mem.remember_fact(s_key, pred, val, source=source)
        n += 1
        # recall(subject_key) only ever joins :Fact nodes for the EXACT key
        # being recalled — a Fact about a one-hop neighbor (e.g. "my
        # assistant's email") never surfaces from recall('user:self'), only
        # from recall(that neighbor's own key), which nothing ever calls
        # unless the model already knows to. Mirror salient identity-ish
        # predicates (memory.py's NEIGHBOR_PROPS: email, phone, birthday, ...)
        # onto the entity's own property too, since THAT is what recall()'s
        # one-hop neighbor inlining actually reads.
        if pred in mem.NEIGHBOR_PROPS:
            mem.set_property(s_key, pred, val)

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


# drain_all() is called from three independent places (agent.py's pre-turn hard
# wait, main.py's periodic maintenance loop, and the manual /workflow/cleanup
# trigger) and each call can block for a while on a slow extract() (even with
# Fix A's per-call timeout, a real backlog is still N x up-to-90s sequentially).
# Without this lock, every new turn while a drain is already in flight spawns
# ANOTHER thread running its own drain_all() — confirmed live via py-spy: 8
# threads piled up this way, nearly saturating the shared thread pool and
# making unrelated requests appear to hang/loop too. One drain in flight at a
# time; anything that arrives while it's running just no-ops instead of piling
# on — the in-flight drain already covers whatever's pending.
_drain_lock = threading.Lock()


def drain_all(prune: bool = False) -> dict:
    """Drain everything pending; optionally prune + vacuum SQLite afterward.
    No-ops (returns immediately) if another drain is already in progress."""
    if not _drain_lock.acquire(blocking=False):
        logger.info("drain_all: another drain already in progress, skipping")
        return {"skipped": "already_draining"}
    try:
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
    finally:
        _drain_lock.release()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    out = drain_all(prune="--prune" in sys.argv)
    print("drain result:", out)
