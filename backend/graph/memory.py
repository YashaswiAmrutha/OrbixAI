"""
backend/graph/memory.py — OrbixAI's long-term memory engine (Neo4j).

The single implementation of the **resolve -> reconcile -> upsert** write protocol
and the **current-only** read protocol from Docs/graph-schema.md. Both
`load_profile.py` and the (future) MCP memory server import this module, so the
write discipline lives in exactly one place and the model can never create a
duplicate or a contradiction even if it tries.

Key design decision — mutable attributes are :Fact nodes, not properties:
    A single-valued attribute that can change over time (home_city, employer,
    marital_status, ...) is stored as a `:Fact` keyed by `subject_key + predicate`.
    A new value OVERWRITES the same node and the old value is archived as an
    `:Observation {current:false}`. This is the anti-contradiction core (§4): the
    User node holds stable identity (name, email, birthday); changeable facts live
    in :Fact so there is exactly one place a contradiction can land.

Tool surface (maps to the MCP memory tools, §7.2):
    read :  recall · get_entity · get_fact · fact_history · search
    write:  upsert_entity · remember_fact · correct · link · remember_episode · forget
"""

import logging
import re

from connection import get_driver  # type: ignore  (run from this dir)

logger = logging.getLogger(__name__)

# type -> primary Neo4j label
PRIMARY_LABEL = {
    "user": "User", "person": "Person", "organization": "Organization",
    "role": "Role", "team": "Team", "email": "Email", "message": "Message",
    "conversation": "Conversation", "meeting": "Meeting", "task": "Task",
    "project": "Project", "reminder": "Reminder", "event": "Event",
    "trip": "Trip", "flight": "Flight", "stay": "Stay", "location": "Location",
    "fact": "Fact", "memory": "Memory", "observation": "Observation",
    "preference": "Preference", "interest": "Interest", "note": "Note",
    "topic": "Topic", "document": "Document", "medication": "Medication",
}


# --------------------------------------------------------------------------- #
# low-level run helpers (each opens a short managed session)
# --------------------------------------------------------------------------- #
def _write(cypher: str, **params):
    with get_driver().session() as s:
        return s.execute_write(lambda tx: list(tx.run(cypher, **params)))


def _read(cypher: str, **params):
    with get_driver().session() as s:
        return s.execute_read(lambda tx: list(tx.run(cypher, **params)))


def _nonempty(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip() != ""
    if isinstance(v, (list, tuple, dict)):
        return len(v) > 0
    return True


def _label_str(type_: str, sublabels=None) -> str:
    """'person', ['Family'] -> 'Person:Family' (primary + sub-labels)."""
    primary = PRIMARY_LABEL.get(type_)
    if not primary:
        raise ValueError(f"Unknown entity type: {type_!r}")
    parts = [primary] + [s for s in (sublabels or []) if s]
    return ":".join(parts)


# --------------------------------------------------------------------------- #
# identity — build the type-prefixed `key` (the MERGE target)
# --------------------------------------------------------------------------- #
def key_for(type_: str, name: str | None = None, **kw) -> str:
    """Compute a node's normalized identity key per the recipes in §1."""
    n = (name or "").strip().lower()
    if type_ == "user":
        return "user:self"
    if type_ == "person":
        return "person:" + n
    if type_ == "organization":
        return "org:" + (kw.get("domain") or n).lower()
    if type_ == "project":
        return "project:" + n
    if type_ == "interest":
        return "interest:" + n
    if type_ == "topic":
        return "topic:" + n
    if type_ == "medication":
        return "med:" + n
    if type_ == "preference":
        cat = (kw.get("category") or "general").lower()
        leaf = (kw.get("pref_key") or n).lower()
        return f"pref:{cat}:{leaf}"
    if type_ == "fact":
        return f"fact:{kw['subject_key']}:{kw['predicate']}"
    if type_ == "trip":
        return f"trip:{n}:{kw.get('start_date', '')}"
    # generic fallback
    if not n:
        raise ValueError(f"Cannot derive key for type {type_!r} without a name/key.")
    return f"{type_}:{n}"


# --------------------------------------------------------------------------- #
# WRITE — upsert_entity / remember_fact / link / remember_episode / forget
# --------------------------------------------------------------------------- #
def upsert_entity(type_: str, name: str | None = None, *, key: str | None = None,
                  props: dict | None = None, sublabels=None,
                  source: str = "manual") -> str:
    """
    MERGE a typed node on its identity key; set its props. Idempotent.
    Returns the key. `key` is auto-derived from (type, name) if not given.
    """
    key = key or key_for(type_, name=name, **(props or {}))
    labels = _label_str(type_, sublabels)
    clean = {k: v for k, v in (props or {}).items() if _nonempty(v) and k != "key"}
    # MERGE on the :Entity mixin + key only, then SET labels additively. This means
    # re-mentioning an entity with a NEW sub-label (e.g. plain :Person later becoming
    # :Person:Family) just adds the label instead of colliding on the key constraint.
    rows = _write(
        f"""
        MERGE (e:Entity {{key: $key}})
          ON CREATE SET e.id = randomUUID(), e.created_at = timestamp(), e.source = $source
        SET e:{labels}, e.type = $type, e.name = coalesce($name, e.name),
            e.updated_at = timestamp(), e.last_seen = timestamp(),
            e += $props
        RETURN e.key AS key
        """,
        key=key, type=type_, name=name, source=source, props=clean,
    )
    return rows[0]["key"]


def remember_fact(subject_key: str, predicate: str, value, *,
                  confidence: float = 0.9, source: str = "chat") -> dict:
    """
    Upsert a mutable single-valued fact (resolve -> reconcile -> upsert, §4.3).
    If a different current value exists it is archived as :Observation history,
    then overwritten. Returns {previous, current, changed}.
    """
    key = key_for("fact", subject_key=subject_key, predicate=predicate)
    rows = _write(
        """
        MERGE (f:Entity:Fact {key: $key})
          ON CREATE SET f.id = randomUUID(), f.subject_key = $subject_key,
                        f.predicate = $predicate, f.created_at = timestamp(),
                        f.source = $source
        WITH f, f.value AS old
        FOREACH (_ IN CASE WHEN old IS NOT NULL AND toString(old) <> toString($value)
                           THEN [1] ELSE [] END |
          CREATE (f)-[:HAS_OBSERVATION]->(:Observation {
              id: randomUUID(), text: toString(old), value: old,
              current: false, created_at: timestamp(),
              source: coalesce(f.source, $source) }))
        SET f.value = $value, f.current = true, f.confidence = $confidence,
            f.type = 'fact', f.name = $predicate,
            f.updated_at = timestamp(), f.last_seen = timestamp()
        RETURN old AS previous, f.value AS current
        """,
        key=key, subject_key=subject_key, predicate=predicate,
        value=value, confidence=confidence, source=source,
    )
    prev, cur = rows[0]["previous"], rows[0]["current"]
    return {"previous": prev, "current": cur, "changed": prev != cur}


def correct(subject_key: str, predicate: str, value, *, source: str = "chat") -> dict:
    """Explicit correction — same as remember_fact at high confidence."""
    return remember_fact(subject_key, predicate, value, confidence=0.95, source=source)


_SAFE_PROP_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")


def set_property(key: str, prop: str, value) -> bool:
    """
    Mirror a value directly onto an entity's own node property (not a :Fact node).
    Use this for salient identity-ish attributes (email, phone, birthday, ...) of a
    NON-self entity: recall(subject_key)'s one-hop neighbor inlining only reads a
    neighbor's own properties() — it does not join that neighbor's separate :Fact
    nodes (those only surface when the neighbor itself is the one being recalled).
    Without this, "my assistant's email is X" would be written correctly by
    remember_fact() but silently invisible from recall('user:self') forever, since
    nothing ever asks for person:assistant's own facts directly. Returns False if
    the entity doesn't exist (never creates one — upsert_entity owns creation).

    `prop` is validated against a strict snake_case pattern and interpolated into
    the Cypher property name (Cypher can't parameterize a property key in SET on
    every server version) — same trusted-identifier convention already used for
    dynamic label/relationship-type strings elsewhere in this module (_label_str,
    link()). Never pass a caller-supplied string here un-validated.
    """
    if not _nonempty(value):
        return False
    if not _SAFE_PROP_NAME.match(prop):
        raise ValueError(f"Unsafe property name: {prop!r}")
    rows = _write(
        f"MATCH (e:Entity {{key: $key}}) SET e.{prop} = $value, e.updated_at = timestamp() "
        f"RETURN e.key AS key",
        key=key, value=value,
    )
    return bool(rows)


def add_alias(key: str, alias: str) -> bool:
    """
    Append a searchable alias to an entity (idempotent — no duplicates). The
    `entitySearch` fulltext index already covers e.aliases (schema.cypher) but
    nothing ever wrote to it — e.g. an entity named "burgirr" was never
    findable by the ROLE the user actually calls them ("my assistant"), only by
    their literal name. Use this to register the relationship word itself
    (assistant, doctor, manager, ...) as an alias whenever a role is stated, so
    a later role-based query can resolve through search() instead of needing
    the exact name.
    """
    a = (alias or "").strip().lower()
    if not a:
        return False
    rows = _write(
        """
        MATCH (e:Entity {key: $key})
        SET e.aliases = CASE WHEN $alias IN coalesce(e.aliases, [])
                              THEN e.aliases ELSE coalesce(e.aliases, []) + $alias END,
            e.updated_at = timestamp()
        RETURN e.key AS key
        """,
        key=key, alias=a,
    )
    return bool(rows)


def link(a_key: str, rel: str, b_key: str, props: dict | None = None) -> None:
    """MERGE a relationship so an edge is never duplicated; bumps r.updated_at."""
    _write(
        f"""
        MATCH (a:Entity {{key: $a}}), (b:Entity {{key: $b}})
        MERGE (a)-[r:{rel}]->(b)
          ON CREATE SET r.created_at = timestamp()
        SET r.updated_at = timestamp(), r += $props
        """,
        a=a_key, b=b_key, props={k: v for k, v in (props or {}).items() if _nonempty(v)},
    )


def remember_episode(text: str, *, links: list[str] | None = None,
                     owner_key: str = "user:self", kind: str = "episodic",
                     importance: float = 0.5, source: str = "chat") -> str:
    """
    Append an episodic :Memory (episodes accumulate — they never contradict),
    link it to the owner via HAS_MEMORY and to mentioned entities via MENTIONS.
    Returns the memory key.
    """
    rows = _write(
        """
        WITH randomUUID() AS uid
        CREATE (m:Entity:Memory {
            id: uid, key: 'mem:' + uid, type: 'memory', text: $text,
            kind: $kind, importance: $importance, source: $source,
            created_at: timestamp(), updated_at: timestamp(), last_seen: timestamp() })
        WITH m
        OPTIONAL MATCH (owner:Entity {key: $owner})
        FOREACH (_ IN CASE WHEN owner IS NULL THEN [] ELSE [1] END |
          MERGE (owner)-[:HAS_MEMORY]->(m))
        WITH m
        UNWIND (CASE WHEN $links = [] THEN [null] ELSE $links END) AS lk
          OPTIONAL MATCH (e:Entity {key: lk})
          FOREACH (_ IN CASE WHEN e IS NULL THEN [] ELSE [1] END |
            MERGE (m)-[:MENTIONS]->(e))
        RETURN m.key AS key
        """,
        text=text, kind=kind, importance=importance, source=source,
        owner=owner_key, links=links or [],
    )
    return rows[0]["key"]


def forget(key: str) -> bool:
    """Soft-delete: mark archived + current:false (history is preserved)."""
    rows = _write(
        """
        MATCH (e:Entity {key: $key})
        SET e.archived = true, e.current = false, e.updated_at = timestamp()
        RETURN e.key AS key
        """,
        key=key,
    )
    return bool(rows)


# --------------------------------------------------------------------------- #
# READ — current-only by default (§4.6)
# --------------------------------------------------------------------------- #
_SKIP_PROPS = {"id", "created_at", "updated_at", "last_seen", "embedding"}

# Salient scalar fields of a 1-hop neighbor to inline into recall(), so common
# single-hop questions are answerable without the model chaining a second recall.
# Public (not _-prefixed): extractor.py's _persist() mirrors facts about non-self
# entities onto this same set of node properties via set_property(), because
# recall() below only ever joins THIS set — never a neighbor's own :Fact nodes —
# so a fact write that isn't also mirrored here is invisible from recall('user:self').
NEIGHBOR_PROPS = {
    "birthday", "anniversary", "date", "email", "phone", "city", "country",
    "value", "status", "due_date", "start_date", "end_date", "deadline",
    "role", "relationship", "recurrence", "title", "category",
}


def count_unkeyed() -> int:
    """How many :Entity nodes are missing the `key` property (should always be 0)."""
    rows = _read("MATCH (e:Entity) WHERE e.key IS NULL RETURN count(e) AS c")
    return rows[0]["c"] if rows else 0


def backfill_missing_keys(limit: int = 1000) -> dict:
    """
    Assign deterministic keys to any :Entity node that has none, and label any
    keyed memory node that is missing :Entity.

    Why this exists as a runtime repair rather than a one-off script: `key` is the
    sole MERGE target for every write in this module, and a node without one is
    invisible to recall()/get_entity() AND can never be deduplicated — the next
    mention of the same person creates a second node instead of updating the first.
    That is the exact failure the resolve->reconcile->upsert protocol exists to
    prevent, so it must not be reachable by any code path.

    Neo4j COMMUNITY (what this project ships) cannot enforce this at the database
    level: property-existence and node-key constraints are Enterprise-only features,
    and a uniqueness constraint ignores nulls. So the invariant has to be maintained
    in code. Callers that write through upsert_entity() already satisfy it; this
    sweep is the safety net for anything that writes raw Cypher (the travel/trip and
    email-linking paths in orchestration/ historically did), plus any future code
    that forgets.

    Idempotent and safe to run repeatedly — it only ever fills in what is missing.
    """
    # Identity is not always carried by `name`: event-shaped nodes written by the
    # travel/calendar/email paths use `title` (Meeting, Trip, Todo) or `subject`
    # (Email) instead. Keying only on `name` left those permanently unrepairable —
    # 146 of them in the development graph — so fall back through the labels these
    # writers actually use before giving up.
    rows = _read(
        """
        MATCH (e:Entity)
        WHERE e.key IS NULL
          AND coalesce(e.name, e.title, e.subject) IS NOT NULL
        RETURN elementId(e) AS eid,
               coalesce(e.name, e.title, e.subject) AS name,
               labels(e) AS labels
        LIMIT $limit
        """,
        limit=limit,
    )
    assigned = collided = unnameable = 0
    for row in rows:
        name = (row.get("name") or "").strip()
        if not name:
            unnameable += 1
            continue
        labels = set(row.get("labels") or [])
        # Map the node's label onto the same type vocabulary key_for() uses, so a
        # repaired node is indistinguishable from one written correctly first time.
        if "Person" in labels:
            type_ = "person"
        elif "Organization" in labels:
            type_ = "organization"
        elif "Project" in labels:
            type_ = "project"
        elif "Location" in labels:
            type_ = "location"
        elif "Trip" in labels:
            type_ = "trip"
        elif "Interest" in labels:
            type_ = "interest"
        elif "Topic" in labels:
            type_ = "topic"
        elif "Meeting" in labels:
            type_ = "meeting"
        elif "Email" in labels:
            type_ = "email"
        elif "Flight" in labels:
            type_ = "flight"
        elif "Stay" in labels:
            type_ = "stay"
        elif "Todo" in labels or "Task" in labels:
            type_ = "task"
        elif "Document" in labels:
            type_ = "document"
        else:
            type_ = "entity"
        try:
            key = key_for(type_, name)
        except Exception:  # noqa: BLE001
            unnameable += 1
            continue
        # Never fold two distinct nodes onto one key here: merging them is a data
        # decision, not a repair, so a collision is left alone and reported.
        taken = _read("MATCH (o:Entity {key:$k}) RETURN count(o) AS c", k=key)[0]["c"]
        if taken:
            collided += 1
            continue
        _write("MATCH (e:Entity) WHERE elementId(e) = $eid SET e.key = $k",
               eid=row["eid"], k=key)
        assigned += 1

    # Second half of the invariant: a memory node that carries a key but lost the
    # :Entity label is equally unreachable, since every read starts from :Entity.
    relabeled = _write(
        """
        MATCH (n) WHERE n.key IS NOT NULL AND NOT n:Entity
          AND (n:Person OR n:Organization OR n:Project OR n:Location OR n:Trip
               OR n:Interest OR n:Topic OR n:Fact OR n:Memory)
        WITH n LIMIT $limit SET n:Entity RETURN count(n) AS c
        """,
        limit=limit,
    )
    relabeled_n = relabeled[0]["c"] if relabeled else 0

    result = {"assigned": assigned, "collided": collided,
              "unnameable": unnameable, "relabeled": relabeled_n,
              "remaining_unkeyed": count_unkeyed()}
    if assigned or relabeled_n:
        logger.info("backfill_missing_keys: %s", result)
    return result


def recall(subject_key: str) -> dict | None:
    """
    Resolve a subject and return its CURRENT view: live properties + current facts
    + outgoing relationships. Never returns superseded history (use fact_history).
    """
    rows = _read(
        """
        MATCH (e:Entity {key: $key})
        OPTIONAL MATCH (f:Fact {subject_key: $key})
          WHERE coalesce(f.current, true) = true
        WITH e, collect(DISTINCT CASE WHEN f IS NULL THEN null ELSE
              {predicate: f.predicate, value: f.value, confidence: f.confidence} END) AS facts
        OPTIONAL MATCH (e)-[r]->(o:Entity)
          WHERE type(r) <> 'HAS_OBSERVATION'
        RETURN e AS node,
               [x IN facts WHERE x IS NOT NULL] AS facts,
               collect(DISTINCT CASE WHEN o IS NULL THEN null ELSE
                 {rel: type(r), name: o.name, key: o.key,
                  labels: [l IN labels(o) WHERE l <> 'Entity'],
                  props: properties(o), rel_updated: coalesce(r.updated_at, 0)} END) AS rels
        """,
        key=subject_key,
    )
    if not rows:
        return None
    r = rows[0]
    node = dict(r["node"])
    props = {k: v for k, v in node.items() if k not in _SKIP_PROPS}
    rels = []
    for x in r["rels"]:
        if x is None:
            continue
        # include a few salient neighbor props inline so single-hop questions
        # ("mom's birthday", "colleague's email") are answerable from one recall.
        detail = {k: v for k, v in (x.get("props") or {}).items()
                  if k in NEIGHBOR_PROPS and v is not None}
        entry = {"rel": x["rel"], "name": x["name"], "key": x["key"], "labels": x["labels"],
                 "_updated": x.get("rel_updated") or 0}
        if detail:
            entry["details"] = detail
        rels.append(entry)
    # Cap what recall() hands back — a richly-populated graph can have hundreds
    # of relationships (projects, meetings, todos, emails...), and dumping all
    # of them as one raw tool result was observed overwhelming a small local
    # model: instead of answering a simple question ("who is my spouse"), it
    # started describing the JSON's shape instead of reading it. Personal/
    # identity relationship types are the ones most "who am I" style questions
    # need, so they're kept first when truncating; get_entity(key) or search()
    # remain available for anything specific that gets cut off here.
    # Two priority tiers: close family (few in number, highest-value for "who
    # am I" style questions) ranks above broader personal relations (friends/
    # interests, which can be numerous enough on their own to push family out
    # of a flat single-tier cap).
    # RELATED_TO is what extractor.py's _REL_MAP emits for extended kinship
    # (cousin/aunt/uncle/grandparent/...). Kept in lockstep here deliberately: these
    # two sets are compared by exact string, and letting them drift is precisely the
    # bug documented below.
    _FAMILY_RELS = {"IS_MARRIED_TO", "PARENT_OF", "CHILD_OF", "SIBLING_OF",
                    "NEIGHBOR_OF", "RELATED_TO"}
    _PERSONAL_RELS = {"IS_FRIENDS_WITH", "KNOWS", "WORKS_AT", "HAS_INTEREST"}
    # Sublabels are a small, deliberately-controlled vocabulary (extractor.py's
    # _REL_MAP attaches them alongside whatever freeform edge-type string it
    # generates) — trust them as a SECOND, more durable signal in addition to the
    # exact edge-type match above. Confirmed live: the edge-type string alone
    # drifted out of sync between the two files (FRIENDS_WITH vs IS_FRIENDS_WITH)
    # and silently demoted a real relationship to the generic tier; the "Friend"
    # sublabel stayed correct throughout because it's set from the same small
    # fixed set on both sides. Belt-and-suspenders against the exact-string check
    # drifting again for some future relationship word.
    _FAMILY_LABELS = {"Family"}
    _PERSONAL_LABELS = {"Friend", "Colleague", "Contact"}

    def _rel_rank(x):
        labels = set(x.get("labels") or [])
        if x["rel"] in _FAMILY_RELS or labels & _FAMILY_LABELS:
            return 0
        if x["rel"] in _PERSONAL_RELS or labels & _PERSONAL_LABELS:
            return 1
        # An entity with a real contact detail (email/phone) is actionable —
        # exactly what an "email X" / "call X" request needs to resolve — so it
        # should survive truncation on the same footing as friends/colleagues,
        # regardless of what the specific relationship word was (assistant,
        # doctor, landlord, ...). Without this, a correctly-linked, correctly-
        # detailed contact can still be silently crowded out by generic/stale
        # relationships (projects, old meetings) that vastly outnumber it.
        if (x.get("details") or {}).get("email") or (x.get("details") or {}).get("phone"):
            return 1
        # A directly-linked PERSON is inherently a personal relationship, whatever
        # word was used for it. The checks above are vocabulary-based, so any term
        # outside the known sets fell through to the generic tier — confirmed live:
        # a cousin added seconds earlier ranked below dozens of stale project and
        # meeting nodes and was at risk of being cut by the cap entirely. Ranking on
        # the LABEL instead of the relationship string is the durable form of this
        # check: it covers cousin, assistant, doctor, landlord and every other role
        # nobody thought to enumerate, and it cannot drift the way a word list does.
        if "Person" in labels:
            return 1
        return 2
    # Within a tier, most-recently-touched relationships come first. Without this,
    # a tier alone doesn't guarantee visibility: confirmed live with 54 existing
    # tier-1 (personal/contact) relationships already at the cap, a JUST-added
    # contact still lost to older ones on a plain stable sort. Recency is what a
    # user actually cares about most — "what did I just tell you" should never
    # lose to a relationship that happened to be written first.
    rels.sort(key=lambda x: (_rel_rank(x), -x["_updated"]))
    for x in rels:
        del x["_updated"]
    total = len(rels)
    _REL_CAP = 30
    truncated = total > _REL_CAP
    rels = rels[:_REL_CAP]

    # Fold each surviving neighbor's OWN current :Fact nodes into its details.
    #
    # Without this, recall() could only ever see neighbor values that happened to be
    # mirrored onto the node as one of NEIGHBOR_PROPS — so a fact stored correctly via
    # remember_fact() was invisible unless its predicate was on that whitelist.
    # Confirmed live: asked "where does my cousin Neha work?", the model answered "I
    # couldn't find any information about her workplace" while cheerfully reporting her
    # email — because `email` is whitelisted and `employer` is not, even though both
    # were written the same way by the same extractor.
    #
    # Widening the whitelist would only move the boundary; joining the neighbor's facts
    # removes it. Cost is bounded on purpose: this runs AFTER ranking and truncation, so
    # it touches at most _REL_CAP neighbors, in one batched query, capped per neighbor —
    # which keeps the payload small enough for a local model to actually read.
    _NEIGHBOR_FACT_CAP = 8
    neighbor_keys = [x["key"] for x in rels if x.get("key")]
    if neighbor_keys:
        try:
            fact_rows = _read(
                """
                MATCH (f:Fact) WHERE f.subject_key IN $keys
                  AND coalesce(f.current, true) = true
                RETURN f.subject_key AS key, f.predicate AS predicate,
                       toString(f.value) AS value
                """,
                keys=neighbor_keys,
            )
            by_key: dict[str, dict] = {}
            for fr in fact_rows:
                pred, val = fr.get("predicate"), fr.get("value")
                if not pred or val is None:
                    continue
                slot = by_key.setdefault(fr["key"], {})
                if len(slot) < _NEIGHBOR_FACT_CAP:
                    slot.setdefault(pred, val)
            for x in rels:
                extra = by_key.get(x.get("key") or "")
                if not extra:
                    continue
                detail = x.get("details") or {}
                # Mirrored node properties win: set_property() writes the canonical,
                # validated value, so it should not be shadowed by a stale Fact.
                for k, v in extra.items():
                    detail.setdefault(k, v)
                x["details"] = detail
        except Exception as e:  # noqa: BLE001
            # Enrichment is additive; a failure here must not take down recall itself.
            logger.warning("recall: neighbor-fact enrichment failed: %s", e)

    result = {
        "key": subject_key,
        "labels": [l for l in r["node"].labels if l != "Entity"],
        "properties": props,
        "facts": r["facts"],
        "relationships": rels,
    }
    if truncated:
        result["relationships_truncated"] = True
        result["relationships_total"] = total
        result["note"] = (f"Showing {_REL_CAP} of {total} relationships (personal ones "
                          "like spouse/family/friends first). Use search(name) or "
                          "get_entity(key) for anything specific not shown here.")
    return result


def get_entity(key: str) -> dict | None:
    """Full node + current facts + recent observation history on it (if any)."""
    base = recall(key)
    if base is None:
        return None
    hist = _read(
        """
        MATCH (e:Entity {key: $key})-[:HAS_OBSERVATION]->(o:Observation)
        RETURN o.text AS text, o.created_at AS created_at, o.current AS current
        ORDER BY o.created_at DESC LIMIT 20
        """,
        key=key,
    )
    base["history"] = [dict(h) for h in hist]
    return base


def get_fact(subject_key: str, predicate: str) -> dict | None:
    """Return the current value of a single mutable fact."""
    rows = _read(
        "MATCH (f:Fact {key: $key}) WHERE coalesce(f.current, true) = true "
        "RETURN f.value AS value, f.confidence AS confidence, f.updated_at AS updated_at",
        key=key_for("fact", subject_key=subject_key, predicate=predicate),
    )
    return dict(rows[0]) if rows else None


def fact_history(subject_key: str, predicate: str) -> dict:
    """The 'what did I used to...' read — current value plus archived past values."""
    key = key_for("fact", subject_key=subject_key, predicate=predicate)
    rows = _read(
        """
        MATCH (f:Fact {key: $key})
        OPTIONAL MATCH (f)-[:HAS_OBSERVATION]->(o:Observation)
        RETURN f.value AS current,
               collect(CASE WHEN o IS NULL THEN null ELSE
                 {value: o.text, at: o.created_at} END) AS past
        """,
        key=key,
    )
    if not rows:
        return {"current": None, "past": []}
    return {"current": rows[0]["current"],
            "past": [p for p in rows[0]["past"] if p is not None]}


import re as _re

# Lucene reserved characters that must be escaped in a fulltext query.
_LUCENE_SPECIAL = _re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')


def _lucene_query(query: str) -> str:
    """
    Turn a plain phrase into a forgiving Lucene query: each term gets a prefix
    wildcard and a fuzzy variant, so "orbix" matches "OrbixAI" and small typos
    still hit. Reserved characters are escaped.
    """
    terms = [t for t in _re.split(r"\s+", query.strip()) if t]
    clauses = []
    for t in terms:
        safe = _LUCENE_SPECIAL.sub(r"\\\1", t)
        clauses.append(f"({safe}* OR {safe}~)")
    return " AND ".join(clauses) if clauses else query


def search(query: str, limit: int = 10) -> list[dict]:
    """Keyword search over names/summaries/aliases (the entitySearch fulltext index)."""
    rows = _read(
        """
        CALL db.index.fulltext.queryNodes('entitySearch', $q) YIELD node, score
        RETURN node.key AS key, node.name AS name,
               [l IN labels(node) WHERE l <> 'Entity'] AS labels, score
        ORDER BY score DESC LIMIT $limit
        """,
        q=_lucene_query(query), limit=limit,
    )
    return [dict(r) for r in rows]
