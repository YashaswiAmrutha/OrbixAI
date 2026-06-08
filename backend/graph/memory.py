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
_NEIGHBOR_PROPS = {
    "birthday", "anniversary", "date", "email", "phone", "city", "country",
    "value", "status", "due_date", "start_date", "end_date", "deadline",
    "role", "relationship", "recurrence", "title", "category",
}


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
                  props: properties(o)} END) AS rels
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
                  if k in _NEIGHBOR_PROPS and v is not None}
        entry = {"rel": x["rel"], "name": x["name"], "key": x["key"], "labels": x["labels"]}
        if detail:
            entry["details"] = detail
        rels.append(entry)
    return {
        "key": subject_key,
        "labels": [l for l in r["node"].labels if l != "Entity"],
        "properties": props,
        "facts": r["facts"],
        "relationships": rels,
    }


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
