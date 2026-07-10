#conversation_memory.py
"""Persistent, ordered conversation memory backed by Neo4j.

Important session rule:
The frontend owns the current browser-page session id. The backend must never
auto-resume a previous conversation when it sees a new id, because that leaks
old travel details (for example Singapore) into a fresh request (for example
Kashmir). A page reload is a new Conversation node.
"""

import json
import re
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from graph.neo4j_client import neo4j_client


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize_conversation_schema() -> None:
    for query in [
        "CREATE CONSTRAINT conversation_id_unique IF NOT EXISTS "
        "FOR (c:Conversation) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT message_id_unique IF NOT EXISTS "
        "FOR (m:Message) REQUIRE m.id IS UNIQUE",
        "CREATE INDEX message_created_at IF NOT EXISTS "
        "FOR (m:Message) ON (m.created_at)",
        "CREATE INDEX conversation_updated_at IF NOT EXISTS "
        "FOR (c:Conversation) ON (c.updated_at)",
    ]:
        neo4j_client.run_query(query)


def add_message(conversation_id: str, role: str, content: str,
                metadata: dict | None = None) -> str:
    message_id, created_at = str(uuid4()), _now()
    neo4j_client.run_query("""
    MERGE (u:User {id: $user_id})
      ON CREATE SET u.name = $user_name, u.created_at = $created_at
    MERGE (c:Conversation {id: $conversation_id})
      ON CREATE SET
        c.created_at = $created_at,
        c.title = $title,
        c.session_id = $conversation_id,
        c.source = 'browser-page-session',
        c.memory_scope = 'session-only'
    SET c.updated_at = $created_at,
        c.last_role = $role
    MERGE (u)-[:OWNS]->(c)
    WITH c
    OPTIONAL MATCH (c)-[:HAS_MESSAGE]->(existing:Message)
    WITH c, count(existing) AS turn_index
    CREATE (m:Message {
      id: $message_id, role: $role, content: $content,
      created_at: $created_at, turn_index: turn_index,
      metadata_json: $metadata_json
    })
    CREATE (c)-[:HAS_MESSAGE]->(m)
    """, {
        "user_id": "default-user", "user_name": "Amrutha",
        "conversation_id": conversation_id, "message_id": message_id,
        "role": role, "content": content, "created_at": created_at,
        "title": content[:80],
        "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
    })
    return message_id


def get_recent_messages(conversation_id: str, limit: int = 12) -> list[dict]:
    return neo4j_client.run_query("""
    MATCH (c:Conversation {id: $conversation_id})-[:HAS_MESSAGE]->(m:Message)
    WITH m ORDER BY m.created_at DESC LIMIT $limit
    RETURN m.id AS id, m.role AS role, m.content AS content,
           m.created_at AS created_at, m.turn_index AS turn_index,
           m.metadata_json AS metadata_json
    ORDER BY created_at ASC
    """, {"conversation_id": conversation_id, "limit": limit})


def resolve_conversation_id(requested_id: str) -> str:
    """Use only the id supplied by the current browser page session."""
    return requested_id


def resolve_referential_conversation_id(requested_id: str, current_message: str,
                                        resume_minutes: int = 240) -> str:
    """Recover history only for ambiguous follow-ups when the new id is empty.

    This is intentionally narrow. Fresh explicit prompts should keep their new
    page-session id. But if the UI loses its id while the visible chat continues,
    a message like "flights for that" would otherwise have no destination.
    """
    current = current_message or ""
    is_referential = re.search(
        r"\b(that|there|it|same|also|too|those|this trip|that trip|for that)\b",
        current,
        re.IGNORECASE,
    )
    if not is_referential:
        return requested_id

    existing = neo4j_client.run_query("""
    MATCH (c:Conversation {id: $id})-[:HAS_MESSAGE]->(m:Message)
    RETURN count(m) AS count
    """, {"id": requested_id})
    if existing and existing[0].get("count", 0) > 0:
        return requested_id

    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=resume_minutes)).isoformat()
    latest = neo4j_client.run_query("""
    MATCH (:User {id: 'default-user'})-[:OWNS]->(c:Conversation)-[:HAS_MESSAGE]->(m:Message)
    WHERE c.updated_at >= $cutoff
      AND c.memory_scope = 'session-only'
      AND m.role = 'assistant'
      AND (
        toLower(m.content) CONTAINS 'travel plan for'
        OR toLower(m.content) CONTAINS 'flight details for'
        OR toLower(m.content) CONTAINS 'hotel options for'
        OR toLower(m.content) CONTAINS 'itinerary for'
        OR toLower(m.content) CONTAINS 'travel details for'
      )
      AND NOT toLower(m.content) CONTAINS 'could not determine'
    RETURN c.id AS id, max(c.updated_at) AS updated_at
    ORDER BY updated_at DESC
    LIMIT 1
    """, {"cutoff": cutoff})
    return latest[0]["id"] if latest else requested_id


def build_context(messages: list[dict], current_message: str) -> str:
    if not messages:
        return current_message
    transcript = "\n".join(
        f"{item.get('role', 'user').title()}: {item.get('content', '')[:1200]}"
        for item in messages[-8:]
    )
    return (
        "Use only the previous messages from this exact browser-page session "
        "to resolve references and missing details. Do not use older sessions "
        "or unrelated trips. The final user message is the current request.\n\n"
        f"Previous messages in this session:\n{transcript}\n\n"
        f"Current user message: {current_message}"
    )
