// ============================================================================
// OrbixAI — Neo4j Memory Graph schema (constraints + indexes)
// Applied by backend/graph/init_schema.py. Idempotent (IF NOT EXISTS).
// Compatible with Neo4j 5.x Community. Enterprise-only / version-gated
// statements are clearly marked and commented out.
// See Docs/graph-schema.md for the full model.
// ============================================================================

// ---------------------------------------------------------------------------
// 1. UNIQUENESS CONSTRAINTS  (anti-duplicate identity — Community-safe)
// ---------------------------------------------------------------------------

// Global identity: every memory node shares the :Entity mixin and a unique key.
// This is what makes "resolve -> MERGE -> reconcile" never create duplicates.
CREATE CONSTRAINT entity_key IF NOT EXISTS
FOR (e:Entity) REQUIRE e.key IS UNIQUE;

// External-id uniqueness so re-ingesting the same source row updates in place.
CREATE CONSTRAINT user_email IF NOT EXISTS
FOR (u:User) REQUIRE u.email IS UNIQUE;

CREATE CONSTRAINT email_gmail_id IF NOT EXISTS
FOR (m:Email) REQUIRE m.gmail_id IS UNIQUE;

CREATE CONSTRAINT meeting_event_id IF NOT EXISTS
FOR (mt:Meeting) REQUIRE mt.gcal_event_id IS UNIQUE;

CREATE CONSTRAINT message_external_id IF NOT EXISTS
FOR (msg:Message) REQUIRE msg.external_id IS UNIQUE;

CREATE CONSTRAINT document_id IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT memory_id IF NOT EXISTS
FOR (mem:Memory) REQUIRE mem.id IS UNIQUE;

CREATE CONSTRAINT observation_id IF NOT EXISTS
FOR (o:Observation) REQUIRE o.id IS UNIQUE;

CREATE CONSTRAINT reminder_id IF NOT EXISTS
FOR (r:Reminder) REQUIRE r.id IS UNIQUE;

CREATE CONSTRAINT note_id IF NOT EXISTS
FOR (n:Note) REQUIRE n.id IS UNIQUE;

// ---------------------------------------------------------------------------
// 2. EXISTENCE CONSTRAINTS  (Neo4j Enterprise only — uncomment if available)
// The upsert wrapper already guarantees `key`, so these are belt-and-braces.
// ---------------------------------------------------------------------------
// CREATE CONSTRAINT entity_key_exists IF NOT EXISTS
// FOR (e:Entity) REQUIRE e.key IS NOT NULL;
// CREATE CONSTRAINT entity_id_exists IF NOT EXISTS
// FOR (e:Entity) REQUIRE e.id IS NOT NULL;

// ---------------------------------------------------------------------------
// 3. LOOKUP INDEXES  (range/btree — speed up the common filters)
// ---------------------------------------------------------------------------
CREATE INDEX entity_id        IF NOT EXISTS FOR (e:Entity)  ON (e.id);
CREATE INDEX entity_type      IF NOT EXISTS FOR (e:Entity)  ON (e.type);
CREATE INDEX entity_updated   IF NOT EXISTS FOR (e:Entity)  ON (e.updated_at);
CREATE INDEX entity_name      IF NOT EXISTS FOR (e:Entity)  ON (e.name);

CREATE INDEX person_email     IF NOT EXISTS FOR (p:Person)  ON (p.email);
CREATE INDEX person_name      IF NOT EXISTS FOR (p:Person)  ON (p.name);

CREATE INDEX email_timestamp  IF NOT EXISTS FOR (m:Email)   ON (m.timestamp);
CREATE INDEX email_thread     IF NOT EXISTS FOR (m:Email)   ON (m.thread_id);

CREATE INDEX task_due         IF NOT EXISTS FOR (t:Task)    ON (t.due_date);
CREATE INDEX task_status      IF NOT EXISTS FOR (t:Task)    ON (t.status);

CREATE INDEX meeting_start    IF NOT EXISTS FOR (mt:Meeting) ON (mt.start_time);
CREATE INDEX trip_start       IF NOT EXISTS FOR (tr:Trip)   ON (tr.start_date);
CREATE INDEX reminder_at      IF NOT EXISTS FOR (r:Reminder) ON (r.remind_at);
CREATE INDEX event_date       IF NOT EXISTS FOR (ev:Event)  ON (ev.date);
CREATE INDEX project_name     IF NOT EXISTS FOR (pr:Project) ON (pr.name);
CREATE INDEX topic_name       IF NOT EXISTS FOR (tp:Topic)  ON (tp.name);

// Mutable single-valued facts: fast "all current facts about this subject".
CREATE INDEX fact_subject     IF NOT EXISTS FOR (f:Fact)    ON (f.subject_key);
CREATE INDEX fact_current     IF NOT EXISTS FOR (f:Fact)    ON (f.current);

// Point index for geofencing / nearest-place queries.
CREATE POINT INDEX location_point IF NOT EXISTS FOR (l:Location) ON (l.coordinate);

// ---------------------------------------------------------------------------
// 4. FULLTEXT INDEXES  (keyword RAG — "find the email about the budget task")
// ---------------------------------------------------------------------------
CREATE FULLTEXT INDEX entitySearch IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name, e.summary, e.aliases];

CREATE FULLTEXT INDEX contentSearch IF NOT EXISTS
FOR (n:Memory|Note|Observation) ON EACH [n.text];

CREATE FULLTEXT INDEX emailSearch IF NOT EXISTS
FOR (m:Email) ON EACH [m.subject, m.snippet, m.body];

// ---------------------------------------------------------------------------
// 5. VECTOR INDEX  (semantic recall — Neo4j >= 5.13. Set dims to your model:
// 768 = nomic-embed-text / mxbai; 1024 = bge-large; 1536 = OpenAI text-embed-3.)
// Uncomment once an embedding model is chosen and `embedding` is populated.
// ---------------------------------------------------------------------------
// CREATE VECTOR INDEX memoryEmbedding IF NOT EXISTS
// FOR (m:Memory) ON (m.embedding)
// OPTIONS { indexConfig: {
//   `vector.dimensions`: 768,
//   `vector.similarity_function`: 'cosine' } };
//
// CREATE VECTOR INDEX observationEmbedding IF NOT EXISTS
// FOR (o:Observation) ON (o.embedding)
// OPTIONS { indexConfig: {
//   `vector.dimensions`: 768,
//   `vector.similarity_function`: 'cosine' } };

// ---------------------------------------------------------------------------
// 6. SEED  (the singleton owner node — fill in real values via the app/MCP)
// ---------------------------------------------------------------------------
MERGE (u:Entity:User {key: 'user:self'})
  ON CREATE SET u.id = randomUUID(), u.type = 'user',
                u.created_at = timestamp()
SET u.updated_at = timestamp(), u.last_seen = timestamp();
