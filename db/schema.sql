-- =========================================================================
-- GraphRAG Docs Assistant — PostgreSQL schema
-- Conversation state, ingested-document metadata, and user feedback.
-- =========================================================================

CREATE TABLE IF NOT EXISTS chat_sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content     TEXT NOT NULL,
    citations   JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);

CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL PRIMARY KEY,
    source       TEXT NOT NULL UNIQUE,
    file_path    TEXT,
    num_chunks   INTEGER NOT NULL DEFAULT 0,
    num_nodes    INTEGER NOT NULL DEFAULT 0,
    num_edges    INTEGER NOT NULL DEFAULT 0,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback (
    id          BIGSERIAL PRIMARY KEY,
    message_id  BIGINT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    rating      SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    comment     TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feedback_message_id ON feedback(message_id);
